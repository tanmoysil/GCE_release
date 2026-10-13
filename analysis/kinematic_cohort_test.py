"""
Kinematic Cohort Test — Fig. 5 and Table S8
===========================================
Zero-shot validation of the frozen GCE encoder on three patients from a
separate study at the same center (K1–K3; recorded as subject IDs
LON03–LON05 in the data files), none seen during training.

DATA (paths in config/paths.py)
  LFP features : KINEMATIC_MAT  (HDF5 / MATLAB v7.3)
    Keys: X_{85|125}Hz_{off|cond1|cond2|cond3}_cleaned  shape (70, n_samples)
    idx_*  subject index  1, 2, 3
    amp_*  stimulation amplitude in mA per LFP segment

  Kinematic scores (already Δ relative to per-subject STIMOFF baseline):
    KINEMATIC_FT_DELTA_CSV (finger tapping), KINEMATIC_HM_DELTA_CSV (hand movement)

ANALYSIS
  For each (subject, frequency, amplitude-condition):
    Δz_d = mean_µ(stim ON)_d − mean_µ(STIMOFF)_d   [frozen encoder]
  Correlate dims12 = Δz₁ + Δz₂  (therapeutic subspace) against ΔPC scores.

KINEMATIC COMPONENTS
  Movement Speed (PC1) and Performance Consistency (PC2) for finger tapping
  (FT) and hand movement (HM), the components with a defined direction of
  levodopa-related improvement. PC3 is not analysed (see Methods).
  DBS improvement → FT-PC1↑  FT-PC2↓  HM-PC1↓  HM-PC2↑
  Expected: ρ(dims12, FT-PC1)>0  ρ(dims12, FT-PC2)<0
            ρ(dims12, HM-PC1)<0  ρ(dims12, HM-PC2)>0
  Each recording is paired with the hand contralateral to the stimulated STN.

OUTPUT (working_plots/kinematic_cohort_results/, read by paper_figures/generate_figure5.py)
  displacements_contra.csv                    all recordings (N = 14)
  displacements_contra_cond1+cond3.csv        balanced subset, 4 per patient (N = 12)

AMPLITUDE-CONDITION JOIN
  condN in the mat groups LFP segments by amplitude tier
  (lowest/middle/highest for that subject's electrode).
  Each condN can cover multiple mA values if left/right STN use different
  amplitudes at the same tier.  CSV rows are matched by active stimulation
  amplitude (the ON-side value) ∈ mat amp set for that condN.
"""

import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau
import dcor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gce.utils import select_vae_model
from config.paths import (BEST_ENCODER_PT, KINEMATIC_MAT, KINEMATIC_FT_DELTA_CSV,
                          KINEMATIC_HM_DELTA_CSV)


# ── constants ────────────────────────────────────────────────────────────────
SUBJECT_MAP = {1: "LON03", 2: "LON04", 3: "LON05"}  # subject IDs in the data files
FREQS       = [85, 125]
CONDS       = ["cond1", "cond2", "cond3"]

CONFIG = {
    "encoder_path": BEST_ENCODER_PT,
    "K":            1,
    "L":            5,
    "n_features":   70,
    "mat_path":     KINEMATIC_MAT,
    "ft_delta_csv": KINEMATIC_FT_DELTA_CSV,
    "hm_delta_csv": KINEMATIC_HM_DELTA_CSV,
    "output_dir":   os.path.join(REPO_ROOT, "working_plots", "kinematic_cohort_results"),
    # Balanced subset: conditions with a kinematic match for every patient
    "balanced_conds": ["cond1", "cond3"],
    # expected sign of ρ(dims12, ΔPC)
    "expected_signs": {
        "FT_PC1": +1, "FT_PC2": -1,
        "HM_PC1": -1, "HM_PC2": +1,
    },
}

# ── encoder ──────────────────────────────────────────────────────────────────

def load_encoder(config):
    base_cfg = {
        "vae_model":       "vanilla_Encoder",
        "n_features":      config["n_features"],
        "n_alpha":         config["K"],
        "n_beta":          config["L"],
        "z_dim":           config["K"] + config["L"],
        "include_classes": None,
        "save_dir":        "./",
        "model_name":      "vae_model",
    }
    enc, _ = select_vae_model(base_cfg)
    enc.load_state_dict(torch.load(config["encoder_path"], map_location="cpu"))
    enc.eval()
    print(f"Encoder: K={config['K']}  L={config['L']}  "
          f"z_dim={config['K']+config['L']}  {os.path.basename(config['encoder_path'])}")
    return enc


def get_mu(encoder, X):
    with torch.no_grad():
        _, mu, _ = encoder(torch.FloatTensor(X))
    return mu.numpy()


# ── data loading ─────────────────────────────────────────────────────────────

def _hdf5_array(f, key, squeeze=False):
    """Read a dataset from the h5py file handle.
    Feature matrices (70 × n) are transposed to (n × 70).
    1-row index/amplitude arrays are squeezed to 1-D.
    """
    arr = f[key][:]
    if arr.ndim == 2 and arr.shape[0] == 1:   # (1, n) index/amp arrays
        return arr.squeeze()
    if arr.ndim == 2:                           # (70, n) feature matrices
        return arr.T
    return arr


def load_mat_data(config):
    """
    Returns nested dict:
      raw[subj_idx][freq]["off"]      → X  (n_samples, 70)
      raw[subj_idx][freq]["condN"]    → X
      amps[subj_idx][freq]["condN"]   → sorted array of unique mA values
    Segments are NOT yet normalised.
    """
    path = config["mat_path"]
    raw  = {s: {f: {} for f in FREQS} for s in SUBJECT_MAP}
    amps = {s: {f: {} for f in FREQS} for s in SUBJECT_MAP}

    with h5py.File(path, "r") as f:
        for freq in FREQS:
            freq_str = f"{freq}Hz"

            # OFF condition
            X_off   = _hdf5_array(f, f"X_{freq_str}_off_cleaned")    # (n, 70)
            idx_off = _hdf5_array(f, f"idx_{freq_str}_off_cleaned").astype(int)  # (n,)
            for s in SUBJECT_MAP:
                mask = idx_off == s
                raw[s][freq]["off"] = X_off[mask]

            # ON conditions
            for cond in CONDS:
                key_x   = f"X_{freq_str}_{cond}_cleaned"
                key_idx = f"idx_{freq_str}_{cond}_cleaned"
                key_amp = f"amp_{freq_str}_{cond}_cleaned"
                if key_x not in f:
                    continue
                X_on   = _hdf5_array(f, key_x)                        # (n, 70)
                idx_on = _hdf5_array(f, key_idx).astype(int)          # (n,)
                amp_on = _hdf5_array(f, key_amp)                      # (n,)
                for s in SUBJECT_MAP:
                    mask = idx_on == s
                    if not mask.any():
                        continue
                    raw[s][freq][cond]  = X_on[mask]
                    amps[s][freq][cond] = np.unique(amp_on[mask])

    return raw, amps


def normalise_per_subject(raw):
    """
    Fit one StandardScaler per subject on ALL their segments (both freqs,
    all conds + off) pooled, then return normalised copy.
    Mirrors get_pd_dataloaders_with_save / P11 patient-specific scaler.
    """
    normed = {s: {f: {} for f in FREQS} for s in SUBJECT_MAP}
    for s in SUBJECT_MAP:
        pool = []
        for f in FREQS:
            for cond, X in raw[s][f].items():
                if X.size:
                    pool.append(X)
        scaler = StandardScaler().fit(np.concatenate(pool, axis=0))
        for f in FREQS:
            for cond, X in raw[s][f].items():
                normed[s][f][cond] = scaler.transform(X)
    return normed


# ── latent geometry ───────────────────────────────────────────────────────────

def compute_displacements(encoder, normed, amps):
    """
    Returns:
      displacements[(subj_idx, freq, cond)] = Δz  (z_dim,)
      off_centroids[(subj_idx, freq)]        = µ_OFF centroid
    """
    z_dim = CONFIG["K"] + CONFIG["L"]
    displacements = {}
    off_centroids = {}

    for s in SUBJECT_MAP:
        for f in FREQS:
            X_off = normed[s][f].get("off")
            if X_off is None or len(X_off) == 0:
                continue
            mu_off = get_mu(encoder, X_off).mean(axis=0)
            off_centroids[(s, f)] = mu_off

            for cond in CONDS:
                X_on = normed[s][f].get(cond)
                if X_on is None or len(X_on) == 0:
                    continue
                mu_on = get_mu(encoder, X_on).mean(axis=0)
                displacements[(s, f, cond)] = mu_on - mu_off

    # ── dim0 sign normalisation ───────────────────────────────────────────────
    # Dim0 polarity is sign-arbitrary: the encoder learns ON/OFF separation but
    # not which direction is "ON".  Convention: positive = stimulation onset.
    # For each subject, if the mean dim0 displacement across all their ON
    # conditions is negative, flip dim0 so the convention is consistent.
    dim0_signs = {}
    for s in SUBJECT_MAP:
        vals = [displacements[(s, f, c)][0]
                for f in FREQS for c in CONDS if (s, f, c) in displacements]
        sign = +1 if np.mean(vals) >= 0 else -1
        dim0_signs[s] = sign
        if sign == -1:
            print(f"  [dim0 sign flip] {SUBJECT_MAP[s]}: mean dim0 = {np.mean(vals):+.3f} → flipping")
        for f in FREQS:
            for c in CONDS:
                if (s, f, c) in displacements:
                    displacements[(s, f, c)][0] *= sign

    print(f"\nDisplacement table  (Δz = z(ON) − z(OFF), dim0 sign-normalised):")
    header = f"  {'subject':>6}  {'freq':>5}  {'cond':>5}  {'amps':>12}  " + \
             "  ".join(f"{'dim'+str(d):>7}" for d in range(z_dim))
    print(header)
    for (s, f, cond), dz in sorted(displacements.items()):
        amp_str = str(amps[s][f].get(cond, []))
        vals    = "  ".join(f"{dz[d]:+7.3f}" for d in range(z_dim))
        print(f"  {SUBJECT_MAP[s]:>6}  {f:>5}  {cond:>5}  {amp_str:>12}  {vals}")

    return displacements, off_centroids, dim0_signs


# ── kinematic score loading & join ────────────────────────────────────────────

def _active_amp(row):
    """Return the ON-side stimulation amplitude for a CSV row."""
    if row["Stim_L_State"] == "ON":
        return row["Stim_L_Amp_mA"]
    if row["Stim_R_State"] == "ON":
        return row["Stim_R_Amp_mA"]
    return 0.0


def load_kinematic_scores(config):
    """
    Load delta PC CSVs, compute active_amp per row.
    Returns (ft_df, hm_df) restricted to P3/P4/P5 subjects.
    """
    subjects = set(SUBJECT_MAP.values())
    ft = pd.read_csv(config["ft_delta_csv"])
    hm = pd.read_csv(config["hm_delta_csv"])
    ft = ft[ft["Subject"].isin(subjects)].copy()
    hm = hm[hm["Subject"].isin(subjects)].copy()
    ft["active_amp"] = ft.apply(_active_amp, axis=1)
    hm["active_amp"] = hm.apply(_active_amp, axis=1)
    return ft, hm


def match_kinematic(subj_name, freq, mat_amps, ft_df, hm_df):
    """
    Find CSV rows matching (subject, frequency, amplitude ∈ mat_amps).

    Keeps only the therapeutically relevant hand per
    stimulation session — Right hand when L-STN is ON, Left hand when R-STN is ON.
    This is correct because each mat segment comes from the stimulated hemisphere,
    and the motor effect is contralateral to that hemisphere.

    When a patient has left- and right-STN sessions at the same amplitude, the
    mat segments for that amplitude are a blend of both hemispheres.
    Contralateral filtering still removes ipsilateral rows from the CSV average,
    which is the best we can do without per-hemisphere hemisphere labels in the mat.

    Returns None if no rows match.
    """
    amp_set = set(np.round(mat_amps, 4))

    def _mean_pcs(df, task):
        mask = (
            (df["Subject"]      == subj_name) &
            (df["Stim_Freq_Hz"] == freq) &
            (df["active_amp"].round(4).isin(amp_set))
        )
        contra = (
            ((df["Stim_L_State"] == "ON") & (df["Analyzed_Hand"] == "Right")) |
            ((df["Stim_R_State"] == "ON") & (df["Analyzed_Hand"] == "Left"))
        )
        mask = mask & contra
        rows = df[mask]
        if rows.empty:
            return None
        return {f"{task}_PC{i}": rows[f"{task}_PC{i}"].mean() for i in (1, 2)}

    ft_scores = _mean_pcs(ft_df, "FingerTap")
    hm_scores = _mean_pcs(hm_df, "HandMovement")
    return ft_scores, hm_scores


# ── analysis table ────────────────────────────────────────────────────────────

def build_analysis_table(displacements, amps, ft_df, hm_df, config, conds=None):
    """
    Build one row per (subject, freq, cond) with latent Δz and ΔPC scores.
    conds: restrict to these amplitude conditions (None → all).
    Rows with no kinematic match are logged and dropped.
    """
    z_dim = config["K"] + config["L"]
    rows  = []

    for (s, f, cond), dz in sorted(displacements.items()):
        if conds is not None and cond not in conds:
            continue
        subj_name = SUBJECT_MAP[s]
        mat_amps  = amps[s][f].get(cond, np.array([]))

        ft_scores, hm_scores = match_kinematic(subj_name, f, mat_amps, ft_df, hm_df)

        if ft_scores is None and hm_scores is None:
            warnings.warn(
                f"No kinematic match for {subj_name} freq={f} {cond} amps={mat_amps}"
            )
            continue

        row = {
            "subject":     subj_name,
            "subject_idx": s,
            "freq":        f,
            "cond":        cond,
            "mat_amps":    str(mat_amps.tolist()),
        }
        for d in range(z_dim):
            row[f"dim{d}_signed"] = float(dz[d])
        row["dims12_signed"] = float(dz[1] + dz[2])

        if ft_scores:
            row.update({k.replace("FingerTap_", "FT_"): v for k, v in ft_scores.items()})
        if hm_scores:
            row.update({k.replace("HandMovement_", "HM_"): v for k, v in hm_scores.items()})

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nAnalysis table: {len(df)} rows  "
          f"({df['subject'].nunique()} subjects × ~6 freq×cond combinations)")
    return df


# ── correlations ──────────────────────────────────────────────────────────────

KINEMATIC_SCORES = ["FT_PC1", "FT_PC2", "HM_PC1", "HM_PC2"]

LATENT_METRICS = {
    "dims12":  lambda df: df["dim1_signed"] + df["dim2_signed"],
    "dim1":    lambda df: df["dim1_signed"],
    "dim2":    lambda df: df["dim2_signed"],
    "dim0":    lambda df: df["dim0_signed"],
}


def _correlate(x, y):
    rho_s, p_s = spearmanr(x, y)
    rho_p, p_p = pearsonr(x, y)
    tau,   p_k = kendalltau(x, y)
    dc         = float(dcor.distance_correlation(x, y))
    return {"spearman": float(rho_s), "p_spearman": float(p_s),
            "pearson":  float(rho_p), "p_pearson":  float(p_p),
            "kendall":  float(tau),   "p_kendall":  float(p_k),
            "dcor":     dc}


def run_correlations(df, config):
    """
    Compute correlations per subject (n≈6) and pooled (n≈18).
    """
    expected = config["expected_signs"]
    rows     = []

    for metric_name, metric_fn in LATENT_METRICS.items():
        for score in KINEMATIC_SCORES:
            if score not in df.columns:
                continue

            # ── per subject ──────────────────────────────────────────────────
            for subj in sorted(df["subject"].unique()):
                sub  = df[df["subject"] == subj].dropna(subset=[score])
                if len(sub) < 3:
                    continue
                x = metric_fn(sub).values
                y = sub[score].values
                r = _correlate(x, y)
                exp = expected.get(score, None)
                r.update({
                    "metric": metric_name, "score": score,
                    "scope": subj, "n": len(sub),
                    "sign_ok": bool(np.sign(r["spearman"]) == exp) if exp else None,
                })
                rows.append(r)

            # ── pooled ───────────────────────────────────────────────────────
            pool = df.dropna(subset=[score])
            if len(pool) < 3:
                continue
            x = metric_fn(pool).values
            y = pool[score].values
            r = _correlate(x, y)
            exp = expected.get(score, None)
            r.update({
                "metric": metric_name, "score": score,
                "scope": "pooled", "n": len(pool),
                "sign_ok": bool(np.sign(r["spearman"]) == exp) if exp else None,
            })
            rows.append(r)

    return pd.DataFrame(rows)


def print_correlation_table(results_df):
    print("\n══════════════════════════════════════════════════════")
    print(" Spearman ρ   [* p<0.05  ** p<0.01]")
    print("══════════════════════════════════════════════════════")

    for scope in ["pooled"] + sorted(s for s in results_df["scope"].unique()
                                     if s != "pooled"):
        sub = results_df[results_df["scope"] == scope]
        if sub.empty:
            continue
        n = sub["n"].iloc[0]
        print(f"\n── {scope}  (n={n}) ──")
        active_scores = [s for s in KINEMATIC_SCORES if s in sub["score"].values]
        header = f"  {'metric':>8}  " + "".join(f"{s:^10}" for s in active_scores)
        print(header)
        print("  " + "─" * (10 + 10 * len(KINEMATIC_SCORES)))

        metric_order = ["dims12", "dim1", "dim2", "dim0"]
        for m in metric_order:
            row_data = sub[sub["metric"] == m]
            if row_data.empty:
                continue
            tag = " (neg ctrl)" if m == "dim0" else ""
            line = f"  {m:>8}{tag:<10}"
            for score in active_scores:
                if score not in row_data["score"].values:
                    continue
                r   = row_data[row_data["score"] == score].iloc[0]
                rho = r["spearman"]
                p   = r["p_spearman"]
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
                line += f"  {rho:+.2f}{sig:<2}"
            print(line)


# ── stimulation-axis check ────────────────────────────────────────────────────

def check_stim_axis(encoder, normed, config, dim0_signs=None):
    """
    Per-subject ON vs OFF AUC for each latent dim.
    Confirms dim0 is the stimulation-detection axis.
    dim0_signs: dict {subj_idx: +1|-1} from compute_displacements sign normalisation.
    """
    z_dim = config["K"] + config["L"]
    if dim0_signs is None:
        dim0_signs = {s: +1 for s in SUBJECT_MAP}

    print("\n── Secondary check: ON vs OFF AUC per dim (per subject) ──")
    print(f"  {'subject':>6}  " + "  ".join(f"{'dim'+str(d):>7}" for d in range(z_dim))
          + "   best_dim  dim0_sign")
    print("  " + "─" * (8 + 10 * z_dim + 20))

    results = {}
    for s in SUBJECT_MAP:
        mu_off = get_mu(encoder, np.concatenate(
            [normed[s][f]["off"] for f in FREQS], axis=0))
        mu_on  = get_mu(encoder, np.concatenate(
            [normed[s][f][c]
             for f in FREQS for c in CONDS
             if c in normed[s][f]], axis=0))

        X_all = np.concatenate([mu_on, mu_off], axis=0)
        y_all = np.concatenate([np.ones(len(mu_on)), np.zeros(len(mu_off))])

        aucs = {}
        for d in range(z_dim):
            x_d  = StandardScaler().fit_transform(X_all[:, d:d+1])
            clf  = LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(x_d, y_all)
            aucs[d] = float(roc_auc_score(y_all, clf.predict_proba(x_d)[:, 1]))

        best     = max(aucs, key=aucs.get)
        auc_str  = "  ".join(f"{aucs[d]:.3f}" for d in range(z_dim))
        flag     = "✓" if best == 0 else f"✗ (best=dim{best})"
        sign_str = f"{'flipped' if dim0_signs[s] == -1 else 'natural'}"
        print(f"  {SUBJECT_MAP[s]:>6}  {auc_str}   {flag}   {sign_str}")
        results[s] = aucs

    return results


def main(config=CONFIG):
    os.makedirs(config["output_dir"], exist_ok=True)

    print("Loading encoder ...")
    encoder = load_encoder(config)

    print("\nLoading LFP mat file ...")
    raw, amps = load_mat_data(config)
    normed    = normalise_per_subject(raw)

    for s in SUBJECT_MAP:
        segs = {f"{SUBJECT_MAP[s]} {f} {c}": normed[s][f].get(c, np.array([])).shape[0]
                for f in FREQS for c in (["off"] + CONDS)}
        print("  " + "  ".join(f"{k}={v}" for k, v in segs.items() if v > 0))

    print("\nComputing latent displacements ...")
    displacements, _, dim0_signs = compute_displacements(encoder, normed, amps)

    print("\nSecondary check — stim-axis AUC ...")
    check_stim_axis(encoder, normed, config, dim0_signs)

    print("\nLoading kinematic scores ...")
    ft_df, hm_df = load_kinematic_scores(config)

    out = config["output_dir"]
    analyses = [("all recordings", None, "contra"),
                ("balanced subset", config["balanced_conds"],
                 "contra_" + "+".join(config["balanced_conds"]))]
    for label, conds, tag in analyses:
        print(f"\n{'=' * 60}\n{label.upper()}\n{'=' * 60}")
        df = build_analysis_table(displacements, amps, ft_df, hm_df, config, conds=conds)
        results_df = run_correlations(df, config)
        print_correlation_table(results_df)
        df.to_csv(os.path.join(out, f"displacements_{tag}.csv"), index=False)
        results_df.to_csv(os.path.join(out, f"correlations_{tag}.csv"), index=False)

    print(f"\nDone — results in {out}/")


if __name__ == "__main__":
    main()
