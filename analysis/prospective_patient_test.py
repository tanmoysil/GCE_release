"""
Prospective Patient Test (P11) — Fig. 4
=======================================
Zero-shot validation of the frozen encoder on a held-out patient (P11) who
was excluded from training, hyperparameter selection and analysis development.

For each amplitude (full, half) the patient's recordings are normalised with
one patient-specific scaler over all conditions, and the OFF-referenced
displacement Δz = mean µ(stim ON) − mean µ(OFF) is computed per frequency.

PRIMARY ENDPOINT (specified before the held-out outcomes were inspected)
  Spearman ρ between dims₁₂ (Δz₁ + Δz₂) and normalized rigidity improvement
  across 4 frequencies × 2 amplitudes (n = 8), expected POSITIVE as in the
  development cohort. Each amplitude is normalised separately, so both
  variables are centred within amplitude before pooling (analogous to the
  amplitude term in the mixed-effects model). Significance: one-sided exact
  permutation test, permuting rigidity within amplitude (4! × 4! = 576
  orderings).

EXPLORATORY
  All other metric × subscore correlations (combined and per amplitude).
  Dim 0 (stimulation axis): signed displacement from OFF and per-dimension
  ON/OFF AUC.
"""

import os
import sys
from itertools import permutations, product

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr, kendalltau
import dcor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gce.utils import select_vae_model
from config.paths import BEST_ENCODER_PT, P11_HALFAMP_MAT, P11_FULLAMP_MAT, P11_SCORES_CSV

CONFIG = {
    # ── encoder ───────────────────────────────────────────────────────────────
    "encoder_path": BEST_ENCODER_PT,
    "K":          1,
    "L":          5,
    "n_features": 70,

    # ── patient ───────────────────────────────────────────────────────────────
    "patient_id": 11,

    # ── data (X_{freq}_cleaned + X_off_cleaned, one mat per amplitude) ───────
    "patient_mat_fullamp": P11_FULLAMP_MAT,
    "patient_mat_halfamp": P11_HALFAMP_MAT,

    # ── motor scores (all subscores, all 4 frequencies) ───────────────────────
    # Read from P11_SCORES_CSV (not distributed; see Data availability).
    # Convention: (off − on) / off  →  higher = more improvement from OFF.
    "patient_scores_csv": P11_SCORES_CSV,

    # ── expected signs from the development cohort (full-amplitude, n=36) ──
    # dims12_signed (Δz₁+Δz₂) vs norm_rigidity: ρ=+0.434, p=0.008 → POSITIVE
    #   (primary endpoint direction)
    # dims12_signed (Δz₁+Δz₂) vs norm_total: ρ=+0.497, p=0.002  → POSITIVE
    # dim_1_signed:  ρ=+0.425, p=0.010                           → POSITIVE
    # dim_2_signed:  ρ=+0.405, p=0.014                           → POSITIVE
    # dim_0_signed:  ρ=−0.032, p=0.85  (negative control)        → ~ZERO
    # More positive (less negative) displacement → better therapy.
    "expected_sign_dims12": +1,
    "expected_sign_dim1":   +1,
    "expected_sign_dim2":   +1,

    # ── settings ──────────────────────────────────────────────────────────────
    "frequencies": [125, 110, 85, 55],

    # ── primary endpoint ──────────────────────────────────────────────────────
    "primary_metric":   "dims12",
    "primary_subscore": "rigidity",

    # ── output ────────────────────────────────────────────────────────────────
    "output_dir": os.path.join(REPO_ROOT, "working_plots", "prospective_results"),
}
# ─────────────────────────────────────────────────────────────────────────────


def load_patient_scores(config):
    """Fill patient_scores_{full,half}amp from the scores CSV.

    Expected columns: amp ('full'/'half'), frequency, total, rigidity, brady, tremor.
    """
    df = pd.read_csv(config["patient_scores_csv"])
    subscores = ["total", "rigidity", "brady", "tremor"]
    for amp in ("full", "half"):
        sub = df[df["amp"] == amp].set_index("frequency")
        config[f"patient_scores_{amp}amp"] = {
            ss: {int(f): float(sub.loc[f, ss]) for f in config["frequencies"]}
            for ss in subscores
        }
    config["patient_scores"] = config["patient_scores_fullamp"]
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_patient_data(config):
    """
    Returns dict: condition_str → X_norm  (n_segs × n_features, float32).
    Patient-specific StandardScaler fitted on all conditions pooled —
    mirrors get_latent_representations() in utils_results.py.
    """
    freqs = config["frequencies"]
    mat   = loadmat(config["patient_mat"])
    X_by_cond = {str(f): mat[f"X_{f}_cleaned"].astype(np.float32) for f in freqs}
    X_by_cond["off"] = mat["X_off_cleaned"].astype(np.float32)

    # Patient-specific normalisation: fit scaler on all conditions pooled
    X_pool  = np.concatenate(list(X_by_cond.values()), axis=0)
    scaler  = StandardScaler().fit(X_pool)
    X_by_cond = {k: scaler.transform(v) for k, v in X_by_cond.items()}

    segs = {k: v.shape[0] for k, v in X_by_cond.items()}
    print(f"Segments per condition: {segs}")
    return X_by_cond


# ─────────────────────────────────────────────────────────────────────────────
# Latent geometry — signed displacement from OFF
# ─────────────────────────────────────────────────────────────────────────────

def compute_latent_geometry(encoder, X_by_cond, config):
    """
    Returns:
      off_centroid  : (z_dim,)           — patient's resting-state latent mean
      cond_means    : dict freq_str → (z_dim,)   — mean µ per stim condition
      disp_from_off : dict freq_str → (z_dim,)   — Δz = z(cond) − z(OFF)
    """
    freqs = config["frequencies"]

    mu_by_cond   = {cond: get_mu(encoder, X) for cond, X in X_by_cond.items()}
    off_centroid = mu_by_cond["off"].mean(axis=0)

    cond_means    = {}
    disp_from_off = {}
    for f in freqs:
        fs = str(f)
        cond_means[fs]    = mu_by_cond[fs].mean(axis=0)
        disp_from_off[fs] = cond_means[fs] - off_centroid   # Δz = z(cond) − z(OFF)

    z_dim = config["K"] + config["L"]
    print(f"\nOFF centroid (dim 0–{z_dim-1}): {off_centroid.round(3)}")
    print(f"\nSigned displacement Δz = z(cond) − z(OFF):")
    print(f"{'freq':>5}  " + "  ".join(f"{'dim'+str(d):>7}" for d in range(z_dim)))
    for f in freqs:
        vals = "  ".join(f"{disp_from_off[str(f)][d]:+7.3f}" for d in range(z_dim))
        print(f"  {f:3d}    {vals}")

    return off_centroid, cond_means, disp_from_off


# ─────────────────────────────────────────────────────────────────────────────
# Secondary check — ON vs OFF AUC per dim
# ─────────────────────────────────────────────────────────────────────────────

def check_stimulation_axis(encoder, X_by_cond, config):
    """
    Verifies that Dim 0 (stimulation detection component) discriminates ON from
    OFF for this patient.  Single-dim logistic regression AUC, within-patient.
    Expected: Dim 0 has highest AUC (trained as stim axis).
    """
    freqs = config["frequencies"]
    z_dim = config["K"] + config["L"]

    mu_on  = np.concatenate([get_mu(encoder, X_by_cond[str(f)]) for f in freqs], axis=0)
    mu_off = get_mu(encoder, X_by_cond["off"])

    X_all = np.concatenate([mu_on, mu_off], axis=0)
    y_all = np.concatenate([np.ones(len(mu_on)), np.zeros(len(mu_off))])

    aucs     = {}
    mean_abs = {}
    off_centroid = mu_off.mean(axis=0)
    for d in range(z_dim):
        x_d  = X_all[:, d:d+1]
        x_ds = StandardScaler().fit_transform(x_d)
        clf  = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(x_ds, y_all)
        aucs[d] = float(roc_auc_score(y_all, clf.predict_proba(x_ds)[:, 1]))

        per_freq_disp = [abs(get_mu(encoder, X_by_cond[str(f)]).mean(axis=0)[d]
                            - off_centroid[d]) for f in freqs]
        mean_abs[d] = float(np.mean(per_freq_disp))

    best_auc_dim = max(aucs, key=aucs.get)
    dim0_ok      = (best_auc_dim == 0)

    print(f"\n── Secondary check: ON vs OFF AUC per dim ──")
    print(f"  n_on={len(mu_on)}  n_off={len(mu_off)}")
    print(f"  {'dim':>4}  {'AUC':>6}  {'|Δz|':>8}  note")
    print(f"  {'─'*45}")
    for d in range(z_dim):
        tag = ""
        if d == 0:
            tag = "← stimulation axis (expected highest AUC)"
        elif d == best_auc_dim:
            tag = "← ACTUAL highest AUC  ⚠"
        print(f"  dim{d}  {aucs[d]:.3f}  {mean_abs[d]:>8.4f}  {tag}")

    status = "✓ Dim 0 has highest AUC." if dim0_ok else \
             f"✗ Dim 0 not highest (dim{best_auc_dim} is, AUC={aucs[best_auc_dim]:.3f})."
    print(f"  {status}")

    return mean_abs, aucs, dim0_ok


# ─────────────────────────────────────────────────────────────────────────────
# Per-amplitude correlations (exploratory, n = 4)
# ─────────────────────────────────────────────────────────────────────────────

def run_per_amplitude_test(disp_from_off, config):
    """
    Exploratory. Spearman, Pearson, Kendall τ, and distance correlation between:
      - dims12_signed  (Δz₁ + Δz₂)  — combined auxiliary metric
      - dim_1_signed, dim_2_signed   — individual auxiliary axes
      - dim_0_signed                 — negative control
    vs. motor scores (total, rigidity, brady, tremor) across 4 frequencies.

    Expected signs (from training cohort full-amplitude, n=36):
      dims12, dim_1, dim_2: POSITIVE  (ρ ≈ +0.40–0.50)
      dim_0:                ~ZERO     (negative control)
    """
    freqs  = config["frequencies"]
    scores = config["patient_scores"]
    z_dim  = config["K"] + config["L"]

    # Build per-frequency arrays for each metric
    metrics = {}
    for d in range(z_dim):
        metrics[f"dim{d}"] = np.array([disp_from_off[str(f)][d] for f in freqs])
    metrics["dims12"]  = metrics["dim1"] + metrics["dim2"]
    metrics["dims1_5"] = sum(metrics[f"dim{d}"] for d in range(1, z_dim))

    exp_sign = {
        "dims12":  config["expected_sign_dims12"],
        "dims1_5": config["expected_sign_dims12"],   # same positive direction
        "dim1":    config["expected_sign_dim1"],
        "dim2":    config["expected_sign_dim2"],
        "dim0":    0,   # negative control — no expected sign
    }

    rows = []
    for metric_name, x in metrics.items():
        for subscore, score_dict in scores.items():
            y = np.array([score_dict[f] for f in freqs])

            rho_s, p_s = spearmanr(x, y)
            rho_p, p_p = pearsonr(x, y)
            tau,   p_k = kendalltau(x, y)
            dc         = float(dcor.distance_correlation(x, y))

            exp = exp_sign.get(metric_name, None)
            if exp and exp != 0:
                sign_ok = bool(np.sign(rho_s) == exp) if rho_s != 0 else False
            else:
                sign_ok = None   # neg control — no expected sign

            rows.append({
                "metric":     metric_name,
                "subscore":   subscore,
                "spearman":   float(rho_s),
                "p_spearman": float(p_s),
                "pearson":    float(rho_p),
                "p_pearson":  float(p_p),
                "kendall":    float(tau),
                "p_kendall":  float(p_k),
                "dcor":       dc,
                "sign_ok":    sign_ok,
            })

    df = pd.DataFrame(rows)

    subscores_list = list(scores.keys())

    print(f"\n── Per-amplitude correlations (n=4, exploratory) ──")
    print(f"  Training cohort (full-amp, n=36): dims12 ρ=+0.497; dims1_5 ρ=+0.394; "
          f"dim1 ρ=+0.425; dim2 ρ=+0.405")
    print(f"  Negative control dim0: ρ=−0.032  |  Min achievable p ≈ 0.083 at n=4")

    metric_order = ["dims12", "dims1_5", "dim1", "dim2", "dim3", "dim4", "dim5", "dim0"]

    for stat, stat_label in [("spearman", "Spearman ρ"), ("pearson", "Pearson r"),
                               ("kendall", "Kendall τ"),  ("dcor", "Dist. corr.")]:
        p_col = f"p_{stat}" if stat != "dcor" else None
        header = f"\n  {stat_label}\n"
        header += f"  {'metric':>8}  "
        for ss in subscores_list:
            header += f"{'ρ':>6} {'p':>5}  " if p_col else f"{'dc':>6}  "
        header = header.rstrip()
        # rebuild cleanly
        col_w = 14 if p_col else 8
        header = f"\n  {stat_label}\n  {'metric':>8}  " + \
                 "".join(f"{ss:^13}" for ss in subscores_list)
        print(header)
        print("  " + "─" * (10 + 13 * len(subscores_list)))

        for m in metric_order:
            row_data = df[df["metric"] == m]
            if row_data.empty:
                continue
            tag = " (neg ctrl)" if m == "dim0" else ""
            line = f"  {m:>8}{tag:<10}"
            for ss in subscores_list:
                r = row_data[row_data["subscore"] == ss].iloc[0]
                val = r[stat]
                if p_col:
                    p   = r[p_col]
                    sig = "*" if p < 0.083 else " "
                    line += f"  {val:+.3f} {p:.3f}{sig}"
                else:
                    line += f"  {val:.3f}      "
            # sign check indicator
            first_r = row_data[row_data["subscore"] == subscores_list[0]].iloc[0]
            if stat == "spearman" and first_r["sign_ok"] is False:
                line += "  ✗"
            print(line)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Combined amplitude analysis (n = 4 freqs × 2 amps = 8)
# ─────────────────────────────────────────────────────────────────────────────

def within_amp_permutation_p(x, y, amp):
    """
    One-sided exact permutation P for Spearman ρ(x, y) > 0.

    y is permuted within each amplitude level (every combination of
    within-level orderings; 4! × 4! = 576 for P11). Within-level permutation
    leaves each level's mean unchanged, so centred y stays centred.
    Returns (rho, p, n_permutations).
    """
    x, y, amp = np.asarray(x), np.asarray(y), np.asarray(amp)
    rho = spearmanr(x, y)[0]
    groups = [np.flatnonzero(amp == a) for a in pd.unique(amp)]
    null = []
    for orders in product(*(permutations(g) for g in groups)):
        y_perm = y.copy()
        for g, order in zip(groups, orders):
            y_perm[g] = y[list(order)]
        null.append(spearmanr(x, y_perm)[0])
    null = np.asarray(null)
    p = float(np.mean(null >= rho - 1e-12))
    return float(rho), p, len(null)


def run_combined_amp_test(disp_full, disp_half, config):
    """
    Pool full-amp and half-amp displacements (n=8) and test the primary endpoint.
    Each amplitude was normalised with its own patient-specific scaler, so the
    raw Δz values live on different numeric scales.  Before pooling, each metric
    and each clinical score is centred within its amplitude level — analogous
    to the amplitude term in the mixed-effects model.  This makes the rank
    ordering across the 8 pooled observations reflect within-amplitude relative
    position rather than between-amplitude offsets.

    Primary endpoint: dims12 vs rigidity, one-sided exact permutation test
    (within_amp_permutation_p). All other combinations are exploratory.
    """
    freqs   = config["frequencies"]
    z_dim   = config["K"] + config["L"]
    pid     = config["patient_id"]

    # Build combined arrays (full-amp rows first, then half-amp rows)
    scores_full = config["patient_scores_fullamp"]
    scores_half = config["patient_scores_halfamp"]

    # Within-amplitude demeaning — mirrors the LME 'amp' fixed-effect correction.
    # Each amp's displacements were computed with its own patient-specific scaler,
    # so raw values are on different scales.  Subtracting the within-amp mean of
    # each metric removes the amplitude-level offset before pooling.
    metrics_combined = {}
    for d in range(z_dim):
        arr_f = np.array([disp_full[str(f)][d] for f in freqs])
        arr_h = np.array([disp_half[str(f)][d] for f in freqs])
        # demean each amplitude separately
        arr_f_dm = arr_f - arr_f.mean()
        arr_h_dm = arr_h - arr_h.mean()
        metrics_combined[f"dim{d}"] = np.concatenate([arr_f_dm, arr_h_dm])
    metrics_combined["dims12"]  = metrics_combined["dim1"] + metrics_combined["dim2"]
    metrics_combined["dims1_5"] = sum(metrics_combined[f"dim{d}"] for d in range(1, z_dim))

    scores_combined = {}
    for ss in scores_full:
        arr_f = np.array([scores_full[ss][f] for f in freqs])
        arr_h = np.array([scores_half[ss][f] for f in freqs])
        # demean scores within each amplitude too
        arr_f_dm = arr_f - arr_f.mean()
        arr_h_dm = arr_h - arr_h.mean()
        scores_combined[ss] = np.concatenate([arr_f_dm, arr_h_dm])

    amp_labels = np.array(["full"] * len(freqs) + ["half"] * len(freqs))

    # ── primary endpoint ──────────────────────────────────────────────────────
    pm, ps = config["primary_metric"], config["primary_subscore"]
    rho_primary, p_primary, n_perm = within_amp_permutation_p(
        metrics_combined[pm], scores_combined[ps], amp_labels)

    exp_sign = {
        "dims12":  config["expected_sign_dims12"],
        "dims1_5": config["expected_sign_dims12"],
        "dim1":    config["expected_sign_dim1"],
        "dim2":    config["expected_sign_dim2"],
        "dim0":    0,
    }

    rows = []
    for metric_name, x in metrics_combined.items():
        for subscore, y in scores_combined.items():
            rho_s, p_s = spearmanr(x, y)
            rho_p, p_p = pearsonr(x, y)
            tau,   p_k = kendalltau(x, y)
            dc         = float(dcor.distance_correlation(x, y))
            exp = exp_sign.get(metric_name, None)
            sign_ok = (bool(np.sign(rho_s) == exp) if (exp and rho_s != 0) else
                       None if (not exp or exp == 0) else False)
            rows.append({
                "metric": metric_name, "subscore": subscore,
                "spearman": float(rho_s), "p_spearman": float(p_s),
                "pearson":  float(rho_p), "p_pearson":  float(p_p),
                "kendall":  float(tau),   "p_kendall":  float(p_k),
                "dcor":     dc,           "sign_ok":    sign_ok,
            })

    df = pd.DataFrame(rows)
    # The primary cell carries its pre-specified test, not the asymptotic P
    is_primary = (df["metric"] == pm) & (df["subscore"] == ps)
    df.loc[is_primary, "p_spearman"] = p_primary
    subscores_list = list(scores_combined.keys())
    n = len(amp_labels)

    print(f"\n{'=' * 60}")
    print(f"Combined amplitude test — Patient {pid}  (n={n}: {len(freqs)} full + {len(freqs)} half)")
    print("  Both variables centred within amplitude before pooling")
    print(f"{'=' * 60}")
    print(f"\n  PRIMARY ENDPOINT: {pm} vs {ps}")
    print(f"    Spearman ρ = {rho_primary:+.3f}   one-sided exact permutation "
          f"P = {p_primary:.3f}  ({n_perm} within-amplitude orderings)")
    print(f"    Criterion (ρ > 0, P < 0.05): "
          f"{'MET' if rho_primary > 0 and p_primary < 0.05 else 'NOT MET'}")
    print("\n  Exploratory (asymptotic P; * P < 0.05; † primary endpoint, "
          "one-sided exact permutation P):")

    metric_order = ["dims12", "dims1_5", "dim1", "dim2", "dim3", "dim4", "dim5", "dim0"]
    for stat, stat_label in [("spearman", "Spearman ρ"), ("pearson", "Pearson r"),
                               ("kendall", "Kendall τ"),  ("dcor", "Dist. corr.")]:
        p_col = f"p_{stat}" if stat != "dcor" else None
        header = f"\n  {stat_label}\n  {'metric':>8}  " + \
                 "".join(f"{ss:^13}" for ss in subscores_list)
        print(header)
        print("  " + "─" * (10 + 13 * len(subscores_list)))
        for m in metric_order:
            row_data = df[df["metric"] == m]
            if row_data.empty:
                continue
            tag = " (neg ctrl)" if m == "dim0" else ""
            line = f"  {m:>8}{tag:<10}"
            for ss in subscores_list:
                r = row_data[row_data["subscore"] == ss].iloc[0]
                val = r[stat]
                if p_col:
                    p   = r[p_col]
                    sig = "*" if p < 0.05 else " "
                    if stat == "spearman" and m == pm and ss == ps:
                        sig = "†"
                    line += f"  {val:+.3f} {p:.3f}{sig}"
                else:
                    line += f"  {val:.3f}      "
            first_r = row_data[row_data["subscore"] == subscores_list[0]].iloc[0]
            if stat == "spearman" and first_r["sign_ok"] is False:
                line += "  ✗"
            print(line)

    # Save
    out = config["output_dir"]
    df.to_csv(os.path.join(out, f"combined_amp_test_pat{pid}.csv"), index=False)

    # Also save the 8-row displacement table
    pool_rows = []
    for amp_tag, disp, scores in [("full", disp_full, scores_full),
                                   ("half", disp_half, scores_half)]:
        for f in freqs:
            row = {"amp": amp_tag, "freq": f}
            for d in range(z_dim):
                row[f"dim{d}_signed"] = float(disp[str(f)][d])
            row["dims12_signed"] = float(disp[str(f)][1] + disp[str(f)][2])
            for ss, sd in scores.items():
                row[f"score_{ss}"] = sd[f]
            pool_rows.append(row)
    pool_df = pd.DataFrame(pool_rows)
    pool_df.to_csv(os.path.join(out, f"displacements_combined_pat{pid}.csv"), index=False)
    print(f"\n  Saved combined tables → {out}/")

    return df, {"metric": pm, "subscore": ps, "rho": rho_primary,
                "p_one_sided": p_primary, "n_permutations": n_perm}


def run_both_amps_combined(config=CONFIG):
    """Run full-amp, half-amp, and combined (n=8) analyses for one patient."""
    encoder    = load_encoder(config)
    os.makedirs(config["output_dir"], exist_ok=True)

    # ── Full amplitude ────────────────────────────────────────────────────────
    cfg_full = dict(config)
    cfg_full["amp"]            = "full"
    cfg_full["patient_mat"]    = config["patient_mat_fullamp"]
    cfg_full["patient_scores"] = config["patient_scores_fullamp"]

    print(f"\n{'=' * 60}")
    print(f"Prospective Patient Test — Patient {config['patient_id']}  (amp=full)")
    print(f"{'=' * 60}")
    X_full = load_patient_data(cfg_full)
    _, _, disp_full = compute_latent_geometry(encoder, X_full, cfg_full)
    _, _, dim0_full = check_stimulation_axis(encoder, X_full, cfg_full)
    res_full = run_per_amplitude_test(disp_full, cfg_full)
    res_full.to_csv(
        os.path.join(config["output_dir"],
                     f"per_amplitude_test_pat{config['patient_id']}_fullamp.csv"), index=False)

    # ── Half amplitude ────────────────────────────────────────────────────────
    cfg_half = dict(config)
    cfg_half["amp"]            = "half"
    cfg_half["patient_mat"]    = config["patient_mat_halfamp"]
    cfg_half["patient_scores"] = config["patient_scores_halfamp"]

    print(f"\n{'=' * 60}")
    print(f"Prospective Patient Test — Patient {config['patient_id']}  (amp=half)")
    print(f"{'=' * 60}")
    X_half = load_patient_data(cfg_half)
    _, _, disp_half = compute_latent_geometry(encoder, X_half, cfg_half)
    _, _, dim0_half = check_stimulation_axis(encoder, X_half, cfg_half)
    res_half = run_per_amplitude_test(disp_half, cfg_half)
    res_half.to_csv(
        os.path.join(config["output_dir"],
                     f"per_amplitude_test_pat{config['patient_id']}_halfamp.csv"), index=False)

    # ── Combined (n = 8) ──────────────────────────────────────────────────────
    combined_df, primary = run_combined_amp_test(disp_full, disp_half, config)

    print(f"\nAll done — results in {config['output_dir']}/")
    return {
        "disp_full":   disp_full,  "res_full":   res_full,   "dim0_ok_full": dim0_full,
        "disp_half":   disp_half,  "res_half":   res_half,   "dim0_ok_half": dim0_half,
        "combined_df": combined_df, "primary": primary,
    }


if __name__ == "__main__":
    run_both_amps_combined(load_patient_scores(CONFIG))
