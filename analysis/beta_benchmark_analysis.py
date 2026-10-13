"""
Beta-power reference analysis for the GCE latent-axis paper.

Runs the exact same analyses used for latent Dim 2 / Dims-1+2 on FOOOF periodic
band power (low-beta, high-beta, low-gamma, finely-tuned gamma), as contextual
reference for readers.  Results are reported neutrally -- no verdicts.

Feature indices verified from analysis/captum_ig_analysis.py FEATURE_NAMES:
  index 1 -> power_low-beta           (13-20 Hz)
  index 2 -> power_high-beta          (21-30 Hz)
  index 3 -> power_low-gamma          (31-50 Hz)
  index 4 -> power_finely-tuned-gamma (55-90 Hz)
Single column per band (no hemispheric split; only PDC features have LR/RL pairs).

Analysis 1 -- LME (norm_total ~ band_wc + amp + C(condition), n=72)
Analysis 2 -- Spearman delta-beta vs clinical subscores (n=36, full amplitude)
Analysis 3 -- Per-frequency mean delta-beta summary with figure

Outputs (all in working_plots/beta_benchmark/):
  beta_benchmark_lme.csv
  beta_benchmark_spearman_n36.csv
  beta_benchmark_by_frequency.csv
  fig_beta_vs_auxiliary_by_frequency.png/.svg
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    }
)

# ── repository root ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config.paths import (
    FULLAMP_ALL_CONDITIONS_MAT,
    HALFAMP_ALL_CONDITIONS_MAT,
    SCORES_FULLAMP_XLSX,
    SCORES_HALFAMP_XLSX,
)

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "halfamp_mat": HALFAMP_ALL_CONDITIONS_MAT,
    "fullamp_mat": FULLAMP_ALL_CONDITIONS_MAT,
    "scores_halfamp": SCORES_HALFAMP_XLSX,
    "scores_fullamp": SCORES_FULLAMP_XLSX,
    "motor_df_csv": os.path.join(
        REPO_ROOT, "working_plots", "latent_axis_validation_results", "motor_df.csv"
    ),
    "fullamp_combined_csv": os.path.join(
        REPO_ROOT,
        "analysis",
        "fullamp_correlation_analysis",
        "fullamp_combined_data.csv",
    ),
    "perm_results_csv": os.path.join(
        REPO_ROOT,
        "working_plots",
        "latent_axis_validation_results",
        "permutation_results.csv",
    ),
    "output_dir": os.path.join(REPO_ROOT, "working_plots", "beta_benchmark"),
    "frequencies": [125, 110, 85, 55],
    "N_PERM": 5000,
    "SEEDS": [42, 2026, 4],  # matches permutation_test_lme.py
}

# =============================================================================
# BAND DEFINITIONS
# Indices are zero-based positions in the 70-feature vector.
# Source: analysis/captum_ig_analysis.py FEATURE_NAMES (lines 105-189).
# =============================================================================
BANDS = {
    "low_beta": {
        "feat_idx": 1,  # power_low-beta
        "label": "Low-beta (13-20 Hz)",
        "short": "Low-beta",
        "color": "#0072B2",  # blue (Wong)
    },
    "high_beta": {
        "feat_idx": 2,  # power_high-beta
        "label": "High-beta (21-30 Hz)",
        "short": "High-beta",
        "color": "#D55E00",  # vermilion (Wong)
    },
    "low_gamma": {
        "feat_idx": 3,  # power_low-gamma
        "label": "Low-gamma (31-50 Hz)",
        "short": "Low-gamma",
        "color": "#009E73",  # bluish green (Wong)
    },
    "finely_tuned_gamma": {
        "feat_idx": 4,  # power_finely-tuned-gamma
        "label": "Finely-tuned gamma (55-90 Hz)",
        "short": "FT-gamma",
        "color": "#E69F00",  # orange (Wong)
    },
}

AUX_DIST_COLOR = "#CC79A7"  # reddish purple (Wong) for auxiliary latent distance

# =============================================================================
# HELPERS
# =============================================================================


def patient_normalise(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Per-patient StandardScaler across all rows for that patient.

    Matches the normalization used in captum_ig_analysis.py and
    patient_specific_norm_hemi_fullamp.py (all conditions pooled per patient).
    """
    X_norm = np.zeros_like(X, dtype=np.float32)
    for pat in np.unique(subjects):
        mask = subjects == pat
        X_norm[mask] = (
            StandardScaler().fit_transform(X[mask]) if mask.sum() > 1 else X[mask]
        )
    return X_norm


def load_amp_data(mat_path: str, frequencies: list) -> tuple:
    """Load all stim conditions + OFF from one amplitude mat file.

    Normalization is applied to the full stacked matrix per patient so the
    scaler sees all conditions (including OFF) at once -- consistent with the
    training pipeline and captum analysis.

    Returns
    -------
    X_norm : (N, 70) float32  -- per-patient z-scored features
    subjects : (N,) int
    conditions : (N,) str -- '125', '110', '85', '55', 'off'
    """
    mat = loadmat(mat_path)
    X_list, idx_list, cond_list = [], [], []
    for freq in frequencies:
        X = mat[f"X_{freq}_cleaned"].astype(np.float32)
        idx = mat[f"idx_{freq}_cleaned"].flatten().astype(int)
        X_list.append(X)
        idx_list.append(idx)
        cond_list.extend([str(freq)] * len(X))
    X_off = mat["X_off_cleaned"].astype(np.float32)
    idx_off = mat["idx_off_cleaned"].flatten().astype(int)
    X_list.append(X_off)
    idx_list.append(idx_off)
    cond_list.extend(["off"] * len(X_off))

    X_all = np.vstack(X_list)
    subjects_all = np.hstack(idx_list)
    cond_all = np.array(cond_list)

    X_norm = patient_normalise(X_all, subjects_all)
    return X_norm, subjects_all, cond_all


def per_patient_condition_mean(
    X_norm: np.ndarray,
    subjects: np.ndarray,
    conditions: np.ndarray,
    feat_idx: int,
    target_conditions: list,
) -> dict:
    """Return {(patient, condition_str): mean_feature_value} for target conditions."""
    result = {}
    for cond in target_conditions:
        cond_mask = conditions == str(cond)
        for pat in np.unique(subjects[cond_mask]):
            mask = cond_mask & (subjects == pat)
            result[(int(pat), str(cond))] = float(X_norm[mask, feat_idx].mean())
    return result


# =============================================================================
# BUILD BETA MOTOR DF  (n=72, for LME)
# =============================================================================


def build_beta_motor_df(config: dict) -> pd.DataFrame:
    """
    Build a DataFrame with the same (patient, condition, amp, norm_total) index
    as motor_df.csv, adding per-patient within-centred band power columns.

    Normalization: all conditions for each amplitude dataset are stacked and
    normalised together per patient (captum-style, not per-condition).
    Within-patient centering: over all 8 rows per patient (4 freqs x 2 amps).
    """
    motor_ref = pd.read_csv(config["motor_df_csv"])
    # motor_ref has: patient, condition, amp, norm_total, dim_X_signed_wc ...

    all_rows = []
    for amp_label, mat_path in [
        ("half", config["halfamp_mat"]),
        ("full", config["fullamp_mat"]),
    ]:
        print(f"  Loading {amp_label}-amplitude data ...")
        X_norm, subjects, conditions = load_amp_data(mat_path, config["frequencies"])

        for band_name, band_info in BANDS.items():
            feat_idx = band_info["feat_idx"]
            feat_map = per_patient_condition_mean(
                X_norm, subjects, conditions, feat_idx, config["frequencies"]
            )
            for (pat, cond_str), val in feat_map.items():
                # match row in existing data structure
                all_rows.append(
                    {
                        "patient": pat,
                        "condition": int(cond_str),
                        "amp": amp_label,
                        "band": band_name,
                        "value": val,
                    }
                )

    feat_df = pd.DataFrame(all_rows)
    # pivot bands to columns
    feat_wide = feat_df.pivot_table(
        index=["patient", "condition", "amp"],
        columns="band",
        values="value",
        aggfunc="first",
    ).reset_index()
    feat_wide.columns.name = None

    # merge norm_total from motor_df
    df = motor_ref[["patient", "condition", "amp", "norm_total"]].merge(
        feat_wide, on=["patient", "condition", "amp"], how="inner"
    )

    # within-patient centering (across all 8 rows per patient, both amps)
    for band_name in BANDS:
        df[f"{band_name}_wc"] = df[band_name] - df.groupby("patient")[
            band_name
        ].transform("mean")

    print(f"  beta_motor_df: {len(df)} rows")
    assert len(df) == len(motor_ref), (
        f"Row count mismatch: got {len(df)}, expected {len(motor_ref)}"
    )
    return df


# =============================================================================
# BUILD BETA SPEARMAN DF  (n=36, full amplitude, OFF-referenced)
# =============================================================================


def build_beta_spearman_df(config: dict) -> pd.DataFrame:
    """
    Build a 36-row DataFrame (9 patients x 4 stim conditions, full amplitude).

    beta_delta = mean(band at stim_condition) - mean(band at OFF)
    using per-patient normalization over all conditions including OFF.

    Clinical scores and reference latent columns are pulled from
    fullamp_combined_data.csv (written by patient_specific_norm_hemi_fullamp.py).
    """
    print("  Loading full-amplitude data for Spearman analysis ...")
    X_norm, subjects, conditions = load_amp_data(
        config["fullamp_mat"], config["frequencies"]
    )

    rows = []
    for freq in config["frequencies"]:
        freq_str = str(freq)
        for pat in np.unique(subjects):
            stim_mask = (conditions == freq_str) & (subjects == pat)
            off_mask = (conditions == "off") & (subjects == pat)
            if not stim_mask.any() or not off_mask.any():
                continue
            row = {"patient": int(pat), "condition": freq}
            for band_name, band_info in BANDS.items():
                fi = band_info["feat_idx"]
                delta = (
                    float(X_norm[stim_mask, fi].mean())
                    - float(X_norm[off_mask, fi].mean())
                )
                row[f"delta_{band_name}"] = delta
            rows.append(row)

    beta_df = pd.DataFrame(rows)

    # merge clinical + latent reference columns
    ref = pd.read_csv(config["fullamp_combined_csv"])
    ref["condition"] = ref["condition"].astype(int)
    ref_cols = [
        "patient",
        "condition",
        "norm_total",
        "norm_rigidity",
        "norm_brady",
        "norm_tremor",
        "dim_2_signed",
        "dims_1_2_sum_signed",
    ]
    df = beta_df.merge(ref[ref_cols], on=["patient", "condition"], how="inner")
    print(f"  beta_spearman_df: {len(df)} rows")
    return df


# =============================================================================
# LME HELPER
# =============================================================================


def fit_lme(df: pd.DataFrame, predictor: str) -> tuple:
    """Fit LME: norm_total ~ predictor + amp + C(condition).  Returns (beta, p)."""
    formula = f"norm_total ~ {predictor} + amp + C(condition)"
    md = smf.mixedlm(formula, df, groups=df["patient"].astype(str))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = md.fit(reml=True, method="lbfgs")
    return float(res.params[predictor]), float(res.pvalues[predictor])


def phipson_smyth_p(null_arr: np.ndarray, obs_beta: float) -> float:
    """Two-sided permutation p with Phipson-Smyth +1 correction."""
    valid = null_arr[~np.isnan(null_arr)]
    count = int(np.sum(np.abs(valid) >= np.abs(obs_beta)))
    return (count + 1) / (len(valid) + 1)


def _perm_worker(args) -> tuple:
    """Worker: run N_PERM permutations for one (band_name, seed) pair.

    Sets OMP/MKL/OPENBLAS thread count to blas_threads before importing numpy
    so that BLAS does not oversubscribe across parallel workers.

    Returns (band_name, seed, null_array).
    Module-level so ProcessPoolExecutor can pickle it on all platforms.
    """
    (
        band_name, col_wc, df_vals, norm_total, amp, condition, patient,
        seed, N_PERM, blas_threads,
    ) = args

    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(blas_threads)

    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(seed)
    null = np.full(N_PERM, np.nan)

    df = pd.DataFrame(
        {
            col_wc: df_vals,
            "norm_total": norm_total,
            "amp": amp,
            "condition": condition,
            "patient": patient.astype(str),
        }
    )

    for i in range(N_PERM):
        d = df.copy()
        d[col_wc] = rng.permutation(df_vals)
        formula = f"norm_total ~ {col_wc} + amp + C(condition)"
        try:
            md = smf.mixedlm(formula, d, groups=d["patient"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = md.fit(reml=True, method="lbfgs")
            null[i] = float(res.params[col_wc])
        except Exception:
            pass

    return band_name, seed, null


# =============================================================================
# ANALYSIS 1 -- LME
# =============================================================================


def run_analysis_lme(beta_motor_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Fit observed LME for every band, then run permutation tests.

    All (band x seed) pairs are submitted to a single ProcessPoolExecutor so
    the full 12-task grid saturates as many cores as available.
    blas_threads = max(1, n_jobs_total // n_tasks) keeps total thread count
    at or below the available hardware threads.

    Appends Dim 2 reference row from permutation_results.csv.
    """
    N_PERM  = config["N_PERM"]
    seeds   = config["SEEDS"]
    n_jobs  = config.get("n_jobs", len(BANDS) * len(seeds))
    n_tasks = len(BANDS) * len(seeds)                # 12 independent tasks
    n_workers = min(n_jobs, n_tasks)
    # Distribute remaining cores as BLAS threads within each worker
    blas_threads = max(1, n_jobs // n_tasks)

    print(
        f"\n  Permutation pool: {n_workers} workers, "
        f"{blas_threads} BLAS thread(s)/worker, "
        f"{N_PERM} perms x {len(seeds)} seeds x {len(BANDS)} bands "
        f"= {N_PERM * len(seeds) * len(BANDS):,} total LME fits"
    )

    # ── Observed LME fits (fast, sequential) ─────────────────────────────────
    obs = {}
    for band_name, band_info in BANDS.items():
        col_wc = f"{band_name}_wc"
        beta, p = fit_lme(beta_motor_df, col_wc)
        obs[band_name] = (beta, p)
        print(f"  Observed  {band_info['short']:12s}: beta={beta:+.4f}  p={p:.4f}")

    # ── Build worker args for all 12 tasks ────────────────────────────────────
    norm_total = beta_motor_df["norm_total"].values
    amp        = beta_motor_df["amp"].values
    condition  = beta_motor_df["condition"].values
    patient    = beta_motor_df["patient"].values

    task_args = []
    for band_name in BANDS:
        col_wc  = f"{band_name}_wc"
        df_vals = beta_motor_df[col_wc].values.copy()
        obs_beta, _ = obs[band_name]
        for seed in seeds:
            task_args.append((
                band_name, col_wc, df_vals,
                norm_total, amp, condition, patient,
                seed, N_PERM, blas_threads,
            ))

    # ── Run all tasks in one pool ─────────────────────────────────────────────
    null_store: dict[str, dict[int, np.ndarray]] = {b: {} for b in BANDS}

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_perm_worker, args): args for args in task_args}
            for fut in as_completed(futures):
                band_name, seed_val, null = fut.result()
                null_store[band_name][seed_val] = null
                obs_beta, _ = obs[band_name]
                p_val = phipson_smyth_p(null, obs_beta)
                short = BANDS[band_name]["short"]
                print(
                    f"  Done  {short:12s}  seed={seed_val}: "
                    f"p_perm={p_val:.4f}  (n_valid={int(np.sum(~np.isnan(null)))})"
                )
    else:
        for args in task_args:
            band_name, seed_val, null = _perm_worker(args)
            null_store[band_name][seed_val] = null
            obs_beta, _ = obs[band_name]
            p_val = phipson_smyth_p(null, obs_beta)
            short = BANDS[band_name]["short"]
            print(
                f"  Done  {short:12s}  seed={seed_val}: "
                f"p_perm={p_val:.4f}  (n_valid={int(np.sum(~np.isnan(null)))})"
            )

    # ── Assemble result rows ──────────────────────────────────────────────────
    rows = []
    for band_name, band_info in BANDS.items():
        obs_beta, obs_p = obs[band_name]
        p_perm_list = []
        row = {
            "predictor": band_info["label"],
            "beta": obs_beta,
            "parametric_p": obs_p,
        }
        for seed in seeds:
            null = null_store[band_name][seed]
            p_val = phipson_smyth_p(null, obs_beta)
            row[f"p_perm_seed{seed}"] = p_val
            p_perm_list.append(p_val)
        row["p_perm_mean"] = float(np.mean(p_perm_list))
        row["p_perm_max_delta"] = float(max(p_perm_list) - min(p_perm_list))
        rows.append(row)

    # Dim 2 reference row from permutation_results.csv
    perm_ref = pd.read_csv(config["perm_results_csv"])
    dim2_row = perm_ref[perm_ref["test"] == "dim2_alone"].iloc[0]
    p_list = [
        dim2_row["p_perm_seed42"],
        dim2_row["p_perm_seed2026"],
        dim2_row["p_perm_seed4"],
    ]
    ref_row = {
        "predictor": "Dim 2 (latent, reference)",
        "beta": dim2_row["observed_beta"],
        "parametric_p": dim2_row["parametric_p"],
        "p_perm_seed42": dim2_row["p_perm_seed42"],
        "p_perm_seed2026": dim2_row["p_perm_seed2026"],
        "p_perm_seed4": dim2_row["p_perm_seed4"],
        "p_perm_mean": float(np.mean(p_list)),
        "p_perm_max_delta": float(max(p_list) - min(p_list)),
    }
    rows.append(ref_row)

    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS 2 -- Spearman (n=36)
# =============================================================================


def run_analysis_spearman(beta_spearman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation of OFF-referenced delta-beta vs clinical subscores.
    Appends reference rows for Dim 2 and dims-1-2 from the fullamp CSV.
    """
    clinical_cols = {
        "norm_total": "Total",
        "norm_rigidity": "Rigidity",
        "norm_brady": "Bradykinesia",
        "norm_tremor": "Tremor",
    }

    rows = []

    for band_name, band_info in BANDS.items():
        col = f"delta_{band_name}"
        row = {"predictor": band_info["label"]}
        for clin_col, clin_label in clinical_cols.items():
            valid = beta_spearman_df.dropna(subset=[col, clin_col])
            rho, p = spearmanr(valid[col], valid[clin_col])
            row[f"rho_{clin_label}"] = round(float(rho), 4)
            row[f"p_{clin_label}"] = round(float(p), 4)
        rows.append(row)

    # Reference rows from the fullamp CSV
    for ref_col, ref_label in [
        ("dim_2_signed", "Dim 2 (latent, reference)"),
        ("dims_1_2_sum_signed", "Dims 1+2 sum (latent, reference)"),
    ]:
        row = {"predictor": ref_label}
        for clin_col, clin_label in clinical_cols.items():
            valid = beta_spearman_df.dropna(subset=[ref_col, clin_col])
            rho, p = spearmanr(valid[ref_col], valid[clin_col])
            row[f"rho_{clin_label}"] = round(float(rho), 4)
            row[f"p_{clin_label}"] = round(float(p), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS 3 -- Per-frequency summary
# =============================================================================


def run_analysis_by_frequency(beta_spearman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-frequency (125/110/85/55): mean +/- SD of OFF-referenced delta-beta
    for each band, mean dims_1_2_sum_signed, mean norm_total.
    """
    freq_order = [125, 110, 85, 55]
    rows = []

    for freq in freq_order:
        sub = beta_spearman_df[beta_spearman_df["condition"] == freq]
        row = {"frequency_hz": freq, "n": len(sub)}
        row["mean_norm_total"] = round(float(sub["norm_total"].mean()), 4)
        row["sd_norm_total"] = round(float(sub["norm_total"].std(ddof=1)), 4)
        row["mean_dims12_signed"] = round(float(sub["dims_1_2_sum_signed"].mean()), 4)
        row["sd_dims12_signed"] = round(float(sub["dims_1_2_sum_signed"].std(ddof=1)), 4)
        for band_name, band_info in BANDS.items():
            col = f"delta_{band_name}"
            row[f"mean_{band_name}"] = round(float(sub[col].mean()), 4)
            row[f"sd_{band_name}"] = round(float(sub[col].std(ddof=1)), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FIGURE
# =============================================================================


def plot_beta_vs_auxiliary_by_frequency(freq_df: pd.DataFrame, output_dir: str):
    """
    Two panels sharing the x-axis (frequency order 125 -> 55):
      Panel A: per-frequency mean +/- SD of OFF-referenced delta-beta (one line per band)
      Panel B: per-frequency mean +/- SD of dims-1+2 auxiliary latent distance

    Wong palette throughout.  Neutral figure title.
    """
    freqs = freq_df["frequency_hz"].tolist()
    x = np.arange(len(freqs))
    freq_labels = [str(f) for f in freqs]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    # ── Panel A: delta-beta per band ──────────────────────────────────────────
    ax = axes[0]
    for band_name, band_info in BANDS.items():
        means = freq_df[f"mean_{band_name}"].values
        sds = freq_df[f"sd_{band_name}"].values
        ax.errorbar(
            x, means, yerr=sds,
            label=band_info["short"],
            color=band_info["color"],
            marker="o", markersize=5,
            linewidth=1.5, capsize=3,
        )
    ax.axhline(0, color="#AAAAAA", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Mean delta-beta (z-score units)", fontsize=11)
    ax.set_title(
        "A: OFF-referenced delta band power by stimulation frequency",
        fontsize=10, fontweight="bold", loc="left"
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: auxiliary latent distance ────────────────────────────────────
    ax = axes[1]
    means_aux = freq_df["mean_dims12_signed"].values
    sds_aux = freq_df["sd_dims12_signed"].values
    ax.errorbar(
        x, means_aux, yerr=sds_aux,
        label="Dims 1+2 (auxiliary latent)",
        color=AUX_DIST_COLOR,
        marker="s", markersize=5,
        linewidth=1.5, capsize=3,
    )
    ax.axhline(0, color="#AAAAAA", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Mean signed distance from OFF", fontsize=11)
    ax.set_title(
        "B: Auxiliary latent distance (Dims 1+2) by stimulation frequency",
        fontsize=10, fontweight="bold", loc="left"
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_xlabel("Stimulation frequency", fontsize=11)

    fig.suptitle(
        "FOOOF band power and auxiliary latent distance per stimulation frequency\n"
        "(mean +/- SD across n=9 patients, full amplitude, OFF-referenced)",
        fontsize=10,
    )

    plt.tight_layout(h_pad=3.0)
    base = os.path.join(output_dir, "fig_beta_vs_auxiliary_by_frequency")
    plt.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight", format="svg")
    plt.close()
    print(f"  Saved: {base}.png/.svg")


# =============================================================================
# PRINT SUMMARIES
# =============================================================================


def print_lme_table(lme_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("ANALYSIS 1 -- LME  (norm_total ~ band_wc + amp + C(condition), n=72)")
    print("=" * 70)
    cols = ["predictor", "beta", "parametric_p", "p_perm_mean", "p_perm_max_delta"]
    col_widths = [36, 8, 13, 13, 15]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, col_widths))
    print(header)
    print("-" * len(header))
    for _, row in lme_df.iterrows():
        vals = [
            str(row["predictor"])[:36],
            f"{row['beta']:+.3f}",
            f"{row['parametric_p']:.4f}",
            f"{row['p_perm_mean']:.4f}",
            f"{row['p_perm_max_delta']:.4f}",
        ]
        print("  ".join(f"{v:<{w}}" for v, w in zip(vals, col_widths)))


def print_spearman_table(sp_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("ANALYSIS 2 -- Spearman (OFF-referenced delta-beta, n=36, full amp)")
    print("=" * 70)
    subheadings = ["Total", "Rigidity", "Bradykinesia", "Tremor"]
    header = f"  {'Predictor':<36}" + "".join(f"  rho_{s:<6} p_{s:<8}" for s in subheadings)
    print(header)
    print("-" * (len(header) + 4))
    for _, row in sp_df.iterrows():
        line = f"  {str(row['predictor'])[:36]:<36}"
        for s in subheadings:
            rho = row.get(f"rho_{s}", float("nan"))
            p = row.get(f"p_{s}", float("nan"))
            line += f"  {rho:+.3f}    {p:.4f}  "
        print(line)


def print_frequency_table(freq_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("ANALYSIS 3 -- Per-frequency summary (mean +/- SD, n=9, full amp)")
    print("=" * 70)
    band_names = list(BANDS.keys())
    header = f"  {'Freq (Hz)':<10}  {'norm_total':<14}  {'dims12_signed':<16}" + \
             "".join(f"  {BANDS[b]['short']:<14}" for b in band_names)
    print(header)
    print("-" * (len(header) + 4))
    for _, row in freq_df.iterrows():
        line = (
            f"  {int(row['frequency_hz']):<10d}"
            f"  {row['mean_norm_total']:.3f}+/-{row['sd_norm_total']:.3f}  "
            f"  {row['mean_dims12_signed']:.3f}+/-{row['sd_dims12_signed']:.3f}  "
        )
        for b in band_names:
            line += f"  {row[f'mean_{b}']:.3f}+/-{row[f'sd_{b}']:.3f}  "
        print(line)


# =============================================================================
# MAIN
# =============================================================================


def main(config: dict = CONFIG, quick_n: int = 0):
    if quick_n > 0:
        config = {**config, "N_PERM": quick_n}
        print(f"[--quick mode] N_PERM overridden to {quick_n}")
    os.makedirs(config["output_dir"], exist_ok=True)

    print("\n" + "=" * 70)
    print("BETA BENCHMARK ANALYSIS")
    print("Feature indices (from captum_ig_analysis.py FEATURE_NAMES):")
    for band_name, band_info in BANDS.items():
        print(f"  index {band_info['feat_idx']} -> {band_info['label']}")
    print("=" * 70)

    # ── Build data frames ─────────────────────────────────────────────────────
    print("\n[1/3] Building beta motor df (LME, n=72) ...")
    beta_motor_df = build_beta_motor_df(config)

    print("\n[2/3] Building beta spearman df (n=36) ...")
    beta_spearman_df = build_beta_spearman_df(config)

    # ── Analysis 1: LME ───────────────────────────────────────────────────────
    print("\n[Running Analysis 1: LME + permutation tests]")
    lme_df = run_analysis_lme(beta_motor_df, config)
    lme_path = os.path.join(config["output_dir"], "beta_benchmark_lme.csv")
    lme_df.to_csv(lme_path, index=False)
    print(f"  Saved: {lme_path}")

    # ── Analysis 2: Spearman ──────────────────────────────────────────────────
    print("\n[Running Analysis 2: Spearman n=36]")
    spearman_df = run_analysis_spearman(beta_spearman_df)
    sp_path = os.path.join(config["output_dir"], "beta_benchmark_spearman_n36.csv")
    spearman_df.to_csv(sp_path, index=False)
    print(f"  Saved: {sp_path}")

    # ── Analysis 3: Per-frequency summary ─────────────────────────────────────
    print("\n[Running Analysis 3: Per-frequency summary]")
    freq_df = run_analysis_by_frequency(beta_spearman_df)
    freq_path = os.path.join(config["output_dir"], "beta_benchmark_by_frequency.csv")
    freq_df.to_csv(freq_path, index=False)
    print(f"  Saved: {freq_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\n[Generating figure]")
    plot_beta_vs_auxiliary_by_frequency(freq_df, config["output_dir"])

    # ── Print summaries ───────────────────────────────────────────────────────
    print_lme_table(lme_df)
    print_spearman_table(spearman_df)
    print_frequency_table(freq_df)

    print(f"\n{'=' * 70}")
    print(f"Done. Outputs in: {config['output_dir']}/")
    print(f"  beta_benchmark_lme.csv")
    print(f"  beta_benchmark_spearman_n36.csv")
    print(f"  beta_benchmark_by_frequency.csv")
    print(f"  fig_beta_vs_auxiliary_by_frequency.png/.svg")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Beta benchmark analysis")
    ap.add_argument(
        "--quick",
        type=int,
        default=0,
        metavar="N",
        help="Run N permutations instead of the full 5000 (for testing)",
    )
    import multiprocessing
    ap.add_argument(
        "--jobs",
        type=int,
        default=multiprocessing.cpu_count(),
        metavar="N",
        help="Parallel workers (default: all logical CPUs)",
    )
    args = ap.parse_args()
    cfg = {**CONFIG, "n_jobs": args.jobs}
    main(config=cfg, quick_n=args.quick)
