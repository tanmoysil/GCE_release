"""
Latent axis validation — inputs and mixed-effects models for Table S4 (total score)
==================================================================================
Builds the 72-row table (9 patients × 4 frequencies × 2 amplitudes) of per-condition
mean latent values, centred within patient, and fits

    norm_total ~ dim_X_wc + amp + C(frequency) + (1 | patient)

for Dim 0 (negative control), Dim 1, Dim 2 and the joint Dim 1 + Dim 2 model.

Outputs (working_plots/latent_axis_validation_results/):
  motor_df.csv                 model table, read by permutation_test_lme.py,
                               perm_test_lme_subscores.py, pca_comparison.py and
                               beta_benchmark_analysis.py
  step3_lme_dim2.csv, step3_lme_dim1.csv, step3_lme_dim12.csv, step5_lme_dim0.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gce.utils import select_vae_model
from config.paths import (
    HALFAMP_ALL_CONDITIONS_MAT, FULLAMP_ALL_CONDITIONS_MAT,
    SCORES_HALFAMP_XLSX, SCORES_FULLAMP_XLSX, BEST_ENCODER_PT,
)

CONFIG = {
    "halfamp_all_mat":  HALFAMP_ALL_CONDITIONS_MAT,
    "fullamp_mat":      FULLAMP_ALL_CONDITIONS_MAT,
    "scores_halfamp":   SCORES_HALFAMP_XLSX,
    "scores_fullamp":   SCORES_FULLAMP_XLSX,
    "encoder_path":     BEST_ENCODER_PT,
    "output_dir":       os.path.join(REPO_ROOT, "working_plots", "latent_axis_validation_results"),
    "K":          1,
    "L":          5,
    "n_features": 70,
    "frequencies": [125, 110, 85, 55],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(config: dict):
    base_cfg = {
        "vae_model": "vanilla_Encoder",
        "n_features": config["n_features"],
        "n_alpha": config["K"],
        "n_beta":  config["L"],
        "z_dim":   config["K"] + config["L"],
        "include_classes": None,
        "save_dir": "./", "model_name": "vae_model",
    }
    encoder, _ = select_vae_model(base_cfg)
    encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu"))
    encoder.eval()
    print(f"Encoder loaded  K={config['K']}  L={config['L']}")
    return encoder


def patient_normalise(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Per-patient StandardScaler — mirrors the training pipeline."""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for pat in np.unique(subjects):
        mask = subjects == pat
        X_norm[mask] = StandardScaler().fit_transform(X[mask]) if mask.sum() > 1 else X[mask]
    return X_norm


def get_mu(encoder, X_norm: np.ndarray) -> np.ndarray:
    """Return encoder posterior mean µ.  Shape: (n, z_dim)."""
    with torch.no_grad():
        _, mu, _ = encoder(torch.FloatTensor(X_norm))
    return mu.numpy()


def load_allcond(mat_path: str, config: dict) -> dict:
    """Load an all-conditions mat file (halfamp_all or fullamp).
    Returns dict[str(freq)] -> {'X_norm': ndarray, 'subjects': ndarray}."""
    mat = loadmat(mat_path)
    out = {}
    for freq in config["frequencies"]:
        X   = mat[f"X_{freq}_cleaned"].astype(np.float32)
        idx = mat[f"idx_{freq}_cleaned"].flatten().astype(int)
        out[str(freq)] = {"X_norm": patient_normalise(X, idx), "subjects": idx}
    return out


def load_motor_scores(path: str) -> pd.DataFrame:
    """Returns DataFrame indexed by PatNum (int)."""
    return pd.read_excel(path).set_index("PatNum")


# ─────────────────────────────────────────────────────────────────────────────
# Model table and mixed-effects model
# ─────────────────────────────────────────────────────────────────────────────

def build_motor_df(encoder,
                   halfamp_all: dict, fullamp: dict,
                   s_half: pd.DataFrame, s_full: pd.DataFrame,
                   config: dict) -> pd.DataFrame:
    """
    Builds a 72-row DataFrame:
      36 rows — halfamp_all: four frequencies × 9 patients (amp='half')
      36 rows — fullamp:     four frequencies × 9 patients (amp='full')

    Columns: patient, condition (int Hz), amp ('half'|'full'),
             dim_0_signed, dim_2_signed, norm_total,
             dim_0_signed_wc, dim_2_signed_wc (within-subject centred)
    """
    rows = []

    for amp_label, data_dict, scores_df in [
        ("half", halfamp_all, s_half),
        ("full", fullamp,     s_full),
    ]:
        for freq_str, d in data_dict.items():
            freq = int(freq_str)
            mu   = get_mu(encoder, d["X_norm"])          # (n, z_dim)
            score_col = f"Normalized_{freq}_Total"
            for pat in np.unique(d["subjects"]):
                mask = d["subjects"] == pat
                if not mask.any():
                    continue
                if pat not in scores_df.index or score_col not in scores_df.columns:
                    print(f"  Warning: missing score for pat={pat} freq={freq} amp={amp_label}")
                    continue
                z_dim = config["K"] + config["L"]
                row = {
                    "patient":    int(pat),
                    "condition":  freq,
                    "amp":        amp_label,
                    "norm_total": float(scores_df.loc[pat, score_col]),
                }
                for di in range(z_dim):
                    row[f"dim_{di}_signed"] = float(mu[mask, di].mean())
                rows.append(row)

    df = pd.DataFrame(rows)

    # within-subject centering — all dims
    z_dim = config["K"] + config["L"]
    for d in range(z_dim):
        col = f"dim_{d}_signed"
        df[col + "_wc"] = df[col] - df.groupby("patient")[col].transform("mean")

    print(f"Motor DataFrame: {len(df)} rows  amp counts: {df['amp'].value_counts().to_dict()}")
    return df


def run_mixed_effects(motor_df: pd.DataFrame,
                      predictor_col: "str | list[str]" = "dim_2_signed_wc") -> dict:
    """
    Fits: norm_total ~ {predictor_col(s)} + amp + C(condition)
    Groups: patient (random intercept).  REML=True.
    predictor_col may be a string or list of strings.

    Returns dict with 'result' (MixedLMResults) and 'summary' (DataFrame).
    """
    df = motor_df.copy()
    df["patient_str"] = df["patient"].astype(str)
    df["amp"]         = df["amp"].astype("category")
    df["condition"]   = df["condition"].astype("category")

    if isinstance(predictor_col, list):
        pred_terms = " + ".join(predictor_col)
        label = " + ".join(predictor_col)
    else:
        pred_terms = predictor_col
        label = predictor_col

    formula = f"norm_total ~ {pred_terms} + amp + C(condition)"
    model   = smf.mixedlm(formula, data=df, groups=df["patient_str"])
    result  = model.fit(reml=True)

    print(f"\n  Mixed Effects ({label}):")
    print(result.summary())

    params = result.fe_params
    fe_idx = params.index
    conf   = result.conf_int().loc[fe_idx]
    pvals  = result.pvalues.loc[fe_idx]
    se     = result.bse_fe

    summary_df = pd.DataFrame({
        "term":    fe_idx,
        "beta":    params.values,
        "se":      se.values,
        "ci_low":  conf.iloc[:, 0].values,
        "ci_high": conf.iloc[:, 1].values,
        "p_value": pvals.values,
    })
    return {"result": result, "summary": summary_df}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(config: dict = CONFIG) -> dict:
    os.makedirs(config["output_dir"], exist_ok=True)

    enc         = load_encoder(config)
    halfamp_all = load_allcond(config["halfamp_all_mat"], config)
    fullamp     = load_allcond(config["fullamp_mat"], config)
    s_half      = load_motor_scores(config["scores_halfamp"])
    s_full      = load_motor_scores(config["scores_fullamp"])

    motor_df = build_motor_df(enc, halfamp_all, fullamp, s_half, s_full, config)
    motor_df.to_csv(os.path.join(config["output_dir"], "motor_df.csv"), index=False)

    models = [
        ("step3_lme_dim2.csv",  "Dim 2",               "dim_2_signed_wc"),
        ("step3_lme_dim1.csv",  "Dim 1",               "dim_1_signed_wc"),
        ("step3_lme_dim12.csv", "Dim 1 + Dim 2 joint", ["dim_1_signed_wc", "dim_2_signed_wc"]),
        ("step5_lme_dim0.csv",  "Dim 0 (neg. ctrl)",   "dim_0_signed_wc"),
    ]
    results = {}
    for fname, label, pred in models:
        print(f"\n=== Mixed effects — {label} ===")
        res = run_mixed_effects(motor_df, predictor_col=pred)
        res["summary"].to_csv(os.path.join(config["output_dir"], fname), index=False)
        results[label] = res

    print(f"\n{'=' * 60}\nTable S4A (total score): beta, parametric P")
    for label, res in results.items():
        s = res["summary"].set_index("term")
        for term in [t for t in s.index if t.startswith("dim_")]:
            print(f"  {label:<22} {term:<17} beta={s.loc[term, 'beta']:+.3f}  "
                  f"P={s.loc[term, 'p_value']:.3f}")
    print(f"\nOutputs in: {config['output_dir']}/")
    return results


if __name__ == "__main__":
    run_validation(CONFIG)
