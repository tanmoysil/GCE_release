"""
PCA benchmark — unsupervised principal components through the Table S4 model.

PCA (20 components) is fitted to the pooled ON-condition features of both
amplitudes (1008 segments), prepared as in analysis/latent_axis_validation.py.
For each of the first 10 components, the per-condition mean score is centred
within patient and entered in the same mixed-effects model as the latent
dimensions:

    norm_total ~ PC_wc + amp + C(frequency) + (1 | patient)

Prints variance explained, beta and P per component, next to the Dim 2 model
from step3_lme_dim2.csv (written by latent_axis_validation.py).
Writes working_plots/latent_axis_validation_results/motor_df_with_pcs.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config.paths import (
    HALFAMP_ALL_CONDITIONS_MAT, FULLAMP_ALL_CONDITIONS_MAT,
    SCORES_HALFAMP_XLSX, SCORES_FULLAMP_XLSX,
)

OUT_DIR = os.path.join(REPO_ROOT, "working_plots", "latent_axis_validation_results")

CONFIG = {
    "halfamp_all_mat": HALFAMP_ALL_CONDITIONS_MAT,
    "fullamp_mat":     FULLAMP_ALL_CONDITIONS_MAT,
    "scores_halfamp":  SCORES_HALFAMP_XLSX,
    "scores_fullamp":  SCORES_FULLAMP_XLSX,
    "frequencies": [125, 110, 85, 55],
    "n_components": 20,
    "n_report": 10,
}


def patient_normalise(X, subjects):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for pat in np.unique(subjects):
        mask = subjects == pat
        X_norm[mask] = StandardScaler().fit_transform(X[mask]) if mask.sum() > 1 else X[mask]
    return X_norm


def load_blocks(config):
    """Per (amplitude, frequency) block: normalised features, patient ids, scores.
    Same preparation as load_allcond in latent_axis_validation.py."""
    blocks = []
    for amp_label, mat_path, score_path in [
        ("half", config["halfamp_all_mat"], config["scores_halfamp"]),
        ("full", config["fullamp_mat"],     config["scores_fullamp"]),
    ]:
        mat = loadmat(mat_path)
        scores_df = pd.read_excel(score_path, index_col=0)
        for freq in config["frequencies"]:
            X_raw = mat[f"X_{freq}_cleaned"].astype(np.float32)
            idx   = mat[f"idx_{freq}_cleaned"].flatten().astype(int)
            score_col = f"Normalized_{freq}_Total"
            blocks.append({"amp": amp_label, "freq": freq, "idx": idx,
                           "X_norm": patient_normalise(X_raw, idx),
                           "scores": scores_df[score_col]})
    return blocks


def main(config=CONFIG):
    blocks = load_blocks(config)
    X_pool = np.vstack([b["X_norm"] for b in blocks])
    print(f"Feature matrix for PCA: {X_pool.shape}")

    pca = PCA(n_components=config["n_components"], random_state=42).fit(X_pool)
    var_explained = pca.explained_variance_ratio_

    rows = []
    for b in blocks:
        X_pc = pca.transform(b["X_norm"])
        for pat in np.unique(b["idx"]):
            mask = b["idx"] == pat
            # Average all components together (row-wise mean), exactly as in the
            # original analysis.
            pc_mean = X_pc[mask].mean(axis=0)
            row = {"patient": int(pat), "condition": b["freq"], "amp": b["amp"],
                   "norm_total": float(b["scores"].loc[pat])}
            for k in range(config["n_report"]):
                row[f"pc_{k + 1}"] = pc_mean[k]
            rows.append(row)
    motor_df = pd.DataFrame(rows)
    for k in range(1, config["n_report"] + 1):
        col = f"pc_{k}"
        motor_df[col + "_wc"] = motor_df[col] - motor_df.groupby("patient")[col].transform("mean")
    os.makedirs(OUT_DIR, exist_ok=True)
    motor_df.to_csv(os.path.join(OUT_DIR, "motor_df_with_pcs.csv"), index=False)

    print("\n=== LME: each PC vs normalized total improvement ===")
    pvals = []
    for k in range(1, config["n_report"] + 1):
        col = f"pc_{k}_wc"
        res = smf.mixedlm(f"norm_total ~ {col} + amp + C(condition)",
                          motor_df, groups=motor_df["patient"]).fit(reml=True, method="lbfgs")
        b, p = res.params[col], res.pvalues[col]
        pvals.append(p)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.1 else ""))
        print(f"  PC{k:<2} ({100 * var_explained[k - 1]:4.1f}% var): "
              f"beta={b:+.3f}  P={p:.4f} {sig}")

    ref_path = os.path.join(OUT_DIR, "step3_lme_dim2.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path).set_index("term").loc["dim_2_signed_wc"]
        print(f"\nReference, Dim 2 (same model): beta={ref['beta']:+.3f}  P={ref['p_value']:.4f}")
    sig_pcs = [k + 1 for k, p in enumerate(pvals) if p < 0.05]
    print(f"Components with P < 0.05: {sig_pcs if sig_pcs else 'none'} "
          f"(of {config['n_report']} tested)")


if __name__ == "__main__":
    main()
