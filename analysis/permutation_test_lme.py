"""
Permutation test for LME model significance.

Four tests: dim2 alone, dim1 alone, joint dim1 in (dim1+dim2), joint dim2 in (dim1+dim2).
Runs N=5000 permutations at seeds [42, 2026, 4] for stability verification.
p_perm uses Phipson-Smyth adjustment: (count + 1) / (N + 1).

Also fits the joint model with all five auxiliary dimensions (Dims 1–5) and
prints Dims 3–5, reported as non-significant in the text.

Reads working_plots/latent_axis_validation_results/motor_df.csv
(analysis/latent_axis_validation.py); writes permutation_results.csv there (Table S4A).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "working_plots", "latent_axis_validation_results")
OUT_CSV = os.path.join(DATA_DIR, "permutation_results.csv")

N_PERM = 5000
SEEDS = [42, 2026, 4]


# ── LME helper ────────────────────────────────────────────────────────────────

def fit_lme(df, predictors):
    """Fit REML mixed-LM; returns {predictor: (beta, p_value)}."""
    formula = "norm_total ~ {} + amp + C(condition)".format(" + ".join(predictors))
    md = smf.mixedlm(formula, df, groups=df["patient"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = md.fit(reml=True, method="lbfgs")
    return {p: (float(res.params[p]), float(res.pvalues[p])) for p in predictors}


def phipson_smyth_p(null_arr, obs_beta):
    """Two-sided permutation p with Phipson-Smyth +1 correction."""
    valid = null_arr[~np.isnan(null_arr)]
    count = int(np.sum(np.abs(valid) >= np.abs(obs_beta)))
    return (count + 1) / (len(valid) + 1)


# ── Permutation loop for one seed ─────────────────────────────────────────────

def run_perm_seed(df, seed):
    """Return dict of null-distribution arrays for all four tests."""
    rng = np.random.default_rng(seed)
    null = {k: np.full(N_PERM, np.nan) for k in
            ["dim2_alone", "dim1_alone", "joint_dim1", "joint_dim2"]}

    col1 = df["dim_1_signed_wc"].values.copy()
    col2 = df["dim_2_signed_wc"].values.copy()

    for i in range(N_PERM):
        p1 = rng.permutation(col1)
        p2 = rng.permutation(col2)

        # dim2 alone: permute dim2 only
        d = df.copy()
        d["dim_2_signed_wc"] = p2
        try:
            null["dim2_alone"][i] = fit_lme(d, ["dim_2_signed_wc"])["dim_2_signed_wc"][0]
        except Exception:
            pass

        # dim1 alone: permute dim1 only
        d = df.copy()
        d["dim_1_signed_wc"] = p1
        try:
            null["dim1_alone"][i] = fit_lme(d, ["dim_1_signed_wc"])["dim_1_signed_wc"][0]
        except Exception:
            pass

        # joint: permute dim1 and dim2 independently
        d = df.copy()
        d["dim_1_signed_wc"] = p1
        d["dim_2_signed_wc"] = p2
        try:
            r = fit_lme(d, ["dim_1_signed_wc", "dim_2_signed_wc"])
            null["joint_dim1"][i] = r["dim_1_signed_wc"][0]
            null["joint_dim2"][i] = r["dim_2_signed_wc"][0]
        except Exception:
            pass

        if (i + 1) % 1000 == 0:
            print(f"    seed={seed}: {i + 1}/{N_PERM}")

    return null


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    motor_df = pd.read_csv(os.path.join(DATA_DIR, "motor_df.csv"))
    print(f"Loaded motor_df: {motor_df.shape}")

    # ── Fit observed models ────────────────────────────────────────────────────
    print("\n=== Observed LME fits ===")
    r_dim2 = fit_lme(motor_df, ["dim_2_signed_wc"])
    r_dim1 = fit_lme(motor_df, ["dim_1_signed_wc"])
    r_joint = fit_lme(motor_df, ["dim_1_signed_wc", "dim_2_signed_wc"])

    obs = {
        "dim2_alone": r_dim2["dim_2_signed_wc"],
        "dim1_alone": r_dim1["dim_1_signed_wc"],
        "joint_dim1": r_joint["dim_1_signed_wc"],
        "joint_dim2": r_joint["dim_2_signed_wc"],
    }

    for test, (beta, p) in obs.items():
        print(f"  {test:<15}: beta={beta:+.3f}  p={p:.3f}")

    r_all = fit_lme(motor_df, [f"dim_{d}_signed_wc" for d in range(1, 6)])
    print("\n  Joint model, Dims 1–5 (Dims 3–5 reported in text):")
    for d in range(3, 6):
        beta, p = r_all[f"dim_{d}_signed_wc"]
        print(f"    Dim {d}: beta={beta:+.3f}  p={p:.3f}")

    print("\nStarting permutation tests...\n")

    # ── Run permutation tests ──────────────────────────────────────────────────
    seed_nulls = {}
    for seed in SEEDS:
        print(f"  Seed {seed}:")
        seed_nulls[seed] = run_perm_seed(motor_df, seed)

    # ── Compute p-values and assemble results ──────────────────────────────────
    tests = ["dim2_alone", "dim1_alone", "joint_dim1", "joint_dim2"]
    rows = []
    for test in tests:
        beta, param_p = obs[test]
        p_by_seed = {s: phipson_smyth_p(seed_nulls[s][test], beta) for s in SEEDS}
        p_vals = list(p_by_seed.values())
        rows.append({
            "test": test,
            "observed_beta": beta,
            "parametric_p": param_p,
            "p_perm_seed42": p_by_seed[42],
            "p_perm_seed2026": p_by_seed[2026],
            "p_perm_seed4": p_by_seed[4],
            "range_across_seeds": max(p_vals) - min(p_vals),
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ── Print summary table ────────────────────────────────────────────────────
    max_range = results["range_across_seeds"].max()
    print("\n=== Permutation Results ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nStability: max Δ across seeds = {max_range:.4f} (N={N_PERM} per seed)")
    print(f"  → Body-text claim: Δ ≤ {max_range:.3f} for all four tests at N = {N_PERM:,}")


if __name__ == "__main__":
    main()
