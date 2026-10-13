"""
Permutation test for LME significance — bradykinesia and rigidity subscores.

Four tests per outcome (norm_brady, norm_rigidity):
  - Dim 1 alone
  - Dim 2 alone
  - Dim 1 in joint (Dim 1+2) model
  - Dim 2 in joint (Dim 1+2) model

N=5000 permutations, seed=42, parallelised across all available CPU cores.
p_perm uses Phipson-Smyth adjustment: (count + 1) / (N + 1).

Writes working_plots/latent_axis_validation_results/permutation_results_subscores.csv
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "working_plots", "latent_axis_validation_results")
OUT_CSV = os.path.join(DATA_DIR, "permutation_results_subscores.csv")

N_PERM = 5000
SEED = 42
N_WORKERS = min(64, os.cpu_count() or 1)

OUTCOMES = ["norm_brady", "norm_rigidity"]


# ── Build merged dataframe ─────────────────────────────────────────────────────

def _load_df():
    motor_df = pd.read_csv(os.path.join(DATA_DIR, "motor_df.csv"))
    full = pd.read_csv(os.path.join(BASE_DIR, "analysis", "fullamp_correlation_analysis",
                                    "fullamp_combined_data.csv"))[
        ["patient", "condition", "norm_brady", "norm_rigidity"]]
    half = pd.read_csv(os.path.join(BASE_DIR, "analysis", "halfamp_correlation_analysis",
                                    "halfamp_combined_data.csv"))[
        ["patient", "condition", "norm_brady", "norm_rigidity"]]
    full["amp"] = "full"
    half["amp"] = "half"
    return motor_df.merge(pd.concat([half, full], ignore_index=True),
                          on=["patient", "condition", "amp"], how="inner")


# ── LME helpers ────────────────────────────────────────────────────────────────

def _fit_single(data, outcome, pred):
    formula = f"{outcome} ~ {pred} + amp + C(condition)"
    md = smf.mixedlm(formula, data, groups=data["patient"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "powell", "bfgs"):
            try:
                res = md.fit(reml=True, method=method)
                return float(res.params[pred])
            except Exception:
                continue
    return np.nan


def _fit_joint(data, outcome):
    formula = f"{outcome} ~ dim_1_signed_wc + dim_2_signed_wc + amp + C(condition)"
    md = smf.mixedlm(formula, data, groups=data["patient"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "powell", "bfgs"):
            try:
                res = md.fit(reml=True, method=method)
                return (float(res.params["dim_1_signed_wc"]),
                        float(res.params["dim_2_signed_wc"]))
            except Exception:
                continue
    return np.nan, np.nan


def phipson_smyth_p(null_arr, obs_beta):
    valid = null_arr[~np.isnan(null_arr)]
    count = int(np.sum(np.abs(valid) >= np.abs(obs_beta)))
    return (count + 1) / (len(valid) + 1)


# ── Worker function (module-level for pickle) ──────────────────────────────────

def _worker(args):
    """Run a batch of permutations and return null beta arrays."""
    d1_vals, d2_vals, df_vals, outcomes_data, child_seed, n_local = args

    import os as _os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        _os.environ[var] = "1"

    import numpy as _np
    import pandas as _pd
    import statsmodels.formula.api as _smf
    import warnings as _warnings

    rng = _np.random.default_rng(child_seed)

    # Outcome labels and their column index in outcomes_data
    outcome_names = list(outcomes_data.keys())

    # Null containers: outcome → {dim1, dim2, jdim1, jdim2}
    nulls = {oc: {"dim1": _np.full(n_local, _np.nan),
                  "dim2": _np.full(n_local, _np.nan),
                  "jdim1": _np.full(n_local, _np.nan),
                  "jdim2": _np.full(n_local, _np.nan)}
             for oc in outcome_names}

    base_df = _pd.DataFrame(df_vals)

    def fit_s(data, outcome, pred):
        formula = f"{outcome} ~ {pred} + amp + C(condition)"
        md = _smf.mixedlm(formula, data, groups=data["patient"])
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for method in ("lbfgs", "powell", "bfgs"):
                try:
                    res = md.fit(reml=True, method=method)
                    return float(res.params[pred])
                except Exception:
                    continue
        return _np.nan

    def fit_j(data, outcome):
        formula = f"{outcome} ~ dim_1_signed_wc + dim_2_signed_wc + amp + C(condition)"
        md = _smf.mixedlm(formula, data, groups=data["patient"])
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for method in ("lbfgs", "powell", "bfgs"):
                try:
                    res = md.fit(reml=True, method=method)
                    return (float(res.params["dim_1_signed_wc"]),
                            float(res.params["dim_2_signed_wc"]))
                except Exception:
                    continue
        return _np.nan, _np.nan

    for i in range(n_local):
        p1 = rng.permutation(len(d1_vals))
        p2 = rng.permutation(len(d2_vals))

        for oc in outcome_names:
            df_p1 = base_df.copy()
            df_p1["dim_1_signed_wc"] = _np.array(d1_vals)[p1]
            nulls[oc]["dim1"][i] = fit_s(df_p1, oc, "dim_1_signed_wc")

            df_p2 = base_df.copy()
            df_p2["dim_2_signed_wc"] = _np.array(d2_vals)[p2]
            nulls[oc]["dim2"][i] = fit_s(df_p2, oc, "dim_2_signed_wc")

            df_pj = base_df.copy()
            df_pj["dim_1_signed_wc"] = _np.array(d1_vals)[p1]
            df_pj["dim_2_signed_wc"] = _np.array(d2_vals)[p2]
            b1, b2 = fit_j(df_pj, oc)
            nulls[oc]["jdim1"][i] = b1
            nulls[oc]["jdim2"][i] = b2

    return nulls


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = _load_df()
    print(f"Working dataframe: {df.shape}  workers: {N_WORKERS}")

    print("\n=== Observed betas ===")
    obs = {}
    for outcome in OUTCOMES:
        b1 = _fit_single(df, outcome, "dim_1_signed_wc")
        b2 = _fit_single(df, outcome, "dim_2_signed_wc")
        jb1, jb2 = _fit_joint(df, outcome)
        obs[outcome] = {"dim1_alone": b1, "dim2_alone": b2,
                        "joint_dim1": jb1, "joint_dim2": jb2}
        for test, beta in obs[outcome].items():
            print(f"  {outcome} {test}: beta={beta:.4f}")

    print("\n=== Parametric fits (Table S4B/C: beta, P) ===")
    for outcome in OUTCOMES:
        models = [("Dim 0 (neg. ctrl)", ["dim_0_signed_wc"]),
                  ("Dim 1", ["dim_1_signed_wc"]),
                  ("Dim 2", ["dim_2_signed_wc"]),
                  ("joint Dims 1+2", ["dim_1_signed_wc", "dim_2_signed_wc"])]
        for label, preds in models:
            formula = f"{outcome} ~ {' + '.join(preds)} + amp + C(condition)"
            md = smf.mixedlm(formula, df, groups=df["patient"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for method in ("lbfgs", "powell", "bfgs"):
                    try:
                        res = md.fit(reml=True, method=method)
                        break
                    except Exception:
                        continue
            terms = "  ".join(f"{p[:5]} beta={res.params[p]:+.3f} P={res.pvalues[p]:.3f}"
                              for p in preds)
            print(f"  {outcome:<13} {label:<18} {terms}")

    # Spawn independent child seeds from root seed
    ss = np.random.SeedSequence(SEED)
    child_seeds = ss.spawn(N_WORKERS)

    # Distribute N_PERM across workers
    base_chunk = N_PERM // N_WORKERS
    remainder = N_PERM % N_WORKERS
    chunk_sizes = [base_chunk + (1 if i < remainder else 0) for i in range(N_WORKERS)]

    d1_vals = df["dim_1_signed_wc"].tolist()
    d2_vals = df["dim_2_signed_wc"].tolist()
    df_dict = df.to_dict("list")
    outcomes_data = {oc: df[oc].tolist() for oc in OUTCOMES}

    args_list = [
        (d1_vals, d2_vals, df_dict, outcomes_data,
         int(child_seeds[i].generate_state(1)[0]), chunk_sizes[i])
        for i in range(N_WORKERS)
    ]

    print(f"\nDispatching {N_WORKERS} workers ({N_PERM} total permutations)...")
    all_nulls = {oc: {"dim1": [], "dim2": [], "jdim1": [], "jdim2": []}
                 for oc in OUTCOMES}

    completed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, a): i for i, a in enumerate(args_list)}
        for fut in as_completed(futures):
            result = fut.result()
            for oc in OUTCOMES:
                for k in ("dim1", "dim2", "jdim1", "jdim2"):
                    all_nulls[oc][k].append(result[oc][k])
            completed += chunk_sizes[futures[fut]]
            print(f"  {completed}/{N_PERM} permutations done", flush=True)

    # Concatenate chunks
    for oc in OUTCOMES:
        for k in ("dim1", "dim2", "jdim1", "jdim2"):
            all_nulls[oc][k] = np.concatenate(all_nulls[oc][k])

    # Save null distributions
    for oc_short, oc in [("brady", "norm_brady"), ("rigidity", "norm_rigidity")]:
        np.save(os.path.join(DATA_DIR, f"perm_null_{oc_short}_dim1.npy"),   all_nulls[oc]["dim1"])
        np.save(os.path.join(DATA_DIR, f"perm_null_{oc_short}_dim2.npy"),   all_nulls[oc]["dim2"])
        np.save(os.path.join(DATA_DIR, f"perm_null_{oc_short}_d1_joint.npy"), all_nulls[oc]["jdim1"])
        np.save(os.path.join(DATA_DIR, f"perm_null_{oc_short}_d2_joint.npy"), all_nulls[oc]["jdim2"])

    # Compute p-values and save
    rows = []
    print("\n=== Results ===")
    test_map = [
        ("dim1_alone",  "dim1"),
        ("dim2_alone",  "dim2"),
        ("joint_dim1",  "jdim1"),
        ("joint_dim2",  "jdim2"),
    ]
    for outcome in OUTCOMES:
        for test_name, null_key in test_map:
            null_arr = all_nulls[outcome][null_key]
            obs_beta = obs[outcome][test_name]
            p_perm = phipson_smyth_p(null_arr, obs_beta)
            n_valid = int(np.sum(~np.isnan(null_arr)))
            print(f"  {outcome} {test_name}: obs_beta={obs_beta:.4f}  p_perm={p_perm:.4f}  n_valid={n_valid}")
            rows.append({
                "outcome": outcome,
                "test": test_name,
                "observed_beta": obs_beta,
                "p_perm_seed42": p_perm,
                "n_valid": n_valid,
            })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
