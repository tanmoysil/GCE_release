"""
generate_figure5.py
====================================
Supplementary Table S8 and publication Figure 5 for the independent kinematic
cohort (three patients from a separate study at the same centre).

Reads the displacement tables written by analysis/kinematic_cohort_test.py to
working_plots/kinematic_cohort_results/ -- no encoder re-run required.

Produces in working_plots/kinematic_cohort_results/:
  table_s8.csv                         machine-readable Supplementary Table S8
  table_s8.md                          markdown table for manuscript paste
  figure5_kinematic.pdf / .png / .svg  publication Figure 5 (panels a + b)
"""

import math
import os
import sys
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "Roboto", "font.weight": "medium",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
KINEMATIC_DIR = os.path.join(BASE_DIR, "working_plots", "kinematic_cohort_results")
OUT_DIR    = os.path.join(BASE_DIR, "working_plots", "kinematic_cohort_results")

# Neutral patient labels: K1/K2/K3 -- avoids collision with main-cohort P-numbers
PATIENT_MAP = {"LON03": "K1", "LON04": "K2", "LON05": "K3"}

# Okabe-Ito palette
PATIENT_COLORS = {"K1": "#0173B2", "K2": "#DE8F05", "K3": "#029E73"}

# Four examined PC scores (PC3 not analysed, see Methods)
SCORES_OF_INTEREST = ["FT_PC1", "FT_PC2", "HM_PC1", "HM_PC2"]

# Expected sign of rho(dims12, DeltaPC) under therapeutic improvement
# Source: kinematic_cohort_test.py CONFIG expected_signs + training-cohort convention
EXPECTED_DIRECTION = {
    "FT_PC1": "positive",   # higher FT-PC1 = better tapping speed
    "FT_PC2": "negative",   # lower  FT-PC2 = better tapping consistency
    "HM_PC1": "negative",   # lower  HM-PC1 = faster hand movement
    "HM_PC2": "positive",   # higher HM-PC2 = better hand movement quality
}



# ── helpers ──────────────────────────────────────────────────────────────────

def _sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "·"
    return ""


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _relabel_patients(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patient"] = df["subject"].map(PATIENT_MAP)
    return df


# ── data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_all, df_balanced):
      df_all:      n=14 pooled rows, columns include dims12_signed, FT_PC*, HM_PC*
      df_balanced: n=12 balanced-subset rows (cond1+cond3 only, 4 rows per patient)
    """
    path_all = os.path.join(KINEMATIC_DIR, "displacements_contra.csv")
    path_bal = os.path.join(KINEMATIC_DIR, "displacements_contra_cond1+cond3.csv")

    for p in (path_all, path_bal):
        if not os.path.exists(p):
            sys.exit(f"ERROR: required CSV not found: {p}\n"
                     "Run analysis/kinematic_cohort_test.py first to generate it.")

    df_all = pd.read_csv(path_all)
    df_bal = pd.read_csv(path_bal)

    df_all = _relabel_patients(df_all)
    df_bal = _relabel_patients(df_bal)
    return df_all, df_bal


# ── verification ──────────────────────────────────────────────────────────────

# ── z-scoring ─────────────────────────────────────────────────────────────────

def zscore_pc_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Z-score the specified PC columns within the provided cohort (fit on this
    df only, not refitted per patient).  Returns a copy with new columns
    named <col>_z.  Spearman rho is rank-invariant so correlations are
    identical before and after z-scoring.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            vals = df[col].values.reshape(-1, 1)
            df[col + "_z"] = StandardScaler().fit_transform(vals).ravel()
    return df


# ── exact permutation p-value ─────────────────────────────────────────────────

def exact_permutation_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-sided exact permutation p-value for Spearman rho.

    Enumerates all n! permutations of y (keeping x fixed), computes Spearman
    rho for each, and returns the fraction with |rho_perm| >= |rho_obs|.
    Safe for n <= 8 (8! = 40320); called only for per-patient rows where n is
    4 or 6 (24 or 720 permutations).
    """
    n = len(x)
    rho_obs, _ = spearmanr(x, y)
    abs_obs    = abs(rho_obs)

    count = sum(
        1 for perm in permutations(y)
        if abs(spearmanr(x, perm)[0]) >= abs_obs - 1e-10
    )
    return count / math.factorial(n)


# ── per-patient correlations ──────────────────────────────────────────────────

def compute_per_patient_correlations(df: pd.DataFrame,
                                     scores: list[str]) -> pd.DataFrame:
    """
    Compute Spearman rho and p for each patient and pooled.

    Per-patient p-values: exact two-sided permutation (all n! enumerations).
    Pooled p-values:      scipy asymptotic.
    """
    rows = []
    patients_ordered = ["K1", "K2", "K3"]

    for score in scores:
        # per patient -- exact permutation p
        for pat in patients_ordered:
            sub = df[df["patient"] == pat].dropna(subset=[score])
            if len(sub) < 3:
                warnings.warn(f"Skipping {pat}/{score}: only {len(sub)} rows")
                continue
            x   = sub["dims12_signed"].values
            y   = sub[score].values
            rho, _ = spearmanr(x, y)          # rho from scipy (exact for ranks)
            p_exact = exact_permutation_spearman_p(x, y)
            rows.append({
                "patient":  pat,
                "score":    score,
                "rho":      float(rho),
                "p":        p_exact,
                "p_method": "exact permutation",
                "n":        len(sub),
            })

        # pooled (all patients) -- scipy asymptotic p
        pool = df.dropna(subset=[score])
        rho, p = spearmanr(pool["dims12_signed"], pool[score])
        rows.append({
            "patient":  "Pooled",
            "score":    score,
            "rho":      float(rho),
            "p":        float(p),
            "p_method": "asymptotic",
            "n":        len(pool),
        })

    return pd.DataFrame(rows)


# ── Table S8 ──────────────────────────────────────────────────────────────────

def build_table_s8(df_all: pd.DataFrame,
                   df_bal: pd.DataFrame) -> pd.DataFrame:
    """
    Build Supplementary Table S8:
      rows = K1, K2, K3, Pooled (n=14), Balanced subset (n=12)
      cols = n (shared across scores for that row), then rho + p per score
             for all four examined components: FT-PC1, FT-PC2, HM-PC1, HM-PC2

    Per-patient and pooled correlations use df_all.
    Balanced-subset row uses df_bal pooled.
    """
    scores      = SCORES_OF_INTEREST
    per_patient = compute_per_patient_correlations(df_all, scores)

    # balanced-subset pooled row -- scipy asymptotic p
    bal_rows = []
    for score in scores:
        pool = df_bal.dropna(subset=[score])
        rho, p = spearmanr(pool["dims12_signed"], pool[score])
        bal_rows.append({
            "patient":  "Balanced subset",
            "score":    score,
            "rho":      float(rho),
            "p":        float(p),
            "p_method": "asymptotic",
            "n":        len(pool),
        })
    bal_df = pd.DataFrame(bal_rows)

    combined = pd.concat([per_patient, bal_df], ignore_index=True)

    # Pivot to wide: one row per patient, n shared (same across all scores for
    # a given patient row), then (rho, p) per score.
    # Store raw p (not rounded) so fmt_p can make fine-grained decisions.
    pivot_rows = []
    row_order  = ["K1", "K2", "K3", "Pooled", "Balanced subset"]
    for pat in row_order:
        sub   = combined[combined["patient"] == pat]
        n_val = int(sub["n"].iloc[0]) if not sub.empty else np.nan
        row   = {"Patient": pat, "n": n_val}
        for score in scores:
            s = sub[sub["score"] == score]
            if s.empty:
                row[f"{score}_rho"] = np.nan
                row[f"{score}_p"]   = np.nan
            else:
                row[f"{score}_rho"] = round(s["rho"].iloc[0], 3)
                row[f"{score}_p"]   = float(s["p"].iloc[0])   # keep full precision
        pivot_rows.append(row)

    return pd.DataFrame(pivot_rows)


def render_table_s8_markdown(t: pd.DataFrame, per_patient_rows: set[str]) -> str:
    """
    Render Supplementary Table S8 as a markdown table.
    Expected direction is shown in the column header (exp: + or -).
    No significance stars -- this is a descriptive table.

    per_patient_rows: set of Patient labels whose p-values are exact
      permutation (shown to 3 d.p. without "< 0.001" shortcut).
    Pooled and balanced-subset rows use "< 0.001" for very small asymptotic p.
    """
    def fmt_rho(v):
        return "--" if pd.isna(v) else f"{v:+.3f}"

    def fmt_p(v, is_exact: bool):
        if pd.isna(v):
            return "--"
        if is_exact:
            # Exact permutation p -- show full 3 d.p.; minimum is 1/720 = 0.001
            return f"{v:.3f}"
        # Asymptotic p
        if v < 0.001:
            return "< 0.001"
        return f"{v:.3f}"

    # Column headers with expected direction
    dir_sym = {"positive": "+", "negative": "-"}
    score_labels = {
        "FT_PC1": f"FT-PC1 (exp: {dir_sym[EXPECTED_DIRECTION['FT_PC1']]})",
        "FT_PC2": f"FT-PC2 (exp: {dir_sym[EXPECTED_DIRECTION['FT_PC2']]})",
        "HM_PC1": f"HM-PC1 (exp: {dir_sym[EXPECTED_DIRECTION['HM_PC1']]})",
        "HM_PC2": f"HM-PC2 (exp: {dir_sym[EXPECTED_DIRECTION['HM_PC2']]})",
    }

    col_pairs = [(sc, score_labels[sc]) for sc in SCORES_OF_INTEREST]

    header_parts = ["| Patient | n"]
    for _, lbl in col_pairs:
        header_parts.append(f" | {lbl} rho | {lbl} p")
    header = "".join(header_parts) + " |"

    sep_parts = ["|---------|---"]
    for _ in col_pairs:
        sep_parts.append("|---------|------")
    sep = "".join(sep_parts) + "|"

    lines = [header, sep]
    for _, row in t.iterrows():
        is_exact = row["Patient"] in per_patient_rows
        n_str    = "--" if pd.isna(row["n"]) else str(int(row["n"]))
        parts    = [f"| {row['Patient']} | {n_str}"]
        for sc, _ in col_pairs:
            parts.append(
                f" | {fmt_rho(row[f'{sc}_rho'])}"
                f" | {fmt_p(row[f'{sc}_p'], is_exact)}"
            )
        lines.append("".join(parts) + " |")

    return "\n".join(lines)


# ── Figure 5 ──────────────────────────────────────────────────────────────────

def plot_figure5(df_all: pd.DataFrame) -> None:
    """
    Two-panel scatter:
      Panel a: dims12 vs z-scored DeltaFT-PC2, coloured by patient
      Panel b: dims12 vs z-scored DeltaHM-PC1, coloured by patient
    OLS regression line; rho/p annotation computed from data.
    """
    df = zscore_pc_columns(df_all, ["FT_PC2", "HM_PC1"])

    panels = [
        ("FT_PC2_z", "FT_PC2", "z-scored ΔFT-PC2",
         "a", "Finger-tapping performance consistency (FT-PC2)"),
        ("HM_PC1_z", "HM_PC1", "z-scored ΔHM-PC1",
         "b", "Hand-movement speed (HM-PC1)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (ycol, raw_col, ylabel, panel_label, subtitle) in zip(axes, panels):
        sub = df.dropna(subset=[ycol])

        x_all = sub["dims12_signed"].values
        y_all = sub[ycol].values

        # Scatter -- coloured by patient
        for pat in ["K1", "K2", "K3"]:
            mask = sub["patient"] == pat
            ax.scatter(
                x_all[mask.values], y_all[mask.values],
                color=PATIENT_COLORS[pat],
                s=120, zorder=4,
                edgecolors="black", linewidths=1.5,
                label=pat,
            )

        # OLS regression line
        z = np.polyfit(x_all, y_all, 1)
        x_fit = np.linspace(x_all.min(), x_all.max(), 200)
        ax.plot(x_fit, np.polyval(z, x_fit),
                "k--", linewidth=1.5, zorder=3)

        # Spearman annotation (computed from data)
        rho, p_val = spearmanr(x_all, y_all)
        n          = len(sub)
        sig        = _sig_marker(p_val)
        ax.text(
            0.05, 0.95,
            f"ρ = {rho:+.3f}{sig}\np = {_fmt_p(p_val)}\nn = {n}",
            transform=ax.transAxes,
            fontsize=11, va="top",
            fontfamily="Roboto", fontweight="medium",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )

        ax.axhline(0, color="#B0B0B0", linestyle=":", linewidth=0.9, zorder=1)
        ax.axvline(0, color="#B0B0B0", linestyle=":", linewidth=0.9, zorder=1)

        ax.set_xlabel("dims₁₂  (signed displacement from OFF)",
                      fontsize=13, fontweight="medium", fontfamily="Roboto")
        ax.set_ylabel(ylabel, fontsize=13,
                      fontweight="medium", fontfamily="Roboto")
        ax.set_title(subtitle, fontsize=12, fontweight="bold",
                     fontfamily="Roboto")
        ax.tick_params(axis="both", labelsize=12, length=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel label -- uppercase to match all other manuscript figures
        ax.text(-0.12, 1.06, panel_label.upper(), transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")

    # Shared legend (patient colours)
    handles = [
        mlines.Line2D([], [], color=PATIENT_COLORS[p], marker="o",
                      linestyle="None", markersize=8,
                      markeredgecolor="black", markeredgewidth=0.7,
                      label=p)
        for p in ["K1", "K2", "K3"]
    ]
    axes[0].legend(handles=handles, fontsize=10, framealpha=0.8,
                   prop={"family": "Roboto", "weight": "medium"},
                   title="Patient", title_fontsize=10)

    plt.tight_layout()

    base = os.path.join(OUT_DIR, "figure5_kinematic")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight", format="svg")
    plt.close()
    print(f"Saved figure5_kinematic.pdf / .png / .svg  in  {OUT_DIR}/")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading pre-computed displacement CSVs ...")
    df_all, df_bal = load_data()
    print(f"  All conditions: {len(df_all)} rows, "
          f"{df_all['patient'].value_counts().to_dict()}")
    print(f"  Balanced subset: {len(df_bal)} rows, "
          f"{df_bal['patient'].value_counts().to_dict()}")

    # ── Table S8 ──────────────────────────────────────────────────────────────
    print("\nBuilding Supplementary Table S8 ...")
    t8 = build_table_s8(df_all, df_bal)
    t8.to_csv(os.path.join(OUT_DIR, "table_s8.csv"), index=False)

    md = render_table_s8_markdown(t8, per_patient_rows={"K1", "K2", "K3"})
    with open(os.path.join(OUT_DIR, "table_s8.md"), "w") as fh:
        fh.write(md + "\n")
    print(f"  Saved table_s8.csv and table_s8.md  in  {OUT_DIR}/")

    # Pretty-print the table to console
    print("\nSupplementary Table S8 -- per-patient Spearman correlations:")
    print(md)

    # ── Figure 5 ──────────────────────────────────────────────────────────────
    print("\nGenerating Figure 5 ...")
    plot_figure5(df_all)

    # ── Summary of per-patient values ─────────────────────────────────────────
    print("\nPer-patient values (dims12 vs all four examined PC scores):")
    for score in SCORES_OF_INTEREST:
        print(f"\n  {score}:")
        for pat in ["K1", "K2", "K3"]:
            sub = df_all[df_all["patient"] == pat].dropna(subset=[score])
            rho, p = spearmanr(sub["dims12_signed"], sub[score])
            print(f"    {pat}  n={len(sub)}  rho={rho:+.3f}  p={p:.3f}")
        pool = df_all.dropna(subset=[score])
        rho_p, p_p = spearmanr(pool["dims12_signed"], pool[score])
        print(f"    Pooled  n={len(pool)}  rho={rho_p:+.3f}  p={p_p:.3f}")
        bal = df_bal.dropna(subset=[score])
        rho_b, p_b = spearmanr(bal["dims12_signed"], bal[score])
        print(f"    Balanced subset  n={len(bal)}  rho={rho_b:+.3f}  p={p_b:.3f}")


if __name__ == "__main__":
    main()
