#!/usr/bin/env python3
"""
frequency_statistics.py — frequency-dependent statistics at full amplitude.

Reproduces the statistics reported for the auxiliary dims₁₂ metric across the
four stimulation frequencies (main text Results; table S3; Friedman column of
table S6C). Results are printed only; nothing is written to disk.

  1. Auxiliary-dimension screen: Spearman ρ of each auxiliary dim (1–5) against
     normalized total improvement (n = 36), Benjamini–Hochberg across the five.
  2. Table S3: per-frequency mean ± SD and range of signed dims₁₂, Euclidean
     magnitude √(Δz₁² + Δz₂²), clinical improvement, and patients worsened.
  3. Sign test per frequency (all nine patients negative on dims₁₂?).
  4. Wilcoxon signed-rank, 55 Hz vs each other frequency on signed dims₁₂,
     Holm-corrected across the three comparisons.
  5. Friedman test across frequencies (patient as block, N = 9) for dims₁₂,
     normalized clinical improvement, and the four canonical band powers.

Inputs:
  analysis/fullamp_correlation_analysis/fullamp_combined_data.csv
      (written by analysis/patient_specific_norm_hemi_fullamp.py)
  config.paths.FULLAMP_ALL_CONDITIONS_MAT (band powers, via
      analysis/beta_benchmark_analysis.py)

Usage:
    python analysis/frequency_statistics.py
"""

import os
import sys

import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from analysis.beta_benchmark_analysis import BANDS, build_beta_spearman_df
from analysis.beta_benchmark_analysis import CONFIG as BETA_CONFIG

# =============================================================================
# CONFIG
# =============================================================================
COMBINED_CSV = os.path.join(
    REPO_ROOT, "analysis", "fullamp_correlation_analysis", "fullamp_combined_data.csv"
)
FREQUENCIES = [125, 110, 85, 55]
AUX_DIMS = [1, 2, 3, 4, 5]
METRIC = "dims_1_2_sum_signed"   # signed Δz₁ + Δz₂
EUCLID = "dims_1_2_norm"         # √(Δz₁² + Δz₂²)
OUTCOME = "norm_total"
REF_FREQ = 55

SEP = "=" * 70


def by_frequency(df, col):
    """(n_patients × n_frequencies) array, patients as rows, columns in FREQUENCIES order."""
    wide = df.pivot(index="patient", columns="condition", values=col)[FREQUENCIES]
    if wide.isna().any().any():
        raise ValueError(f"{col}: missing patient × frequency cells")
    return wide.to_numpy()


def auxiliary_screen(df):
    print(f"\n{SEP}\n1. AUXILIARY-DIMENSION SCREEN  (Spearman vs {OUTCOME}, n={len(df)})\n{SEP}")
    rho, p = zip(*(spearmanr(df[f"dim_{d}_signed"], df[OUTCOME]) for d in AUX_DIMS))
    _, q, _, _ = multipletests(p, method="fdr_bh")
    for d, r, pv, qv in zip(AUX_DIMS, rho, p, q):
        print(f"  Dim {d}:  ρ = {r:+.3f}  P = {pv:.4f}  q(BH) = {qv:.4f}")
    r12, p12 = spearmanr(df[METRIC], df[OUTCOME])
    print(f"  dims₁₂:  ρ = {r12:+.3f}  P = {p12:.4f}")


def table_s3(df):
    print(f"\n{SEP}\n2. TABLE S3  (mean ± SD, N = 9 per frequency)\n{SEP}")
    print(f"  {'Freq':<8}{'Signed dims₁₂':<17}{'Range':<18}"
          f"{'Euclidean':<15}{'Clinical':<16}Worsened")
    for f in FREQUENCIES:
        s = df[df["condition"] == f]
        n_worse = int((s[OUTCOME] < 0).sum())
        print(f"  {f:<3} Hz  "
              f"{s[METRIC].mean():+.2f} ± {s[METRIC].std():.2f}    "
              f"{s[METRIC].min():+.2f} to {s[METRIC].max():+.2f}    "
              f"{s[EUCLID].mean():.2f} ± {s[EUCLID].std():.2f}    "
              f"{s[OUTCOME].mean():+.2f} ± {s[OUTCOME].std():.2f}    "
              f"{n_worse}/{len(s)} ({100 * n_worse / len(s):.0f}%)")


def sign_tests(df):
    print(f"\n{SEP}\n3. SIGN TEST  (signed dims₁₂ < 0, two-sided)\n{SEP}")
    for f in FREQUENCIES:
        x = df.loc[df["condition"] == f, METRIC].to_numpy()
        n_neg = int((x < 0).sum())
        p = binomtest(n_neg, len(x), 0.5).pvalue
        print(f"  {f:<3} Hz:  {n_neg}/{len(x)} negative  P = {p:.4f}")


def wilcoxon_vs_ref(df):
    print(f"\n{SEP}\n4. WILCOXON SIGNED-RANK  ({REF_FREQ} Hz vs others, signed dims₁₂, Holm)\n{SEP}")
    wide = by_frequency(df, METRIC)
    ref = wide[:, FREQUENCIES.index(REF_FREQ)]
    others = [f for f in FREQUENCIES if f != REF_FREQ]
    p = [wilcoxon(ref, wide[:, FREQUENCIES.index(f)]).pvalue for f in others]
    reject, p_holm, _, _ = multipletests(p, method="holm")
    for f, pv, ph, rj in zip(others, p, p_holm, reject):
        print(f"  {REF_FREQ} vs {f:<3} Hz:  P = {pv:.4f}  P(Holm) = {ph:.4f}"
              f"  {'significant' if rj else 'n.s.'}")


def friedman_tests(df, band_df):
    print(f"\n{SEP}\n5. FRIEDMAN ACROSS FREQUENCIES  (patient as block, N = 9)\n{SEP}")
    rows = [("Dims 1+2 distance", df, METRIC),
            ("Norm. clinical improvement", df, OUTCOME)]
    rows += [(info["label"], band_df, f"delta_{name}") for name, info in BANDS.items()]
    for label, d, col in rows:
        stat, p = friedmanchisquare(*by_frequency(d, col).T)
        print(f"  {label:<30}  χ² = {stat:6.2f}  P = {p:.4f}")


def main():
    df = pd.read_csv(COMBINED_CSV)
    df["condition"] = df["condition"].astype(int)
    df = df[df["condition"].isin(FREQUENCIES)]

    auxiliary_screen(df)
    table_s3(df)
    sign_tests(df)
    wilcoxon_vs_ref(df)

    print(f"\n{SEP}\nBuilding OFF-referenced band-power changes ...")
    band_df = build_beta_spearman_df(BETA_CONFIG)
    friedman_tests(df, band_df)


if __name__ == "__main__":
    main()
