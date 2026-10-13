"""
Dim 0 signed displacement vs motor subscores (full-amplitude cohort)
  2×2 panels: total | rigidity | bradykinesia | tremor
  Each panel: scatter coloured by frequency + OLS line + Spearman ρ
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "Roboto", "font.weight": "medium",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = os.path.join(BASE_DIR, "working_plots")
DATA_FILE  = os.path.join(BASE_DIR, "analysis", "fullamp_correlation_analysis", "fullamp_combined_data.csv")

COLORS   = {125: (230/255, 159/255, 0/255),   # Orange
            110: (86/255,  180/255, 233/255),  # Sky blue
            85:  (0/255,   158/255, 115/255),  # Bluish green
            55:  (213/255, 94/255,  0/255)}    # Vermillion
MARKERS  = {125: "o", 110: "s", 85: "^", 55: "D"}
FREQ_ORDER = [125, 110, 85, 55]

SUBSCORES = [
    ("norm_total",    "Normalized total score improvement"),
    ("norm_rigidity", "Normalized rigidity score improvement"),
    ("norm_brady",    "Normalized bradykinesia score improvement"),
    ("norm_tremor",   "Normalized tremor score improvement"),
]
PANEL_LABELS = ["A", "B", "C", "D"]


def _sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "·"
    return ""


def _scatter_panel(ax, df, score_col, ylabel, panel_label):
    for freq in FREQ_ORDER:
        sub = df[df["condition"] == freq]
        ax.scatter(sub["dim_0_signed"], sub[score_col],
                   color=COLORS[freq], marker=MARKERS[freq],
                   s=120, alpha=0.85,
                   edgecolors="black", linewidths=1.5, zorder=3,
                   label=f"{freq} Hz")

    x_all = df["dim_0_signed"].values
    y_all = df[score_col].values

    # OLS regression line
    z     = np.polyfit(x_all, y_all, 1)
    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(x_fit, np.polyval(z, x_fit), "k-", linewidth=2.5, alpha=0.8, zorder=2)

    rho, p_sp = spearmanr(x_all, y_all)
    r,   p_pe = pearsonr(x_all, y_all)
    ax.text(0.05, 0.95,
            f"ρ = {rho:.2f}{_sig_marker(p_sp)},  p = {p_sp:.3f}\n"
            f"r = {r:.2f}{_sig_marker(p_pe)},  p = {p_pe:.3f}",
            transform=ax.transAxes, fontsize=11, va="top",
            fontfamily="Roboto", fontweight="medium",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xlabel("Dim 0 signed displacement from OFF", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.set_ylabel(ylabel, fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.13, 1.05, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    return rho, p_sp, r, p_pe


def main():
    df = pd.read_csv(DATA_FILE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat  = axes.flatten()

    for ax, (score_col, ylabel), panel_label in zip(axes_flat, SUBSCORES, PANEL_LABELS):
        rho, p_sp, r, p_pe = _scatter_panel(ax, df, score_col, ylabel, panel_label)
        print(f"  {score_col}: Spearman rho={rho:.3f} p={p_sp:.4f}  Pearson r={r:.3f} p={p_pe:.4f}")

    # Single shared legend in the first panel
    axes_flat[0].legend(loc="lower right", fontsize=11, framealpha=0.7,
                        prop={"family": "Roboto", "weight": "medium"})

    plt.tight_layout()
    out_svg = os.path.join(OUT_DIR, "figure_dim0_subscores.svg")
    out_pdf = os.path.join(OUT_DIR, "figure_dim0_subscores.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight", format="svg")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
