"""
Figure 3: Full-amplitude latent displacement vs motor scores (panels a + e)
  Panel A — Box-and-whisker + jittered strip of dims_1_2_sum_signed by frequency
  Panel B — Scatter: dims_1_2_sum_signed vs norm_total (Spearman + OLS)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "Roboto", "font.weight": "medium",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(BASE_DIR, "working_plots")
DATA_FILE = os.path.join(BASE_DIR, "analysis", "fullamp_correlation_analysis", "fullamp_combined_data.csv")

# Okabe-Ito palette — matches prospective_patient_test.py / latent_axis_validation.py
COLORS = {125: "#0173B2", 110: "#DE8F05", 85: "#029E73", 55: "#D55E00"}
FREQ_ORDER = [125, 110, 85, 55]
FREQ_LABELS = ["125 Hz", "110 Hz", "85 Hz", "55 Hz"]
Y_COL = "dims_1_2_sum_signed"


def _sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "·"
    return ""


def main():
    df = pd.read_csv(DATA_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Box-and-whisker + jittered strip ───────────────────────────
    ax = axes[0]
    rng = np.random.default_rng(42)

    all_y = df[Y_COL].values
    y_data_min = all_y.min()
    annot_y = y_data_min - 0.50

    data_by_freq = [df[df["condition"] == freq][Y_COL].values for freq in FREQ_ORDER]

    # Box plot — gray transparent box, black median (cosine similarity style)
    ax.boxplot(data_by_freq, positions=range(len(FREQ_ORDER)), widths=0.5,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=(160/255, 160/255, 160/255),
                             alpha=0.3, linewidth=1.5),
               medianprops=dict(color="black", linewidth=2.5),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               zorder=2)

    # Jittered scatter points on top, colored per frequency
    for i, freq in enumerate(FREQ_ORDER):
        y_vals = data_by_freq[i]
        x_jit = i + rng.uniform(-0.15, 0.15, size=len(y_vals))
        ax.scatter(x_jit, y_vals, color=COLORS[freq], s=120, alpha=0.85,
                   edgecolors="black", linewidths=1.5, zorder=3)

    ax.axhline(0, color="#B0B0B0", linestyle="--", linewidth=1.2, zorder=1)

    ax.text(3, annot_y, "all 9 patients negative",
            style="italic", fontsize=10, color="#D55E00",
            ha="center", va="top", fontfamily="Roboto")

    ax.set_xticks(range(len(FREQ_ORDER)))
    ax.set_xticklabels(FREQ_LABELS, fontsize=12, fontfamily="Roboto", fontweight="medium")
    ax.set_ylabel("Signed displacement from OFF (dims₁₂)", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(annot_y - 0.25, all_y.max() + 0.55)
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.13, 1.05, "A", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    # ── Panel B: Scatter ────────────────────────────────────────────────────
    ax = axes[1]

    for freq in FREQ_ORDER:
        sub = df[df["condition"] == freq]
        ax.scatter(sub[Y_COL], sub["norm_total"],
                   color=COLORS[freq], s=120, alpha=0.85,
                   edgecolors="black", linewidths=1.5, zorder=3,
                   label=f"{freq} Hz")

    # OLS regression line
    x_all = df[Y_COL].values
    y_all = df["norm_total"].values
    z = np.polyfit(x_all, y_all, 1)
    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.5, zorder=2)

    # Spearman correlation
    rho, p_val = spearmanr(x_all, y_all)
    sig = _sig_marker(p_val)
    ax.text(0.05, 0.95,
            f"ρ = {rho:.2f}{sig}\np = {p_val:.3f}",
            transform=ax.transAxes, fontsize=11, va="top",
            fontfamily="Roboto", fontweight="medium",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.axvline(0, color="#B0B0B0", linestyle=":", linewidth=1.0, zorder=1)
    ax.axhline(0, color="#B0B0B0", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xlabel("Signed displacement from OFF (dims₁₂)", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.set_ylabel("Normalized total score improvement", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.7,
              prop={"family": "Roboto", "weight": "medium"})
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.13, 1.05, "B", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figure3.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "figure3.svg"), bbox_inches="tight", format="svg")
    plt.close()
    print("Saved figure3.pdf and figure3.svg")
    print(f"  Panel B: Spearman rho={rho:.3f}, p={p_val:.4f}{sig}")


if __name__ == "__main__":
    main()