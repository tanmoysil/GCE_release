"""
Figure 4: P11 prospective patient — stimulation detection and therapeutic encoding
  Panel A — Bar chart of dim_0_signed across 8 conditions (4 freqs × 2 amplitudes)
  Panel B — Scatter: dims_1_2_sum_signed vs norm_rigidity, both centred within
             amplitude, coloured by frequency, marker by amplitude (○=full, △=half);
             Spearman ρ with one-sided exact within-amplitude permutation P
             (primary endpoint, see analysis/prospective_patient_test.py)

Data: working_plots/prospective_results/displacements_combined_pat11.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from analysis.prospective_patient_test import within_amp_permutation_p

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "Roboto", "font.weight": "medium",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

OUT_DIR = os.path.join(BASE_DIR, "working_plots")
DATA_FILE = os.path.join(BASE_DIR, "working_plots", "prospective_results",
                         "displacements_combined_pat11.csv")

FREQ_ORDER = [125, 110, 85, 55]
FREQ_COLORS = {125: "#0173B2", 110: "#DE8F05", 85: "#029E73", 55: "#D55E00"}
FREQ_LABELS = {125: "125 Hz", 110: "110 Hz", 85: "85 Hz", 55: "55 Hz"}

# Amplitude colors for Panel A — Okabe-Ito
AMP_COLORS = {"half": "#0173B2", "full": "#DE8F05"}
AMP_LABELS = {"half": "Half amplitude", "full": "Full amplitude"}
AMP_MARKERS = {"full": "o", "half": "^"}


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
    df = df.rename(columns={
        "dims12_signed": "dims_1_2_sum_signed",
        "score_rigidity": "norm_rigidity",
    })

    # Centre within amplitude for Panel B (mirrors run_combined_amp_test)
    for col in ["dims_1_2_sum_signed", "norm_rigidity"]:
        df[col + "_dm"] = df[col] - df.groupby("amp")[col].transform("mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: Bar chart dim_0_signed × 8 conditions ──────────────────────
    ax = axes[0]

    bar_width = 0.35
    x = np.arange(len(FREQ_ORDER))

    for i, amp in enumerate(["half", "full"]):
        sub = df[df["amp"] == amp].set_index("freq")
        vals = [float(sub.loc[freq, "dim0_signed"]) for freq in FREQ_ORDER]
        offset = (i - 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width,
               color=AMP_COLORS[amp], label=AMP_LABELS[amp],
               alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(0, color="#B0B0B0", linestyle="--", linewidth=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in FREQ_ORDER], fontsize=12,
                       fontfamily="Roboto", fontweight="medium")
    ax.set_ylabel("Dim 0 signed displacement from OFF", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.set_title("P11: Stimulation detection (Dim 0)\n— all conditions positive",
                 fontsize=14, fontweight="bold", fontfamily="Roboto")
    ax.legend(fontsize=11, framealpha=0.8,
              prop={"family": "Roboto", "weight": "medium"})
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.11, 1.06, "A", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    # ── Panel B: Scatter dims₁₂ (demeaned) vs norm_rigidity (demeaned) ──────
    ax = axes[1]

    # Plot each (freq, amp) point — color by freq, marker by amp
    points = {}  # (freq, amp) -> (x, y) for label placement
    for amp in ["full", "half"]:
        for freq in FREQ_ORDER:
            row = df[(df["amp"] == amp) & (df["freq"] == freq)]
            if row.empty:
                continue
            xv = float(row["dims_1_2_sum_signed_dm"].iloc[0])
            yv = float(row["norm_rigidity_dm"].iloc[0])
            ax.scatter(xv, yv,
                       color=FREQ_COLORS[freq], marker=AMP_MARKERS[amp],
                       s=130, edgecolors="black", linewidths=0.8, alpha=0.95, zorder=3)
            points[(freq, amp)] = (xv, yv)

    # Frequency labels near each marker — color matches frequency, nudged to avoid overlap
    x_range = df["dims_1_2_sum_signed_dm"].max() - df["dims_1_2_sum_signed_dm"].min()
    y_range = df["norm_rigidity_dm"].max() - df["norm_rigidity_dm"].min()
    x_off = x_range * 0.04
    # Nudge full amp label up, half amp label down so they don't overlap for
    # 55 Hz and 85 Hz where the two points are nearly identical in x.
    amp_y_nudge = {"full": y_range * 0.055, "half": -y_range * 0.055}
    for (freq, amp), (xv, yv) in points.items():
        ax.annotate(
            FREQ_LABELS[freq],
            xy=(xv, yv), xytext=(xv + x_off, yv + amp_y_nudge[amp]),
            fontsize=9, color=FREQ_COLORS[freq],
            fontfamily="Roboto", fontweight="medium",
            va="center", ha="left", zorder=4,
        )

    # Legend: amplitude shapes only (colors explained by labels)
    for amp, marker in AMP_MARKERS.items():
        ax.scatter([], [], color="gray", marker=marker, s=100,
                   edgecolors="black", linewidths=0.8, label=AMP_LABELS[amp])

    # OLS regression line on demeaned data
    x_dm = df["dims_1_2_sum_signed_dm"].values
    y_dm = df["norm_rigidity_dm"].values
    z = np.polyfit(x_dm, y_dm, 1)
    x_fit = np.linspace(x_dm.min(), x_dm.max(), 200)
    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.5, zorder=2)

    # Primary endpoint: Spearman ρ, one-sided exact within-amplitude permutation P
    rho, p_val, _ = within_amp_permutation_p(x_dm, y_dm, df["amp"].values)
    sig = _sig_marker(p_val)
    ax.text(0.05, 0.95,
            f"ρ = {rho:.2f}{sig}\nP = {p_val:.3f} (one-sided)",
            transform=ax.transAxes, fontsize=11, va="top",
            fontfamily="Roboto", fontweight="medium",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xlabel("Dims₁₂ signed displacement from OFF (centred within amplitude)", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.set_ylabel("Normalized rigidity improvement (centred within amplitude)", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.set_title("P11: Therapeutic encoding — rigidity",
                 fontsize=14, fontweight="bold", fontfamily="Roboto")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.8,
              prop={"family": "Roboto", "weight": "medium"})
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.11, 1.06, "B", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figure4.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "figure4.svg"), bbox_inches="tight", format="svg")
    plt.close()
    print("Saved figure4.pdf and figure4.svg")
    print(f"  Panel B: Spearman rho={rho:.3f}, one-sided exact permutation P={p_val:.4f}{sig}")


if __name__ == "__main__":
    main()
