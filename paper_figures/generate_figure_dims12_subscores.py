"""
Dim 1 and Dim 2 signed displacement vs motor subscores (full-amplitude cohort)
  _make_figure      — one figure per dim, overlays brady + rigidity on same axes
  _make_total_figure — single figure, overlays dim 1 + dim 2 vs total score
  filled markers + solid OLS   = dim 1  (or bradykinesia in per-dim figures)
  hollow markers + dashed OLS  = dim 2  (or rigidity in per-dim figures)
  Color encodes stimulation frequency (Okabe-Ito, matching other paper figures).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import spearmanr, pearsonr, t as t_dist

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "Roboto", "font.weight": "medium",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = os.path.join(BASE_DIR, "working_plots")
DATA_FILE  = os.path.join(BASE_DIR, "analysis", "fullamp_correlation_analysis", "fullamp_combined_data.csv")

COLORS     = {125: (230/255, 159/255, 0/255),
              110: (86/255,  180/255, 233/255),
              85:  (0/255,   158/255, 115/255),
              55:  (213/255, 94/255,  0/255)}
MARKERS    = {125: "o", 110: "s", 85: "^", 55: "D"}
FREQ_ORDER = [125, 110, 85, 55]

# (score column, short label for stat box, OLS linestyle)
SCORES = [
    ("norm_brady",    "bradykinesia", "-"),
    ("norm_rigidity", "rigidity",     "--"),
]


def _sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "·"
    return ""


def _make_figure(df, dim_idx: int):
    x_col  = f"dim_{dim_idx}_signed"
    xlabel = f"Dim {dim_idx} signed displacement from OFF"

    fig, ax = plt.subplots(figsize=(7, 5.5))

    stat_lines = []

    for score_col, label, ols_ls in SCORES:
        filled = (ols_ls == "-")   # bradykinesia → filled; rigidity → hollow

        for freq in FREQ_ORDER:
            sub = df[df["condition"] == freq]
            xv  = sub[x_col].values
            yv  = sub[score_col].values

            if filled:
                ax.scatter(xv, yv,
                           color=COLORS[freq], marker=MARKERS[freq],
                           s=120, alpha=0.85,
                           edgecolors="black", linewidths=1.5, zorder=3)
            else:
                ax.scatter(xv, yv,
                           facecolors="none", edgecolors=COLORS[freq],
                           marker=MARKERS[freq],
                           s=120, alpha=0.85,
                           linewidths=1.5, zorder=3)

        x_all = df[x_col].values
        y_all = df[score_col].values

        z     = np.polyfit(x_all, y_all, 1)
        x_fit = np.linspace(x_all.min(), x_all.max(), 200)
        y_fit = np.polyval(z, x_fit)
        ax.plot(x_fit, y_fit, "k" + ols_ls, linewidth=2.0, alpha=0.8, zorder=2)

        # 95 % CI band around OLS line
        n      = len(x_all)
        x_mean = x_all.mean()
        SSx    = np.sum((x_all - x_mean) ** 2)
        resid  = y_all - np.polyval(z, x_all)
        mse    = np.sum(resid ** 2) / (n - 2)
        se_fit = np.sqrt(mse * (1 / n + (x_fit - x_mean) ** 2 / SSx))
        t_crit = t_dist.ppf(0.975, df=n - 2)
        ci_lo, ci_hi = y_fit - t_crit * se_fit, y_fit + t_crit * se_fit

        if filled:
            ax.fill_between(x_fit, ci_lo, ci_hi, color="black", alpha=0.20, zorder=1)
        else:
            ax.fill_between(x_fit, ci_lo, ci_hi, color="black", alpha=0.20,
                            hatch="///", zorder=1)

        rho, p_sp = spearmanr(x_all, y_all)
        r,   p_pe = pearsonr(x_all, y_all)
        stat_lines.append(
            f"{label}: ρ={rho:.2f}{_sig_marker(p_sp)} p={p_sp:.3f}, "
            f"r={r:.2f}{_sig_marker(p_pe)} p={p_pe:.3f}"
        )
        print(f"  Dim {dim_idx} | {score_col}: "
              f"Spearman rho={rho:.3f} p={p_sp:.4f}  "
              f"Pearson r={r:.3f} p={p_pe:.4f}")

    ax.text(0.05, 0.95, "\n".join(stat_lines),
            transform=ax.transAxes, fontsize=10, va="top",
            fontfamily="Roboto", fontweight="medium",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.axvline(0, color="#B0B0B0", linestyle=":", linewidth=1.0, zorder=1)
    ax.axhline(0, color="#B0B0B0", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xlabel(xlabel, fontsize=13, fontweight="medium", fontfamily="Roboto")
    ax.set_ylabel("Normalized score improvement", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legend: frequency colors + score-type fill ──────────────────────────
    freq_handles = [
        mlines.Line2D([], [], color=COLORS[f], marker=MARKERS[f],
                      linestyle="none", markersize=8,
                      markeredgecolor="black", markeredgewidth=1.2,
                      label=f"{f} Hz")
        for f in FREQ_ORDER
    ]
    score_handles = [
        mlines.Line2D([], [], color="black", marker="o", linestyle="none",
                      markersize=8, markeredgecolor="black", markeredgewidth=1.2,
                      label="bradykinesia (filled)"),
        mlines.Line2D([], [], color="none", marker="o", linestyle="none",
                      markersize=8, markeredgecolor="black", markeredgewidth=1.5,
                      label="rigidity (hollow)"),
        mlines.Line2D([], [], color="black", linestyle="-",
                      linewidth=2.0, label="bradykinesia OLS ± 95% CI"),
        mlines.Line2D([], [], color="black", linestyle="--",
                      linewidth=2.0, label="rigidity OLS ± 95% CI"),
    ]
    ax.legend(handles=freq_handles + score_handles,
              loc="lower right", fontsize=11, framealpha=0.7,
              prop={"family": "Roboto", "weight": "medium"})

    plt.tight_layout()

    stem = f"figure_dim{dim_idx}_subscores"
    plt.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, f"{stem}.svg"), bbox_inches="tight", format="svg")
    plt.close()
    print(f"Saved {stem}.pdf and {stem}.svg")


def _make_total_figure(df, dim_idx: int):
    """Single-axes scatter: dim N signed displacement vs normalized total score."""
    x_col  = f"dim_{dim_idx}_signed"
    xlabel = f"Dim {dim_idx} signed displacement from OFF"

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for freq in FREQ_ORDER:
        sub = df[df["condition"] == freq]
        ax.scatter(sub[x_col], sub["norm_total"],
                   color=COLORS[freq], marker=MARKERS[freq],
                   s=120, alpha=0.85,
                   edgecolors="black", linewidths=1.5, zorder=3,
                   label=f"{freq} Hz")

    x_all = df[x_col].values
    y_all = df["norm_total"].values

    z     = np.polyfit(x_all, y_all, 1)
    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    y_fit = np.polyval(z, x_fit)
    ax.plot(x_fit, y_fit, "k-", linewidth=2.0, alpha=0.8, zorder=2)

    n      = len(x_all)
    x_mean = x_all.mean()
    SSx    = np.sum((x_all - x_mean) ** 2)
    resid  = y_all - np.polyval(z, x_all)
    mse    = np.sum(resid ** 2) / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (x_fit - x_mean) ** 2 / SSx))
    t_crit = t_dist.ppf(0.975, df=n - 2)
    ci_lo, ci_hi = y_fit - t_crit * se_fit, y_fit + t_crit * se_fit
    ax.fill_between(x_fit, ci_lo, ci_hi, color="black", alpha=0.20, zorder=1)

    rho, p_sp = spearmanr(x_all, y_all)
    r,   p_pe = pearsonr(x_all, y_all)
    ax.text(0.05, 0.95,
            f"ρ={rho:.2f}{_sig_marker(p_sp)} p={p_sp:.3f}\n"
            f"r={r:.2f}{_sig_marker(p_pe)} p={p_pe:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            fontfamily="Roboto", fontweight="medium",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    print(f"  Dim {dim_idx} | norm_total: "
          f"Spearman rho={rho:.3f} p={p_sp:.4f}  "
          f"Pearson r={r:.3f} p={p_pe:.4f}")

    ax.axvline(0, color="#B0B0B0", linestyle=":", linewidth=1.0, zorder=1)
    ax.axhline(0, color="#B0B0B0", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xlabel(xlabel, fontsize=13, fontweight="medium", fontfamily="Roboto")
    ax.set_ylabel("Normalized total score improvement", fontsize=13,
                  fontweight="medium", fontfamily="Roboto")
    ax.tick_params(axis="both", labelsize=12, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.7,
              prop={"family": "Roboto", "weight": "medium"})

    plt.tight_layout()

    stem = f"figure_dim{dim_idx}_total"
    plt.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, f"{stem}.svg"), bbox_inches="tight", format="svg")
    plt.close()
    print(f"Saved {stem}.pdf and {stem}.svg")


def main():
    df = pd.read_csv(DATA_FILE)
    for dim_idx in (1, 2):
        _make_figure(df, dim_idx)
    for dim_idx in (1, 2):
        _make_total_figure(df, dim_idx)


if __name__ == "__main__":
    main()
