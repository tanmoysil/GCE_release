"""
Per-dimension LOSO AUC figure
==============================
Extends the existing run_loso_dim0 in latent_axis_validation.py to all K+L
dimensions. Produces two publication-ready figures:

  Figure A — Bar chart: mean LOSO AUC per dim (9-fold bootstrap 95 % CI).
              Dim 0 (causal / stimulation axis) coloured blue,
              Dims 1–5 (auxiliary / therapeutic) coloured orange.
              Chance line at 0.5.

  Figure B — Mean ROC overlay: Dim 0 mean ROC (blue) vs auxiliary dims mean
              ROC (orange), individual fold curves in light gray, 95 % CI bands,
              and chance line.  One panel per dataset (halfamp / fullamp).

Usage
-----
  python analysis/fig_loso_per_dim_auc.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch

from gce.utils import select_vae_model
from config.paths import HALFAMP_MAT, FULLAMP_MAT, BEST_ENCODER_PT

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

# ── config ────────────────────────────────────────────────────────────────────
K           = 1           # causal dims (Dim 0)
L           = 5           # auxiliary dims (Dims 1–5)
N_BOOTSTRAP = 2000
OUTPUT_DIR  = "./fig_loso_dim_auc_results"

BLUE   = "#0173B2"   # causal dim 0
ORANGE = "#DE8F05"   # auxiliary dims 1–5
GRAY   = "#AAAAAA"   # individual folds / chance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder() -> torch.nn.Module:
    cfg = {
        "vae_model": "vanilla_Encoder",
        "n_alpha": K,
        "n_beta":  L,
        "z_dim":   K + L,
        "include_classes": None,
        "save_dir": "./", "model_name": "vae_model",
        "n_features": 70,
    }
    enc, _ = select_vae_model(cfg)
    enc.load_state_dict(torch.load(BEST_ENCODER_PT, map_location="cpu"))
    enc.eval()
    print(f"Encoder  K={K}  L={L}  "
          f"z_dim={K+L}  {os.path.basename(BEST_ENCODER_PT)}")
    return enc


def patient_normalise(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Per-patient StandardScaler — mirrors the training pipeline."""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for pat in np.unique(subjects):
        mask = subjects == pat
        X_norm[mask] = (StandardScaler().fit_transform(X[mask])
                        if mask.sum() > 1 else X[mask])
    return X_norm


def load_binary_mat(mat_path: str):
    """Load a binary ON/OFF mat and return X_norm, y, subjects."""
    mat      = loadmat(mat_path)
    X        = mat["X_cleaned"].astype(np.float32)
    y        = mat["y_cleaned"].flatten().astype(int)
    subjects = mat["idx_cleaned"].flatten().astype(int)
    return patient_normalise(X, subjects), y, subjects


def get_mu(encoder: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        _, mu, _ = encoder(torch.FloatTensor(X))
    return mu.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# LOSO per dimension
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_all_dims(encoder: torch.nn.Module,
                      mat_path: str,
                      dataset_label: str) -> list[dict]:
    """
    Run LOSO logistic regression for each latent dimension independently.

    For each dim d in 0…K+L-1:
      - In each fold, StandardScaler fitted on training subjects' dim-d values only.
      - Logistic regression C=1.0 on the single scalar feature.
    Returns a list of dicts (one per dim) with keys:
      dim, aucs, patients, fpr_grid, tpr_interps, mean_auc, ci_95
    """
    X_norm, y, subjects = load_binary_mat(mat_path)
    mu_all   = get_mu(encoder, X_norm)          # (n, z_dim)
    z_dim    = K + L
    patients = sorted(np.unique(subjects).tolist())
    fpr_grid = np.linspace(0, 1, 200)
    rng      = np.random.default_rng(42)

    results = []
    print(f"\n── LOSO per dim ({dataset_label}) ──")

    for d in range(z_dim):
        d_vals    = mu_all[:, d : d + 1]
        aucs, tpr_interps = [], []

        for held_out in patients:
            tr_mask = subjects != held_out
            te_mask = subjects == held_out

            scaler = StandardScaler()
            d_tr   = scaler.fit_transform(d_vals[tr_mask])
            d_te   = scaler.transform(d_vals[te_mask])

            clf = LogisticRegression(max_iter=50000, C=1.0)
            clf.fit(d_tr, y[tr_mask])

            proba = clf.predict_proba(d_te)[:, 1]
            # handle perfectly separated case (AUC could be 0 or 1 — both valid)
            auc   = roc_auc_score(y[te_mask], proba)
            fpr, tpr, _ = roc_curve(y[te_mask], proba)
            tpr_interps.append(np.interp(fpr_grid, fpr, tpr))
            aucs.append(auc)

        boot_means = [np.mean(rng.choice(aucs, size=len(aucs), replace=True))
                      for _ in range(N_BOOTSTRAP)]
        ci = (float(np.percentile(boot_means, 2.5)),
              float(np.percentile(boot_means, 97.5)))

        dim_type = "causal" if d < K else "auxiliary"
        print(f"  dim{d} ({dim_type}): mean AUC = {np.mean(aucs):.3f}  "
              f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"per-fold: {[f'{a:.2f}' for a in aucs]}")

        results.append({
            "dim":         d,
            "aucs":        aucs,
            "patients":    patients,
            "fpr_grid":    fpr_grid,
            "tpr_interps": np.array(tpr_interps),
            "mean_auc":    float(np.mean(aucs)),
            "ci_95":       ci,
            "dataset_label": dataset_label,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure A — AUC bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_dim_auc_bar(loso_list: list[dict], output_dir: str,
                     filename_suffix: str = ""):
    """
    Bar chart: mean LOSO AUC per dim with 95 % CI error bars.
    Dim 0 (causal) in blue, Dims 1–5 (auxiliary) in orange.
    """
    z_dim        = len(loso_list)
    dataset_label = loso_list[0]["dataset_label"]
    dims         = [r["dim"] for r in loso_list]
    means        = [r["mean_auc"] for r in loso_list]
    ci_lows      = [r["mean_auc"] - r["ci_95"][0] for r in loso_list]
    ci_highs     = [r["ci_95"][1] - r["mean_auc"] for r in loso_list]
    colors       = [BLUE if d < K else ORANGE for d in dims]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        dims, means,
        yerr=[ci_lows, ci_highs],
        color=colors, edgecolor="k", linewidth=0.7,
        error_kw={"elinewidth": 1.5, "capsize": 5, "ecolor": "k"},
        zorder=3,
    )

    # Chance line
    ax.axhline(0.5, color=GRAY, linestyle=(0, (6, 4)), lw=1.4, alpha=0.75,
               label="Chance (0.5)", zorder=2)

    # Labels on bars
    for bar, mean_auc in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean_auc + max(ci_highs) * 1.1 + 0.01,
                f"{mean_auc:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(dims)
    ax.set_xticklabels(
        [f"Dim {d}\n({'causal' if d < K else 'auxiliary'})" for d in dims],
        fontsize=10,
    )
    ax.set_ylabel("Mean LOSO AUC (9-fold)", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"LOSO Per-dimension AUC — {dataset_label}\n"
        f"(K={K} causal, L={L} auxiliary)",
        fontweight="bold", fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)

    legend_elements = [
        mpatches.Patch(facecolor=BLUE,   edgecolor="k", label=f"Dim 0 (causal / stim axis)"),
        mpatches.Patch(facecolor=ORANGE, edgecolor="k", label=f"Dims 1–{K+L-1} (auxiliary)"),
        plt.Line2D([0], [0], color=GRAY, linestyle=(0, (6, 4)), lw=1.4, label="Chance"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    suf = f"_{filename_suffix}" if filename_suffix else ""
    base = os.path.join(output_dir, f"figA_loso_dim_auc_bar{suf}")
    plt.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight", format="svg")
    plt.close()
    print(f"  Saved: {base}.png/.svg")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Mean ROC overlay (Dim 0 vs auxiliary group mean)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dim_roc_overlay(loso_list: list[dict], output_dir: str,
                         filename_suffix: str = ""):
    """
    ROC overlay: Dim 0 mean ROC (blue, thick) vs auxiliary dims mean ROC (orange,
    thick), with individual fold curves (gray, thin) for Dim 0 and ±1 SD bands.
    """
    dataset_label = loso_list[0]["dataset_label"]
    fpr_grid      = loso_list[0]["fpr_grid"]
    z_dim         = len(loso_list)

    dim0   = loso_list[0]
    aux    = loso_list[1:]   # dims 1–5

    # Auxiliary group mean (average across dims then folds)
    aux_tpr_stacked = np.stack([r["tpr_interps"] for r in aux], axis=0)  # (L, 9, 200)
    aux_mean_tpr    = aux_tpr_stacked.mean(axis=(0, 1))                   # mean over dims and folds
    aux_std_tpr     = aux_tpr_stacked.mean(axis=0).std(axis=0)           # std over folds of mean-dim
    aux_mean_auc    = float(np.mean([r["mean_auc"] for r in aux]))

    dim0_tpr_arr  = dim0["tpr_interps"]   # (9, 200)
    dim0_mean_tpr = dim0_tpr_arr.mean(axis=0)
    dim0_std_tpr  = dim0_tpr_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Individual fold curves for Dim 0 (gray, thin)
    for tpr in dim0_tpr_arr:
        ax.plot(fpr_grid, tpr, color=GRAY, lw=0.7, alpha=0.4)

    # Dim 0 mean + ±1 SD band
    ax.fill_between(fpr_grid,
                    np.clip(dim0_mean_tpr - dim0_std_tpr, 0, 1),
                    np.clip(dim0_mean_tpr + dim0_std_tpr, 0, 1),
                    color=BLUE, alpha=0.15)
    ax.plot(fpr_grid, dim0_mean_tpr, color=BLUE, lw=2.5,
            label=f"Dim 0 (causal)  AUC={dim0['mean_auc']:.3f} "
                  f"[{dim0['ci_95'][0]:.3f},{dim0['ci_95'][1]:.3f}]")

    # Auxiliary group mean + ±1 SD band
    ax.fill_between(fpr_grid,
                    np.clip(aux_mean_tpr - aux_std_tpr, 0, 1),
                    np.clip(aux_mean_tpr + aux_std_tpr, 0, 1),
                    color=ORANGE, alpha=0.15)
    ax.plot(fpr_grid, aux_mean_tpr, color=ORANGE, lw=2.5,
            label=f"Dims 1–{z_dim-1} mean (auxiliary)  AUC={aux_mean_auc:.3f}")

    # Individual auxiliary dim mean ROCs (thin orange)
    for r in aux:
        ax.plot(fpr_grid, r["tpr_interps"].mean(axis=0),
                color=ORANGE, lw=0.9, alpha=0.35)

    # Chance
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (6, 4)), lw=1.2, alpha=0.7,
            label="Chance")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(
        f"LOSO ROC — Dim 0 vs Auxiliary Dims\n"
        f"({dataset_label}, K={K}, L={L}, 9-fold)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    suf = f"_{filename_suffix}" if filename_suffix else ""
    base = os.path.join(output_dir, f"figB_loso_roc_overlay{suf}")
    plt.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight", format="svg")
    plt.close()
    print(f"  Saved: {base}.png/.svg")


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Combined half + full ROC overlay
# ─────────────────────────────────────────────────────────────────────────────

RED = "#CC3311"   # half-amplitude

def plot_combined_roc_overlay(loso_half: list[dict], loso_full: list[dict],
                               output_dir: str):
    """
    Single plot combining halfamp and fullamp LOSO ROC curves.

    Line styles
    -----------
    Full-amp  Dim 0            blue  solid   lw=2.5
    Half-amp  Dim 0            red   solid   lw=2.5
    Full-amp  auxiliary mean   blue  dashed  lw=1.8
    Half-amp  auxiliary mean   red   dashed  lw=1.8
    Chance                     gray  dashed  lw=1.2
    """
    fpr_grid = loso_half[0]["fpr_grid"]

    def _dim0_mean(loso_list):
        return loso_list[0]["tpr_interps"].mean(axis=0), loso_list[0]["mean_auc"]

    def _aux_mean(loso_list):
        aux = loso_list[1:]
        stacked = np.stack([r["tpr_interps"] for r in aux], axis=0)  # (L, 9, 200)
        mean_tpr = stacked.mean(axis=(0, 1))
        mean_auc = float(np.mean([r["mean_auc"] for r in aux]))
        return mean_tpr, mean_auc

    dim0_half_tpr, dim0_half_auc = _dim0_mean(loso_half)
    dim0_full_tpr, dim0_full_auc = _dim0_mean(loso_full)
    aux_half_tpr,  aux_half_auc  = _aux_mean(loso_half)
    aux_full_tpr,  aux_full_auc  = _aux_mean(loso_full)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Dim 0 lines (solid)
    ax.plot(fpr_grid, dim0_full_tpr, color=BLUE, lw=2.5, linestyle="-",
            label=f"Dim 0 — full amp  (AUC={dim0_full_auc:.3f})")
    ax.plot(fpr_grid, dim0_half_tpr, color=RED,  lw=2.5, linestyle="-",
            label=f"Dim 0 — half amp  (AUC={dim0_half_auc:.3f})")

    # Auxiliary mean lines (dashed)
    ax.plot(fpr_grid, aux_full_tpr, color=BLUE, lw=1.8, linestyle="--",
            label=f"Aux mean — full amp  (AUC={aux_full_auc:.3f})")
    ax.plot(fpr_grid, aux_half_tpr, color=RED,  lw=1.8, linestyle="--",
            label=f"Aux mean — half amp  (AUC={aux_half_auc:.3f})")

    # Chance
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (6, 4)), lw=1.2, alpha=0.7,
            label="Chance")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(
        f"LOSO ROC — Dim 0 vs Auxiliary Dims\n"
        f"(half-amp = red, full-amp = blue,  K={K}, L={L}, 9-fold)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    base = os.path.join(output_dir, "figC_loso_roc_combined")
    plt.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight", format="svg")
    plt.close()
    print(f"  Saved: {base}.png/.svg")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    encoder = load_encoder()

    results_by_dataset = {}
    for mat_path, label in [(HALFAMP_MAT, "halfamp"), (FULLAMP_MAT, "fullamp")]:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {label}  ({mat_path})")
        print(f"{'=' * 60}")

        loso_list = run_loso_all_dims(encoder, mat_path, label)
        results_by_dataset[label] = loso_list

        plot_dim_auc_bar(loso_list, OUTPUT_DIR, filename_suffix=label)
        plot_dim_roc_overlay(loso_list, OUTPUT_DIR, filename_suffix=label)

    plot_combined_roc_overlay(
        results_by_dataset["halfamp"],
        results_by_dataset["fullamp"],
        OUTPUT_DIR,
    )

    print(f"\nAll figures saved to {OUTPUT_DIR}/")