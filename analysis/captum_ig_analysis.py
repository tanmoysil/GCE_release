"""
Feature attribution for the latent dimensions — Fig. 2H (Dim 0) and Fig. 3C (Dims 1, 2)
=====================================================================================
GradientSHAP (Captum) on the encoder posterior mean µ of each latent dimension,
with OFF-condition segments as the baseline distribution. Attributions are
computed for all ON-condition segments (4 frequencies × 2 amplitudes, 1008
segments) and averaged; each amplitude is normalised per patient over all its
conditions, and its own OFF segments serve as the baseline. PDC LR/RL feature
pairs are summed after averaging.

Convergence deltas are reported as a quality check (Methods); feature rankings
for the auxiliary dimensions are qualitative.

Output (working_plots/captum_ig_results/), top 5 positive / top 5 negative features,
laid out to the manuscript panel sizes (Myriad Pro):
  fig2h_attribution_dim0.svg/.pdf/.png      Fig. 2H, Dim 0
  fig3c_attribution_dims12.svg/.pdf/.png    Fig. 3C, Dims 1 and 2
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gce.utils import select_vae_model
from config.paths import BEST_ENCODER_PT, FULLAMP_ALL_CONDITIONS_MAT, HALFAMP_ALL_CONDITIONS_MAT


# Editable text in SVG (Inkscape) and PDF (Illustrator)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# Tick/spine styling
matplotlib.rcParams["axes.linewidth"] = 1.2
matplotlib.rcParams["xtick.major.width"] = 1.2
matplotlib.rcParams["ytick.major.width"] = 1.2
matplotlib.rcParams["xtick.major.size"] = 4
matplotlib.rcParams["ytick.major.size"] = 4


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "data_paths": {"half": HALFAMP_ALL_CONDITIONS_MAT, "full": FULLAMP_ALL_CONDITIONS_MAT},
    "model_path": BEST_ENCODER_PT,
    "output_dir": os.path.join(REPO_ROOT, "working_plots", "captum_ig_results"),
    "frequencies": [125, 110, 85, 55],
    "K": 1,  # stimulation-aligned dims
    "L": 5,  # auxiliary dims
    "n_features": 70,
    "seed": 42,  # GradientSHAP samples baselines (NumPy) and interpolation points (torch)
}

FEATURE_NAMES = [
    # FOOOF spectral power (5)
    "power_alpha",
    "power_low-beta",
    "power_high-beta",
    "power_low-gamma",
    "power_finely-tuned-gamma",
    # Nonlinear dynamics — alpha (4)
    "alpha_lyapunov",
    "alpha_corr_dim",
    "alpha_sample_entropy",
    "alpha_permutation_entropy",
    # Nonlinear dynamics — low beta (4)
    "low-beta_lyapunov",
    "low-beta_corr_dim",
    "low-beta_sample_entropy",
    "low-beta_permutation_entropy",
    # Nonlinear dynamics — high beta (4)
    "high-beta_lyapunov",
    "high-beta_corr_dim",
    "high-beta_sample_entropy",
    "high-beta_permutation_entropy",
    # Nonlinear dynamics — low gamma (4)
    "low-gamma_lyapunov",
    "low-gamma_corr_dim",
    "low-gamma_sample_entropy",
    "low-gamma_permutation_entropy",
    # Nonlinear dynamics — finely-tuned gamma (4)
    "finely-tuned-gamma_lyapunov",
    "finely-tuned-gamma_corr_dim",
    "finely-tuned-gamma_sample_entropy",
    "finely-tuned-gamma_permutation_entropy",
    # Burst dynamics — alpha (6)
    "alpha_burst_mean_dur_long",
    "alpha_burst_mean_dur_short",
    "alpha_burst_mean_amp_long",
    "alpha_burst_mean_amp_short",
    "alpha_burst_rate_long",
    "alpha_burst_rate_short",
    # Burst dynamics — low beta (6)
    "low-beta_burst_mean_dur_long",
    "low-beta_burst_mean_dur_short",
    "low-beta_burst_mean_amp_long",
    "low-beta_burst_mean_amp_short",
    "low-beta_burst_rate_long",
    "low-beta_burst_rate_short",
    # Burst dynamics — high beta (6)
    "high-beta_burst_mean_dur_long",
    "high-beta_burst_mean_dur_short",
    "high-beta_burst_mean_amp_long",
    "high-beta_burst_mean_amp_short",
    "high-beta_burst_rate_long",
    "high-beta_burst_rate_short",
    # Burst dynamics — low gamma (6)
    "low-gamma_burst_mean_dur_long",
    "low-gamma_burst_mean_dur_short",
    "low-gamma_burst_mean_amp_long",
    "low-gamma_burst_mean_amp_short",
    "low-gamma_burst_rate_long",
    "low-gamma_burst_rate_short",
    # Burst dynamics — finely-tuned gamma (6)
    "finely-tuned-gamma_burst_mean_dur_long",
    "finely-tuned-gamma_burst_mean_dur_short",
    "finely-tuned-gamma_burst_mean_amp_long",
    "finely-tuned-gamma_burst_mean_amp_short",
    "finely-tuned-gamma_burst_rate_long",
    "finely-tuned-gamma_burst_rate_short",
    # Phase-Amplitude Coupling (5)
    "PAC_low-beta_x_low-gamma",
    "PAC_low-beta_x_finely-tuned-gamma",
    "PAC_high-beta_x_finely-tuned-gamma",
    "PAC_alpha_x_low-gamma",
    "PAC_alpha_x_finely-tuned-gamma",
    # Partial Directed Coherence — L→R and R→L (10)
    "PDC_alpha_LR",
    "PDC_alpha_RL",
    "PDC_low-beta_LR",
    "PDC_low-beta_RL",
    "PDC_high-beta_LR",
    "PDC_high-beta_RL",
    "PDC_low-gamma_LR",
    "PDC_low-gamma_RL",
    "PDC_finely-tuned-gamma_LR",
    "PDC_finely-tuned-gamma_RL",
]

assert len(FEATURE_NAMES) == 70, f"Expected 70 features, got {len(FEATURE_NAMES)}"


# =============================================================================
# MODEL WRAPPER
# =============================================================================
class EncoderLatentDim(torch.nn.Module):
    """
    Wraps the VAE encoder to expose a single latent dimension as a scalar output.
    This is what Captum differentiates through.

    Returns mu[:, dim_idx] — signed, so attributions preserve therapeutic direction.
    """

    def __init__(self, encoder, dim_idx: int):
        super().__init__()
        self.encoder = encoder
        self.dim_idx = dim_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, mu, _ = self.encoder(x)
        return mu[:, self.dim_idx]  # shape: (batch,)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(config):
    print(f"Loading data from {config['data_path']}")
    mat = loadmat(config["data_path"])

    X_list, conditions, subjects = [], [], []
    for freq in config["frequencies"]:
        Xk = mat[f"X_{freq}_cleaned"]
        ik = mat[f"idx_{freq}_cleaned"].flatten()
        X_list.append(Xk)
        conditions.extend([str(freq)] * len(Xk))
        subjects.extend(ik.tolist())

    Xoff = mat["X_off_cleaned"]
    ioff = mat["idx_off_cleaned"].flatten()
    X_list.append(Xoff)
    conditions.extend(["off"] * len(Xoff))
    subjects.extend(ioff.tolist())

    X = np.vstack(X_list).astype(np.float32)
    conditions = np.array(conditions)
    subjects = np.array(subjects)
    print(f"  Total samples: {len(X)}, features: {X.shape[1]}")
    return X, conditions, subjects


def patient_normalise(X, subjects):
    """Replicate the patient-specific normalisation from your training pipeline."""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for pat in np.unique(subjects):
        mask = subjects == pat
        if mask.sum() > 1:
            scaler = StandardScaler()
            X_norm[mask] = scaler.fit_transform(X[mask])
        else:
            X_norm[mask] = X[mask]
    return X_norm


def load_encoder(config):
    base_cfg = {
        "vae_model": "vanilla_Encoder",
        "n_features": config["n_features"],
        "n_alpha": config["K"],
        "n_beta": config["L"],
        "z_dim": config["K"] + config["L"],
        "include_classes": None,
        "save_dir": "./",
        "model_name": "vae_model",
    }
    encoder, _ = select_vae_model(base_cfg)
    encoder.load_state_dict(torch.load(config["model_path"], map_location="cpu"))
    encoder.eval()
    print(
        f"Encoder loaded: K={config['K']}, L={config['L']}, z_dim={config['K'] + config['L']}"
    )
    return encoder


# =============================================================================
# ATTRIBUTION
# =============================================================================
def compute_attributions(encoder, X_on, X_off, config):
    """
    GradientSHAP for every latent dimension: inputs are the ON segments,
    baselines are drawn from the OFF segments of the same amplitude.

    Returns
    -------
    attrs : dict[dim_label -> np.ndarray (n_on, n_features)], signed
    deltas : dict[dim_label -> np.ndarray], convergence deltas
    """
    z_dim = config["K"] + config["L"]
    attrs, deltas = {}, {}
    for dim in range(z_dim):
        dim_label = f"dim_{dim}" + ("_causal" if dim < config["K"] else "_aux")
        explainer = GradientShap(EncoderLatentDim(encoder, dim).eval())
        a, d = explainer.attribute(
            inputs=torch.FloatTensor(X_on),
            baselines=torch.FloatTensor(X_off),
            return_convergence_delta=True,
        )
        attrs[dim_label] = a.detach().cpu().numpy()
        deltas[dim_label] = d.detach().cpu().numpy()
    return attrs, deltas


def summarise(attrs, feature_names):
    """Mean signed and mean |attribution| per feature and dimension."""
    rows = []
    for dim_label, a in attrs.items():
        mean_signed, mean_abs = a.mean(axis=0), np.abs(a).mean(axis=0)
        for fi, fname in enumerate(feature_names):
            rows.append({"feature": fname, "dim": dim_label,
                         "mean_signed": float(mean_signed[fi]),
                         "mean_abs": float(mean_abs[fi])})
    return pd.DataFrame(rows)


# PDC LR/RL pairs to sum — matches how correlations were computed
PDC_LR_RL_PAIRS = [
    ("PDC_alpha_LR", "PDC_alpha_RL", "PDC_alpha"),
    ("PDC_low-beta_LR", "PDC_low-beta_RL", "PDC_low-beta"),
    ("PDC_high-beta_LR", "PDC_high-beta_RL", "PDC_high-beta"),
    ("PDC_low-gamma_LR", "PDC_low-gamma_RL", "PDC_low-gamma"),
    (
        "PDC_finely-tuned-gamma_LR",
        "PDC_finely-tuned-gamma_RL",
        "PDC_finely-tuned-gamma",
    ),
]


def aggregate_pdc_lr_rl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum LR and RL attributions for each PDC band into a single row.
    Works on both df_global (dim only) and df_cond (dim + condition).
    Signed values are summed; abs values are recomputed from the summed signed.
    """
    # Group keys are whatever index columns exist
    group_cols = [c for c in ["dim", "condition"] if c in df.columns]
    val_cols = [c for c in ["mean_signed", "mean_abs"] if c in df.columns]

    rows_to_drop = []
    rows_to_add = []

    for lr_name, rl_name, summed_name in PDC_LR_RL_PAIRS:
        lr_mask = df["feature"] == lr_name
        rl_mask = df["feature"] == rl_name
        if not lr_mask.any() or not rl_mask.any():
            continue

        rows_to_drop.extend(df.index[lr_mask].tolist())
        rows_to_drop.extend(df.index[rl_mask].tolist())

        # Merge LR and RL on group keys, then sum
        lr_sub = df[lr_mask][group_cols + val_cols]
        rl_sub = df[rl_mask][group_cols + val_cols]
        merged = lr_sub.merge(rl_sub, on=group_cols, suffixes=("_lr", "_rl"))

        for _, row in merged.iterrows():
            new_row = {col: row[col] for col in group_cols}
            new_row["feature"] = summed_name
            if "mean_signed" in val_cols:
                new_row["mean_signed"] = row["mean_signed_lr"] + row["mean_signed_rl"]
            if "mean_abs" in val_cols:
                # Recompute abs from summed signed for consistency
                new_row["mean_abs"] = abs(
                    row.get("mean_signed_lr", 0) + row.get("mean_signed_rl", 0)
                )
            rows_to_add.append(new_row)

    df_out = df.drop(index=rows_to_drop).reset_index(drop=True)
    if rows_to_add:
        df_out = pd.concat([df_out, pd.DataFrame(rows_to_add)], ignore_index=True)

    n_replaced = len(PDC_LR_RL_PAIRS) - sum(
        1
        for lr, rl, _ in PDC_LR_RL_PAIRS
        if not (df["feature"] == lr).any() or not (df["feature"] == rl).any()
    )
    print(f"  PDC aggregation: {n_replaced}/{len(PDC_LR_RL_PAIRS)} pairs summed")
    return df_out


# =============================================================================
# FIGURE (Fig. 2H, Fig. 3C)
# =============================================================================
MM = 1 / 25.4  # inches per mm

# Panel layouts measured from the assembled manuscript figures, so the exported
# SVG drops into place without rescaling. Sizes: mm (geometry) and pt (fonts, lines).
PANELS = {
    "fig2h_attribution_dim0": {
        "dims": ["dim_0_causal"],
        "axes_mm": (29.72, 65.19),       # width, height of each axes
        "gap_mm": 0.0,
        "xmin_floor": -0.05,             # left x-limit unless data go further
        "feature_pt": 5.80, "feature_weight": "medium",
        "tick_pt": 5.80, "value_pt": 4.55, "xlabel_pt": 8.0, "xlabel_weight": "normal",
        "spine_lw": 0.61, "tick_len": 2.0, "tick_pad": 1.7, "grid_lw": 0.40,
        "zero_lw": 0.51, "sep_lw": 0.41, "bar_edge_lw": 0.09,
    },
    "fig3c_attribution_dims12": {
        "dims": ["dim_1_aux", "dim_2_aux"],
        "axes_mm": (74.35, 109.1),
        "gap_mm": 78.25,                 # right edge of Dim 1 to left edge of Dim 2
        "xmin_floor": -0.10,
        "feature_pt": 12.0, "feature_weight": "normal",
        "tick_pt": 10.80, "value_pt": 10.43, "xlabel_pt": 12.36, "xlabel_weight": "medium",
        "spine_lw": 1.20, "tick_len": 3.4, "tick_pad": 2.8, "grid_lw": 0.98,
        "zero_lw": 0.85, "sep_lw": 0.98, "bar_edge_lw": 0.43,
    },
}
X_PAD = 0.0075  # x-limit padding beyond the longest bar (data units)


def plot_attribution_panels(df_global, name, style, output_dir, top_n=5):
    """
    Top top_n positive (blue) and top_n negative (vermillion) features per
    dimension, by mean signed attribution. One axes per dimension, all with the
    same x-limits; axes are placed in absolute mm so the panel matches the
    manuscript layout. A value label that would cross the axes' left edge is
    drawn inside its bar instead.
    """
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Myriad Pro", "DejaVu Sans"]})
    dims = style["dims"]
    tops = {}
    for d in dims:
        sub = df_global[df_global["dim"] == d]
        neg = sub.nsmallest(top_n, "mean_signed").sort_values("mean_signed", ascending=False)
        pos = sub.nlargest(top_n, "mean_signed").sort_values("mean_signed")
        tops[d] = pd.concat([neg, pos])
    all_vals = pd.concat(tops.values())["mean_signed"]
    xlim = (min(style["xmin_floor"], all_vals.min() - X_PAD), all_vals.max() + X_PAD)

    w, h = style["axes_mm"]
    left, bottom = 60.0, 15.0                       # generous margins; trimmed on save
    fig_w = left + len(dims) * w + (len(dims) - 1) * style["gap_mm"] + 25.0
    fig = plt.figure(figsize=(fig_w * MM, (bottom + h + 5.0) * MM))

    for k, d in enumerate(dims):
        x0 = left + k * (w + style["gap_mm"])
        ax = fig.add_axes([x0 / fig_w, bottom / (bottom + h + 5.0),
                           w / fig_w, h / (bottom + h + 5.0)])
        feats = tops[d]["feature"].tolist()
        vals = tops[d]["mean_signed"].tolist()
        ax.barh(range(len(vals)), vals, height=0.7,
                color=["#0173B2" if v >= 0 else "#D55E00" for v in vals],
                edgecolor="white", linewidth=style["bar_edge_lw"])
        ax.axvline(0, color="black", linewidth=style["zero_lw"], zorder=3)
        ax.axhline(top_n - 0.5, color="#AAAAAA", linewidth=style["sep_lw"],
                   linestyle="--", zorder=2)
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=style["feature_pt"],
                           fontweight=style["feature_weight"])
        ax.tick_params(axis="x", labelsize=style["tick_pt"])
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("medium")
        ax.tick_params(length=style["tick_len"], width=style["spine_lw"],
                       pad=style["tick_pad"])
        ax.grid(axis="x", color="#b0b0b0", alpha=0.25, linewidth=style["grid_lw"])
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(style["spine_lw"])
        ax.set_xlabel("Feature importance", fontsize=style["xlabel_pt"],
                      fontweight=style["xlabel_weight"])

        renderer = fig.canvas.get_renderer()
        ax_left_px = ax.get_window_extent(renderer).x0
        for i, v in enumerate(vals):
            off = 0.002 if v >= 0 else -0.002
            t = ax.text(v + off, i, f"{v:.3f}", va="center",
                        ha="left" if v >= 0 else "right",
                        fontsize=style["value_pt"], fontweight="medium",
                        color="#333333", zorder=4)
            if v < 0 and t.get_window_extent(renderer).x0 < ax_left_px:
                t.set_x(v + 0.002)          # does not fit left of the bar: draw inside
                t.set_ha("left")

    base = os.path.join(output_dir, name)
    for ext in ("svg", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", pad_inches=0.5 * MM)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", pad_inches=0.5 * MM)
    plt.close(fig)
    print(f"  Saved: {name}.svg/.pdf/.png  (x-limits {xlim[0]:.4f} to {xlim[1]:.4f})")


# =============================================================================
# MAIN
# =============================================================================
def run_captum_ig(config=CONFIG):
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])  # Captum draws GradientSHAP baselines with NumPy
    encoder = load_encoder(config)

    per_amp_attrs, per_amp_deltas = [], []
    for amp, path in config["data_paths"].items():
        X, conditions, subjects = load_data({**config, "data_path": path})
        X_norm = patient_normalise(X, subjects)
        on = conditions != "off"
        print(f"  {amp} amplitude: {on.sum()} ON segments, "
              f"{(~on).sum()} OFF segments as baseline")
        a, d = compute_attributions(encoder, X_norm[on], X_norm[~on], config)
        per_amp_attrs.append(a)
        per_amp_deltas.append(d)

    attrs = {k: np.vstack([a[k] for a in per_amp_attrs]) for k in per_amp_attrs[0]}
    deltas = {k: np.concatenate([d[k] for d in per_amp_deltas]) for k in per_amp_deltas[0]}
    n_on = len(next(iter(attrs.values())))
    print(f"\nAttributions over {n_on} ON segments")

    print("\nConvergence check (mean |delta| vs mean |attribution|):")
    for k in attrs:
        mean_delta = float(np.abs(deltas[k]).mean())
        mean_attr = float(np.abs(attrs[k]).mean())
        print(f"  {k:<14} |delta| = {mean_delta:.4f}   |attr| = {mean_attr:.4f}   "
              f"ratio = {mean_delta / mean_attr:.1f}x")

    df_global = aggregate_pdc_lr_rl(summarise(attrs, FEATURE_NAMES))

    print("\nTop 5 positive / negative features (mean signed attribution):")
    for k in ["dim_0_causal", "dim_1_aux", "dim_2_aux"]:
        s = df_global[df_global["dim"] == k].set_index("feature")["mean_signed"]
        print(f"  {k}")
        print(f"    +  {', '.join(s.nlargest(5).index)}")
        print(f"    -  {', '.join(s.nsmallest(5).index)}")

    for name, style in PANELS.items():
        plot_attribution_panels(df_global, name, style, config["output_dir"])
    return df_global


if __name__ == "__main__":
    run_captum_ig(CONFIG)
