#!/usr/bin/env python3
"""
seed_stability_test.py — Fig. S2

Reproducibility of the latent structure across random weight initializations.
The VAE is retrained with K=1, L=5, λ=0.068 from several random seeds (same
frozen classifier, same data split). For each retrained encoder:

  A. Therapeutic subspace. Each of the C(5,2)=10 auxiliary-dim pairs (dims 1–5)
     is scored as the summed OFF-referenced displacement d_a + d_b and
     correlated (Spearman) with normalized total motor score on full-amplitude
     data. The best |ρ| is tested against a selection-aware permutation null:
     scores are shuffled across stimulation conditions within each patient and
     the maximum |ρ| over the same 10 pairs is recorded. A retrain passes if
     p < 0.05.
  B. Stimulation detection. Per-dimension AUROC for ON vs OFF; passes if Dim 0
     ranks first among all six dimensions.

By default the nine reported encoders are scored from
working_plots/seed_stability/encoders/rng6/ (no training needed).

Usage:
    python analysis/seed_stability_test.py                        # reproduce Fig. S2
    python analysis/seed_stability_test.py --train --config CFG   # retrain the seeds
"""

import os
import json
import hashlib
import argparse
import random
from itertools import combinations

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from gce.utils import (
    select_vae_model,
    select_optimizer,
    VAE_LL_loss,
    load_classifier_all,
    select_dataloader,
    to_vae_config,
    prepare_variables_pl,
    log_metrics,
)
from gce.causal.effect import joint_uncond
from gce.utils_results import get_latent_representations
from analysis.patient_specific_norm_hemi_fullamp import load_data, load_normalized_scores
from config.paths import FULLAMP_ALL_CONDITIONS_MAT, SCORES_FULLAMP_XLSX

# ── Fixed architecture & search space ────────────────────────────────────────
K, L, LAMB = 1, 5, 0.068
Z_DIM = K + L
FREQUENCIES = [125, 110, 85, 55]
BETA_DIMS  = list(range(K, K + L))            # [1, 2, 3, 4, 5] — auxiliary dims
BETA_PAIRS = list(combinations(BETA_DIMS, 2)) # C(5,2) = 10 pairs

SEEDS  = [44, 53, 51, 34, 94, 36, 65, 37, 98]  # the nine reported retrains
ALPHA  = 0.05     # per-retrain significance threshold
N_PERM = 5000     # within-patient permutations per retrain
NULL_SEED = 12345 # RNG for the permutation draws (independent of training)

REF_RHO = 0.497   # reported result, drawn as a reference line

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_YML = os.path.join(_REPO_ROOT, "config", "pd_all_save.yml")
OUT_DIR    = os.path.join(_REPO_ROOT, "working_plots", "seed_stability")
ENCODER_DIR = os.path.join(OUT_DIR, "encoders", "rng6")


# =============================================================================
# Config
# =============================================================================

def load_cfg(config_path=None):
    raw = yaml.safe_load(open(config_path or CONFIG_YML))
    cfg = to_vae_config(raw)
    cfg = prepare_variables_pl(cfg)
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable — using CPU")
        cfg["device"] = "cpu"
    return cfg


# =============================================================================
# Training (one seed)
# =============================================================================

def train_one_seed(seed, cfg, classifier, train_loader, test_loader):
    """Train the VAE from scratch for one random seed; return (encoder, nll, causal)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = cfg["device"]
    run_cfg = cfg.copy()
    run_cfg["z_dim"] = K + L
    run_cfg["n_alpha"] = K
    run_cfg["n_beta"] = L
    params = {"K": K, "L": L, "lam": LAMB}

    encoder, decoder = select_vae_model(run_cfg)
    encoder, decoder = encoder.to(device), decoder.to(device)
    optimizer = select_optimizer(run_cfg, encoder, decoder)
    writer = SummaryWriter(
        log_dir=os.path.join(_REPO_ROOT, "pd_logs", f"stability_seed_{seed}")
    )

    for epoch in range(run_cfg["epochs"]):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            latent_out, mu, logvar = encoder(imgs)
            x_gen = decoder(latent_out)
            nll, _, _ = VAE_LL_loss(imgs, x_gen, logvar, mu)
            causal_effect, _ = joint_uncond(run_cfg, decoder, classifier, device)
            loss = run_cfg["use_causal"] * causal_effect + LAMB * nll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log_metrics(writer, params, epoch, total_loss, mode="Train")

        encoder.eval()
        decoder.eval()
        t_nll = t_causal = t_loss = n_b = 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                latent_out, mu, logvar = encoder(imgs)
                x_gen = decoder(latent_out)
                nll, _, _ = VAE_LL_loss(imgs, x_gen, logvar, mu)
                causal_effect, _ = joint_uncond(run_cfg, decoder, classifier, device)
                loss = run_cfg["use_causal"] * causal_effect + LAMB * nll
                t_nll += nll.item()
                t_causal += causal_effect.item()
                t_loss += loss.item()
                n_b += 1
        log_metrics(writer, params, epoch, t_loss, mode="Test")

    writer.flush()
    b = max(n_b, 1)
    return encoder, t_nll / b, t_causal / b


def encoder_cache_path(seed, cfg):
    """Cache file keyed by seed AND the full training config (classifier, data,
    epochs, MC samples, device, ...), so changing --config never reuses an
    encoder trained under a different setup."""
    cfg_hash = hashlib.sha1(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    return os.path.join(_REPO_ROOT, "working_plots", "seed_stability", "encoders",
                        f"seed_{seed}_K{K}_L{L}_lamb{LAMB:.3f}_cfg{cfg_hash}.pt")


def get_encoder(seed, cfg, classifier, train_loader, test_loader, encoder_dir=None):
    """Train (or load the cached) encoder for a seed. Returns (encoder, nll, causal).

    Caching lets the cheap statistics be redone without retraining; delete the
    cache file to force a retrain.

    With encoder_dir, load DIR/encoder_seed{seed}.pt (a bare state_dict) and
    never train; nll/causal are then NaN.
    """
    if encoder_dir is not None:
        run_cfg = cfg.copy()
        run_cfg.update(z_dim=K + L, n_alpha=K, n_beta=L)
        encoder, _ = select_vae_model(run_cfg)
        encoder.load_state_dict(torch.load(
            os.path.join(encoder_dir, f"encoder_seed{seed}.pt"), map_location="cpu"))
        encoder.eval()
        return encoder, float("nan"), float("nan")

    path = encoder_cache_path(seed, cfg)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        run_cfg = cfg.copy()
        run_cfg.update(z_dim=K + L, n_alpha=K, n_beta=L)
        encoder, _ = select_vae_model(run_cfg)
        encoder.load_state_dict(blob["encoder"])
        encoder.eval()
        print(f"  [cache] loaded {os.path.basename(path)}")
        return encoder, blob["nll"], blob["causal"]

    encoder, nll, causal = train_one_seed(seed, cfg, classifier, train_loader, test_loader)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "nll": nll, "causal": causal}, path)
    return encoder, nll, causal


# =============================================================================
# Displacement & correlation
# =============================================================================

def compute_displacements(latent_mu, conditions, subject_indices, normalized_scores):
    """Per-(patient, stimulation condition) centroid shift from OFF state."""
    z_dim = latent_mu.shape[1]
    rows = []
    for pat in np.unique(subject_indices):
        pmask = subject_indices == pat
        off_mask = pmask & (conditions == "off")
        if not np.any(off_mask):
            continue
        off_cent = latent_mu[off_mask].mean(axis=0)

        for freq in FREQUENCIES:
            cond_str = str(freq)
            cmask = pmask & (conditions == cond_str)
            if not np.any(cmask):
                continue
            cond_cent = latent_mu[cmask].mean(axis=0)

            row = {"patient": int(pat), "condition": cond_str}
            for d in range(z_dim):
                row[f"dim_{d}"] = float(cond_cent[d] - off_cent[d])

            sc = normalized_scores.get(int(pat), {}).get(cond_str, {})
            row["norm_total"] = sc.get("total", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def pair_predictors(df):
    """One predictor per auxiliary pair: d_a + d_b. Returns (n_obs × 10, labels)."""
    cols = [df[f"dim_{a}"].to_numpy() + df[f"dim_{b}"].to_numpy() for a, b in BETA_PAIRS]
    return np.column_stack(cols), BETA_PAIRS


def max_abs_rho(X, y):
    """Return (max_j |Spearman ρ(X[:, j], y)|, argmax j); NaN rows dropped per column."""
    best, best_j = 0.0, None
    y_ok = ~np.isnan(y)
    for j in range(X.shape[1]):
        v = y_ok & ~np.isnan(X[:, j])
        if v.sum() < 3:
            continue
        rho = spearmanr(X[v, j], y[v])[0]
        if np.isfinite(rho) and abs(rho) > best:
            best, best_j = abs(rho), j
    return best, best_j


def within_patient_null(df, X, n_perm, rng):
    """Max-|ρ| null over the same candidate set, shuffling each patient's scores
    across that patient's stimulation conditions."""
    y = df["norm_total"].to_numpy()
    groups = [np.flatnonzero(df["patient"].to_numpy() == p) for p in df["patient"].unique()]
    out = np.empty(n_perm)
    for i in range(n_perm):
        yy = y.copy()
        for g in groups:
            yy[g] = y[rng.permutation(g)]
        out[i] = max_abs_rho(X, yy)[0]
    return out


def therapeutic_subspace(df, n_perm, rng):
    """Best |ρ|, its pair, the selection-aware p-value, and the null's 95th pct."""
    X, labels = pair_predictors(df)
    best, j = max_abs_rho(X, df["norm_total"].to_numpy())
    null = within_patient_null(df, X, n_perm, rng)
    p = (1 + int((null >= best).sum())) / (n_perm + 1)
    return best, (labels[j] if j is not None else None), p, float(np.quantile(null, 0.95))


def dim_aurocs(latent_mu, conditions):
    """Per-dim AUROC for ON (any stimulation) vs OFF, orientation-free."""
    labels = (conditions != "off").astype(int)
    return [max(a, 1.0 - a) for a in
            (roc_auc_score(labels, latent_mu[:, d]) for d in range(latent_mu.shape[1]))]


# =============================================================================
# Figure S2
# =============================================================================

def make_figure(res):
    seeds = [str(s) for s in res["seed"]]
    xs = np.arange(len(res))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    # A — therapeutic subspace
    pairs = sorted(set(res["best_pair"]))
    palette = plt.cm.tab10.colors
    color = {p: palette[i % len(palette)] for i, p in enumerate(pairs)}
    for i, r in res.iterrows():
        c = color[r["best_pair"]]
        ax.scatter(i, r["best_rho"], s=90, zorder=3, edgecolor=c, linewidth=1.6,
                   facecolor=c if r["p"] < ALPHA else "none")
        ax.text(i, max(r["best_rho"], r["null95"]) + 0.015, r["best_pair"],
                ha="center", va="bottom",
                fontsize=8, color=c)
    ax.scatter(xs, res["null95"], marker="_", s=300, color="0.45", zorder=2,
               label="Null 95th percentile (per retrain)")
    ax.axhline(REF_RHO, color="steelblue", linestyle=":", lw=1.4,
               label=f"Reported model  |ρ| = {REF_RHO}")
    ax.set_xticks(xs)
    ax.set_xticklabels(seeds, fontsize=8)
    ax.set_xlabel(f"Initialization seed  (filled: P < {ALPHA})", fontsize=10)
    ax.set_ylabel("Best |ρ|  (2D auxiliary subspace vs\nnormalized total motor score)",
                  fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.set_title("A   Therapeutic subspace", loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # B — stimulation detection
    auroc = np.array(res["aurocs"].tolist())
    im = ax2.imshow(auroc.T, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    for s in range(len(res)):
        d = int(np.argmax(auroc[s]))
        ax2.add_patch(plt.Rectangle((s - 0.5, d - 0.5), 1, 1, fill=False,
                                    edgecolor="black", lw=2, zorder=5))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(seeds, fontsize=8)
    ax2.set_xlabel("Initialization seed  (box: best detector)", fontsize=10)
    ax2.set_yticks(range(Z_DIM))
    ax2.set_yticklabels([f"Dim {d}" for d in range(Z_DIM)], fontsize=9)
    ax2.set_title("B   ON vs OFF detection (AUROC)", loc="left", fontsize=11,
                  fontweight="bold")
    fig.colorbar(im, ax=ax2, label="AUROC", shrink=0.85)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = os.path.join(OUT_DIR, f"figS2_seed_stability.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  Saved {path}")
    plt.close(fig)


# =============================================================================
# Main run
# =============================================================================

def run(seeds, n_perm=N_PERM, train=False, config_path=None, encoder_dir=ENCODER_DIR):
    seeds = list(dict.fromkeys(seeds))  # a repeated seed is the same model
    cfg = load_cfg(config_path)
    classifier = train_loader = test_loader = None
    if train:
        encoder_dir = None
        print("Loading classifier (frozen, shared across all seeds)...")
        classifier = load_classifier_all(cfg)
        classifier.to(cfg["device"]).eval()
        for p in classifier.parameters():
            p.requires_grad = False
        print("Loading data split (numpy seed=42, deterministic)...")
        np.random.seed(42)
        train_loader, test_loader = select_dataloader(cfg)
    else:
        print(f"Scoring encoders from {encoder_dir}")

    X_all, conditions, subject_indices = load_data(FULLAMP_ALL_CONDITIONS_MAT, FREQUENCIES)
    normalized_scores = load_normalized_scores(SCORES_FULLAMP_XLSX, FREQUENCIES)

    rows = []
    null_rng = np.random.default_rng(NULL_SEED)
    for i, s in enumerate(seeds):
        print(f"\n── Seed {s}  ({i + 1}/{len(seeds)}) ──")
        encoder, nll, causal = get_encoder(s, cfg, classifier, train_loader, test_loader,
                                           encoder_dir=encoder_dir)
        encoder.cpu()
        latent_mu = get_latent_representations(encoder, X_all, subject_indices)
        df = compute_displacements(latent_mu, conditions, subject_indices, normalized_scores)
        rho, pair, p, null95 = therapeutic_subspace(df, n_perm, null_rng)
        aurocs = dim_aurocs(latent_mu, conditions)
        dim0_best = int(np.argmax(aurocs)) == 0
        print(f"  best |ρ| = {rho:.4f}  pair = {pair}  p = {p:.4f}  null95 = {null95:.3f}  "
              f"[{'PASS' if p < ALPHA else 'fail'}]   Dim 0 AUROC = {aurocs[0]:.3f} "
              f"[{'best' if dim0_best else 'not best'}]")
        rows.append({"seed": s, "best_rho": rho, "best_pair": str(pair), "p": p,
                     "null95": null95, "dim0_auroc": aurocs[0], "dim0_best": dim0_best,
                     "aurocs": aurocs, "nll": nll, "causal": causal})

    res = pd.DataFrame(rows)
    n = len(res)
    n_sig = int((res["p"] < ALPHA).sum())
    sep = "=" * 62
    print(f"\n{sep}\nFIG. S2 SUMMARY  (K={K}, L={L}, λ={LAMB}, {n} retrains, "
          f"{n_perm} permutations)\n{sep}")
    print(f"Therapeutic subspace significant (P < {ALPHA}): {n_sig}/{n}")
    print(f"  best |ρ| median = {res['best_rho'].median():.3f}  "
          f"range = {res['best_rho'].min():.2f}–{res['best_rho'].max():.2f}")
    print("  winning pairs: " + ", ".join(
        f"{k} {v}/{n}" for k, v in res["best_pair"].value_counts().items()))
    n_main = int((res["best_pair"] == str((1, 2))).sum())
    p_main = binomtest(n_main, n, 1 / len(BETA_PAIRS), alternative="greater").pvalue
    print(f"  Dims 1+2 (main-analysis pair) won {n_main}/{n}; one-sided binomial "
          f"vs 1/{len(BETA_PAIRS)}: P = {p_main:.3f}")
    print(f"Dim 0 best ON/OFF detector: {int(res['dim0_best'].sum())}/{n}")
    print(f"  Dim 0 AUROC median = {res['dim0_auroc'].median():.3f}  "
          f"range = {res['dim0_auroc'].min():.3f}–{res['dim0_auroc'].max():.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = res.drop(columns="aurocs").join(
        pd.DataFrame(res["aurocs"].tolist(), columns=[f"auroc_dim{d}" for d in range(Z_DIM)]))
    csv_path = os.path.join(OUT_DIR, "figS2_seed_stability.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nCSV  : {csv_path}")
    make_figure(res)


def main():
    parser = argparse.ArgumentParser(description="Fig. S2: reproducibility across seeds.")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (default: the nine reported retrains).")
    parser.add_argument("--n-perm", type=int, default=N_PERM,
                        help=f"Within-patient permutations per retrain (default: {N_PERM}).")
    parser.add_argument("--encoder-dir", default=ENCODER_DIR,
                        help="Directory of encoder_seed{seed}.pt state_dicts to score.")
    parser.add_argument("--train", action="store_true",
                        help="Retrain each seed instead of loading saved encoders.")
    parser.add_argument("--config", default=None,
                        help="YAML config for --train (default: config/pd_all_save.yml).")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    run(seeds, n_perm=args.n_perm, train=args.train, config_path=args.config,
        encoder_dir=args.encoder_dir)


if __name__ == "__main__":
    main()
