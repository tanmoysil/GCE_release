"""
Latent Space Geometry Analysis: Half-Amplitude vs Full-Amplitude
Examines preservation of latent space structure across stimulation amplitudes
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy.io import loadmat
import torch
from sklearn.preprocessing import StandardScaler
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from gce.utils_results import *
from config.paths import (
    HALFAMP_ALL_CONDITIONS_MAT, FULLAMP_ALL_CONDITIONS_MAT, BEST_ENCODER_PT
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data paths
    'halfamp_data_path': HALFAMP_ALL_CONDITIONS_MAT,
    'fullamp_data_path': FULLAMP_ALL_CONDITIONS_MAT,
    'model_path': BEST_ENCODER_PT,
    'output_dir': './latent_geometry_analysis',

    # Analysis parameters
    'frequencies': [125, 110, 85, 55],
}


@dataclass
class ModelParams:
    subject_idx: int
    K: int
    L: int
    lamb: float
    filepath: str


# =============================================================================
# DATA LOADING
# =============================================================================

def load_amplitude_data(data_path: str, frequencies: List[int], amplitude_label: str):
    """Load neural data for one amplitude condition"""
    print(f"Loading {amplitude_label} data from {data_path}")
    mat_data = loadmat(data_path)
    X_list, idx_list, condition_labels = [], [], []

    for freq in frequencies:
        freq_key = f'X_{freq}_cleaned'
        idx_key = f'idx_{freq}_cleaned'
        if freq_key in mat_data and idx_key in mat_data:
            X_list.append(mat_data[freq_key])
            idx_list.append(mat_data[idx_key].flatten())
            condition_labels.append(np.full(len(mat_data[freq_key]), str(freq)))
            print(f"  {freq} Hz: {len(mat_data[freq_key])} samples")

    X_list.append(mat_data['X_off_cleaned'])
    idx_list.append(mat_data['idx_off_cleaned'].flatten())
    condition_labels.append(np.full(len(mat_data['X_off_cleaned']), 'off'))
    print(f"  OFF: {len(mat_data['X_off_cleaned'])} samples")

    return np.vstack(X_list), np.hstack(condition_labels), np.hstack(idx_list)


# =============================================================================
# LATENT SPACE ANALYSIS
# =============================================================================

def compute_condition_centroids(latent_dims: np.ndarray, conditions: np.ndarray,
                                subject_indices: np.ndarray, frequencies: List[int]) -> Dict:
    """
    Compute mean latent position for each condition for each patient

    Returns:
        Dict[patient][condition] = mean latent vector (z_dim,)
    """
    print("Computing condition centroids...")
    centroids = {}

    for patient in np.unique(subject_indices):
        patient_mask = subject_indices == patient
        centroids[patient] = {}

        # Include OFF and all frequencies
        all_conditions = ['off'] + [str(f) for f in frequencies]

        for condition in all_conditions:
            combined_mask = patient_mask & (conditions == condition)
            if np.any(combined_mask):
                centroids[patient][condition] = np.mean(latent_dims[combined_mask], axis=0)

    return centroids


def compute_position_correlations(halfamp_centroids: Dict, fullamp_centroids: Dict,
                                  z_dim: int, frequencies: List[int]) -> pd.DataFrame:
    """
    Correlate latent positions between half and full amplitude
    For each dimension, correlate values across all conditions and patients
    """
    print("Computing position correlations...")
    results = []

    for dim in range(z_dim):
        halfamp_values = []
        fullamp_values = []

        # Collect values across all patients and conditions
        for patient in halfamp_centroids.keys():
            if patient not in fullamp_centroids:
                continue

            for condition in ['off'] + [str(f) for f in frequencies]:
                if condition in halfamp_centroids[patient] and condition in fullamp_centroids[patient]:
                    halfamp_values.append(halfamp_centroids[patient][condition][dim])
                    fullamp_values.append(fullamp_centroids[patient][condition][dim])

        if len(halfamp_values) >= 3:
            r, p = pearsonr(halfamp_values, fullamp_values)
            results.append({
                'dimension': dim,
                'correlation': r,
                'p_value': p,
                'n_points': len(halfamp_values)
            })

    return pd.DataFrame(results)


def compute_direction_vectors(centroids: Dict, frequencies: List[int]) -> Dict:
    """
    Compute direction vectors from OFF to each stimulation condition

    Returns:
        Dict[patient][condition] = direction vector from OFF (z_dim,)
    """
    print("Computing direction vectors from OFF state...")
    directions = {}

    for patient in centroids.keys():
        if 'off' not in centroids[patient]:
            continue

        directions[patient] = {}
        off_position = centroids[patient]['off']

        for freq in frequencies:
            freq_str = str(freq)
            if freq_str in centroids[patient]:
                stim_position = centroids[patient][freq_str]
                direction = stim_position - off_position
                directions[patient][freq_str] = direction

    return directions


def analyze_direction_similarity(halfamp_directions: Dict, fullamp_directions: Dict,
                                 frequencies: List[int]) -> pd.DataFrame:
    """
    Compute cosine similarity between direction vectors across amplitudes
    Also compute magnitude ratio
    """
    print("Analyzing direction vector similarity...")
    results = []

    for patient in halfamp_directions.keys():
        if patient not in fullamp_directions:
            continue

        for freq in frequencies:
            freq_str = str(freq)
            if freq_str not in halfamp_directions[patient] or freq_str not in fullamp_directions[patient]:
                continue

            half_dir = halfamp_directions[patient][freq_str]
            full_dir = fullamp_directions[patient][freq_str]

            # Cosine similarity (1 - cosine distance)
            cos_sim = 1 - cosine(half_dir, full_dir)

            # Magnitudes
            half_mag = np.linalg.norm(half_dir)
            full_mag = np.linalg.norm(full_dir)
            mag_ratio = full_mag / half_mag if half_mag > 0 else np.nan

            results.append({
                'patient': patient,
                'frequency': freq_str,
                'cosine_similarity': cos_sim,
                'halfamp_magnitude': half_mag,
                'fullamp_magnitude': full_mag,
                'magnitude_ratio': mag_ratio
            })

    return pd.DataFrame(results)


def compute_dimension_wise_correlations(halfamp_directions: Dict, fullamp_directions: Dict,
                                        z_dim: int, frequencies: List[int]) -> pd.DataFrame:
    """
    For each dimension, correlate the direction vector components across patients/frequencies
    """
    print("Computing dimension-wise direction correlations...")
    results = []

    for dim in range(z_dim):
        halfamp_comps = []
        fullamp_comps = []

        for patient in halfamp_directions.keys():
            if patient not in fullamp_directions:
                continue

            for freq in frequencies:
                freq_str = str(freq)
                if freq_str in halfamp_directions[patient] and freq_str in fullamp_directions[patient]:
                    halfamp_comps.append(halfamp_directions[patient][freq_str][dim])
                    fullamp_comps.append(fullamp_directions[patient][freq_str][dim])

        if len(halfamp_comps) >= 3:
            r, p = pearsonr(halfamp_comps, fullamp_comps)
            results.append({
                'dimension': dim,
                'correlation': r,
                'p_value': p,
                'n_vectors': len(halfamp_comps)
            })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_position_correlations(position_corr_df: pd.DataFrame, K: int, L: int, output_dir: str):
    """Plot correlations between half and full amplitude latent positions with 95% CI (horizontal layout)"""
    fig, ax = plt.subplots(figsize=(11, 7))

    dims = position_corr_df['dimension'].values
    corrs = position_corr_df['correlation'].values
    p_vals = position_corr_df['p_value'].values
    n_points = position_corr_df['n_points'].values

    # Compute 95% CI using Fisher's z transformation
    ci_lower = []
    ci_upper = []
    for r, n in zip(corrs, n_points):
        # Fisher's z transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se_z = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z

        # Back-transform to r
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        ci_lower.append(r_lower)
        ci_upper.append(r_upper)

    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Compute error bar sizes (horizontal now)
    xerr_lower = corrs - ci_lower
    xerr_upper = ci_upper - corrs

    # Okabe-Ito colorblind-safe palette
    blue_color = (0 / 255, 114 / 255, 178 / 255)      # Blue (Causal)
    orange_color = (230 / 255, 159 / 255, 0/255)      # Orange (Auxiliary)
    colors = [blue_color if d < K else orange_color for d in dims]

    # Different marker shapes for extra distinction
    markers = ['o' if d < K else 's' for d in dims]  # Circle for causal, square for auxiliary

    # Create y positions (reverse order so Dim 0 is on top)
    z_dim = K + L
    y_positions = np.arange(z_dim)[::-1]

    # Points with error bars (horizontal)
    for i, (dim, corr, color, marker, y_pos) in enumerate(zip(dims, corrs, colors, markers, y_positions)):
        marker_size = 180 if dim < K else 130  # Larger marker for causal
        ax.scatter(corr, y_pos, s=marker_size, color=color, alpha=0.9,
                  marker=marker, edgecolors='black', linewidth=2.5, zorder=3)

    # Horizontal error bars with thicker style
    ax.errorbar(corrs, y_positions, xerr=[xerr_lower, xerr_upper],
                fmt='none', ecolor='black', capsize=7, capthick=3, linewidth=2.5, alpha=0.7, zorder=2)

    # Add r [CI] (p-value) annotations to the right
    for i, (dim, corr, ci_l, ci_u, p, y_pos) in enumerate(zip(dims, corrs, ci_lower, ci_upper, p_vals, y_positions)):
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        annotation = f'r={corr:.3f} [{ci_l:.3f}, {ci_u:.3f}] ({p_str})'

        ax.text(1.06, y_pos, annotation, ha='left', va='center',
               fontsize=11, fontweight='medium', fontfamily='Roboto')

    # Add dimension labels to the left
    for i, (dim, y_pos) in enumerate(zip(dims, y_positions)):
        label = f"Dim {dim} {'(Causal)' if dim < K else '(Auxiliary)'}"
        ax.text(-0.17, y_pos, label, ha='right', va='center',
               fontsize=13 if dim < K else 12, fontweight='bold' if dim < K else 'medium',
               fontfamily='Roboto')

    # Reference lines
    ax.axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.4)
    ax.axvline(1, color=(0/255, 158/255, 115/255), linewidth=2.5, linestyle='--',
              alpha=0.7, label='Perfect correlation')

    ax.set_xlabel('Pearson Correlation (r)', fontsize=15, fontweight='medium', fontfamily='Roboto')
    ax.set_ylabel('')
    ax.set_title('Latent Position Preservation Across Amplitudes\n(with 95% Confidence Intervals)',
                fontsize=15, fontweight='bold', fontfamily='Roboto', pad=15)
    ax.set_xlim([-0.25, 1.25])
    ax.set_ylim([-0.5, z_dim - 0.5])
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(True, alpha=0.3, axis='x', linewidth=1.2)

    # Add column headers with better spacing
    ax.text(1.06, z_dim + 0.2, 'r [95% CI] (p-value)', ha='left', va='bottom',
           fontsize=13, fontweight='bold', fontfamily='Roboto')
    ax.text(-0.17, z_dim + 0.2, 'Dimension', ha='right', va='bottom',
           fontsize=13, fontweight='bold', fontfamily='Roboto')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=blue_color,
               markersize=12, markeredgecolor='black', markeredgewidth=2, label='Causal Dimension'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=orange_color,
               markersize=11, markeredgecolor='black', markeredgewidth=2, label='Auxiliary Dimensions')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12,
             framealpha=0.95, edgecolor='black', prop={'family': 'Roboto', 'weight': 'medium'})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_correlations.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'position_correlations.svg'), bbox_inches='tight')
    plt.close()


def plot_position_scatter(halfamp_centroids: Dict, fullamp_centroids: Dict,
                          frequencies: List[int], K: int, L: int, output_dir: str):
    """Scatter plots of half-amp vs full-amp positions for each dimension"""
    z_dim = K + L

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for dim in range(z_dim):
        ax = axes[dim]

        halfamp_vals, fullamp_vals = [], []
        colors_list, labels_list = [], []

        # Define colors for conditions
        condition_colors = {
            'off': 'gray',
            '55': '#1f77b4',
            '85': '#ff7f0e',
            '110': '#2ca02c',
            '125': '#d62728'
        }

        for patient in halfamp_centroids.keys():
            if patient not in fullamp_centroids:
                continue

            for condition in ['off'] + [str(f) for f in frequencies]:
                if condition in halfamp_centroids[patient] and condition in fullamp_centroids[patient]:
                    halfamp_vals.append(halfamp_centroids[patient][condition][dim])
                    fullamp_vals.append(fullamp_centroids[patient][condition][dim])
                    colors_list.append(condition_colors.get(condition, 'gray'))
                    labels_list.append(condition)

        # Scatter plot
        for condition in ['off'] + [str(f) for f in frequencies]:
            mask = np.array(labels_list) == condition
            if np.any(mask):
                ax.scatter(np.array(halfamp_vals)[mask], np.array(fullamp_vals)[mask],
                           c=condition_colors[condition], s=80, alpha=0.6,
                           edgecolors='black', linewidth=0.5, label=f'{condition}')

        # Unity line
        all_vals = halfamp_vals + fullamp_vals
        lim = [min(all_vals), max(all_vals)]
        ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1.5, label='Unity')

        # Regression line
        if len(halfamp_vals) >= 3:
            z = np.polyfit(halfamp_vals, fullamp_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(halfamp_vals), max(halfamp_vals), 100)
            ax.plot(x_line, p(x_line), 'r-', alpha=0.8, linewidth=2, label='Fit')

            # Correlation
            r, pval = pearsonr(halfamp_vals, fullamp_vals)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            ax.text(0.05, 0.95, f'r = {r:.3f}{sig}\np = {pval:.4f}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        dim_type = "Causal" if dim < K else "Auxiliary"
        ax.set_xlabel('Half-Amplitude Position', fontsize=11)
        ax.set_ylabel('Full-Amplitude Position', fontsize=11)
        ax.set_title(f'Dimension {dim} ({dim_type})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if dim == 0:
            ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_scatter_all_dims.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'position_scatter_all_dims.svg'), bbox_inches='tight')
    plt.close()


def analyze_direction_similarities(direction_sim_df: pd.DataFrame, frequencies: List[int]):
    """
    Quantitative analysis of direction vector similarities and magnitudes
    """
    print("\n" + "=" * 80)
    print("DIRECTION VECTOR ANALYSIS: Quantitative Summary")
    print("=" * 80)

    freqs = sorted(direction_sim_df['frequency'].unique(), key=lambda x: int(x), reverse=True)

    print("\n1. COSINE SIMILARITY (Direction Preservation)")
    print("-" * 80)
    print("Measures whether OFF→stimulation changes occur in same latent direction")
    print("Values close to 1.0 = direction preserved, 0.5 = 60° angle, 0.0 = perpendicular\n")

    overall_cos = direction_sim_df['cosine_similarity'].values
    print(f"Overall: Mean = {overall_cos.mean():.3f}, Median = {np.median(overall_cos):.3f}, "
          f"Range = [{overall_cos.min():.3f}, {overall_cos.max():.3f}]")
    high_sim_count = (overall_cos > 0.9).sum()
    print(f"  High similarity (>0.9): {high_sim_count}/{len(overall_cos)} "
          f"({100*high_sim_count/len(overall_cos):.1f}%)\n")

    print("By frequency:")
    for freq in freqs:
        freq_data = direction_sim_df[direction_sim_df['frequency'] == freq]['cosine_similarity'].values
        mean_cos = freq_data.mean()
        median_cos = np.median(freq_data)
        std_cos = freq_data.std()

        # Interpret cosine similarity as angle
        mean_angle = np.arccos(np.clip(mean_cos, -1, 1)) * 180 / np.pi

        print(f"  {freq} Hz: mean={mean_cos:.3f} (±{std_cos:.3f}), median={median_cos:.3f}, "
              f"~{mean_angle:.1f}° angle")

    print("\n2. MAGNITUDE SCALING (How much vectors grow)")
    print("-" * 80)
    print("Ratio = (Full-amp magnitude) / (Half-amp magnitude)")
    print("Ratio = 1.0 means same magnitude, 2.0 means full-amp is 2× larger\n")

    overall_mag = direction_sim_df['magnitude_ratio'].values
    print(f"Overall: Mean = {overall_mag.mean():.3f}, Median = {np.median(overall_mag):.3f}, "
          f"Range = [{overall_mag.min():.3f}, {overall_mag.max():.3f}]")

    if overall_mag.mean() > 1.0:
        print(f"  → Full-amplitude causes {overall_mag.mean():.2f}× larger latent space movement on average")
    else:
        print(f"  → Unexpected: Full-amplitude causes smaller movement (ratio < 1.0)")

    print("\nBy frequency:")
    for freq in freqs:
        freq_data = direction_sim_df[direction_sim_df['frequency'] == freq]['magnitude_ratio'].values
        mean_mag = freq_data.mean()
        median_mag = np.median(freq_data)
        std_mag = freq_data.std()

        print(f"  {freq} Hz: mean={mean_mag:.3f} (±{std_mag:.3f}), median={median_mag:.3f}")

    print("\n3. INTERPRETATION")
    print("-" * 80)
    if overall_cos.mean() > 0.85 and overall_mag.mean() > 1.0:
        print("✓ EXCELLENT: Directions preserved (high cosine similarity) AND magnitudes scale up")
        print("  → Model learned amplitude-invariant directions with appropriate scaling")
        print("  → Same qualitative neural changes, scaled by stimulation strength")
    elif overall_cos.mean() > 0.85:
        print("✓ GOOD: Directions preserved but unexpected magnitude scaling")
        print("  → Model captured direction of change consistently")
    elif overall_mag.mean() > 1.0:
        print("⚠ PARTIAL: Magnitudes scale but directions not fully preserved")
        print("  → Different amplitudes cause different latent trajectories")
    else:
        print("⚠ CONCERN: Neither direction nor magnitude consistently preserved")

    print("=" * 80 + "\n")


def plot_direction_similarities(direction_sim_df: pd.DataFrame, output_dir: str):
    """Plot cosine similarities between direction vectors with colorblind-safe Okabe-Ito palette"""

    # Okabe-Ito colorblind-safe palette
    okabe_ito = {
        '125': (230/255, 159/255, 0/255),      # Orange
        '110': (86/255, 180/255, 233/255),     # Sky blue
        '85': (0/255, 158/255, 115/255),       # Bluish green
        '55': (213/255, 94/255, 0/255),        # Vermillion
    }

    # Different markers for extra distinction
    markers = {'125': 'o', '110': 's', '85': '^', '55': 'D'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Cosine similarities
    ax = axes[0]
    freqs = sorted(direction_sim_df['frequency'].unique(), key=lambda x: int(x), reverse=True)

    for freq in freqs:
        freq_data = direction_sim_df[direction_sim_df['frequency'] == freq]
        patients = freq_data['patient'].values
        cos_sims = freq_data['cosine_similarity'].values

        ax.scatter([freq] * len(patients), cos_sims,
                  c=[okabe_ito[freq]], marker=markers[freq], s=120, alpha=0.8,
                  edgecolors='black', linewidth=1.5, label=f'{freq} Hz')

    # Box plot overlay
    data_by_freq = [direction_sim_df[direction_sim_df['frequency'] == freq]['cosine_similarity'].values
                    for freq in freqs]
    bp = ax.boxplot(data_by_freq, positions=range(len(freqs)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=(160/255, 160/255, 160/255), alpha=0.3, linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f'{f} Hz' for f in freqs], fontsize=12, fontfamily='Roboto', fontweight='medium')
    ax.set_ylabel('Cosine Similarity', fontsize=15, fontweight='medium', fontfamily='Roboto')
    ax.set_title('Direction Vector Similarity\n(Half-Amp vs Full-Amp)',
                fontsize=14, fontweight='bold', fontfamily='Roboto')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color=(0/255, 158/255, 115/255), linestyle='--', alpha=0.7,
              linewidth=2, label='High similarity (0.9)')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', fontsize=12, prop={'family': 'Roboto', 'weight': 'medium'})

    # Panel B: Magnitude ratios
    ax = axes[1]

    for freq in freqs:
        freq_data = direction_sim_df[direction_sim_df['frequency'] == freq]
        patients = freq_data['patient'].values
        mag_ratios = freq_data['magnitude_ratio'].values

        ax.scatter([freq] * len(patients), mag_ratios,
                  c=[okabe_ito[freq]], marker=markers[freq], s=120, alpha=0.8,
                  edgecolors='black', linewidth=1.5, label=f'{freq} Hz')

    # Box plot overlay
    data_by_freq = [direction_sim_df[direction_sim_df['frequency'] == freq]['magnitude_ratio'].values
                    for freq in freqs]
    bp = ax.boxplot(data_by_freq, positions=range(len(freqs)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=(160/255, 160/255, 160/255), alpha=0.3, linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f'{f} Hz' for f in freqs], fontsize=12, fontfamily='Roboto', fontweight='medium')
    ax.set_ylabel('Magnitude Ratio (Full / Half)', fontsize=15, fontweight='medium', fontfamily='Roboto')
    ax.set_title('Direction Vector Magnitude Scaling',
                fontsize=14, fontweight='bold', fontfamily='Roboto')
    ax.axhline(1.0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax.axhline(2.0, color=(0/255, 114/255, 178/255), linestyle='--', alpha=0.7,
              linewidth=2, label='2× scaling')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=12, prop={'family': 'Roboto', 'weight': 'medium'})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_analysis.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'direction_analysis.svg'), bbox_inches='tight')
    plt.close()


def plot_latent_trajectories_2d(halfamp_centroids: Dict, fullamp_centroids: Dict,
                                frequencies: List[int], output_dir: str):
    """
    Plot 2D projections of latent trajectories
    Show OFF → stimulation trajectories for both amplitudes
    """
    # Use first two auxiliary dimensions (dims 1 and 2)
    dim_x, dim_y = 1, 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    centroids_dict = {
        'Half-Amplitude': halfamp_centroids,
        'Full-Amplitude': fullamp_centroids
    }

    colors = {
        '55': '#1f77b4',
        '85': '#ff7f0e',
        '110': '#2ca02c',
        '125': '#d62728'
    }

    for idx, (amp_label, centroids) in enumerate(centroids_dict.items()):
        ax = axes[idx]

        for patient in centroids.keys():
            if 'off' not in centroids[patient]:
                continue

            off_pos = centroids[patient]['off'][[dim_x, dim_y]]

            # Plot OFF position
            ax.scatter(off_pos[0], off_pos[1], c='black', s=150, marker='o',
                       edgecolors='white', linewidths=2, zorder=10, alpha=0.8)

            # Plot arrows to each frequency
            for freq in frequencies:
                freq_str = str(freq)
                if freq_str in centroids[patient]:
                    stim_pos = centroids[patient][freq_str][[dim_x, dim_y]]

                    ax.arrow(off_pos[0], off_pos[1],
                             stim_pos[0] - off_pos[0], stim_pos[1] - off_pos[1],
                             head_width=0.15, head_length=0.1, fc=colors[freq_str],
                             ec='black', linewidth=1.5, alpha=0.6, length_includes_head=True)

                    # Label endpoint
                    ax.scatter(stim_pos[0], stim_pos[1], c=colors[freq_str], s=100,
                               marker='s', edgecolors='black', linewidths=1, alpha=0.7)
                    ax.text(stim_pos[0], stim_pos[1], f'  {freq}Hz', fontsize=8, alpha=0.8)

        ax.set_xlabel(f'Dimension {dim_x} (Auxiliary)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Dimension {dim_y} (Auxiliary)', fontsize=11, fontweight='bold')
        ax.set_title(f'{amp_label}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Add legend
        from matplotlib.patches import Circle
        legend_elements = [Circle((0, 0), 1, fc='black', ec='white', label='OFF')]
        for freq in frequencies:
            legend_elements.append(Circle((0, 0), 1, fc=colors[str(freq)], ec='black',
                                          label=f'{freq} Hz'))
        ax.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_trajectories_2d.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'latent_trajectories_2d.svg'), bbox_inches='tight')
    plt.close()


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(position_corr_df: pd.DataFrame,
                             direction_sim_df: pd.DataFrame,
                             dimwise_corr_df: pd.DataFrame,
                             K: int, L: int):
    """Print summary statistics for the manuscript"""

    print("\n" + "=" * 80)
    print("LATENT SPACE GEOMETRY ANALYSIS SUMMARY")
    print("=" * 80)

    # Position correlations
    print("\n1. LATENT POSITION CORRELATIONS (Half-Amp vs Full-Amp)")
    print("-" * 80)
    mean_r = position_corr_df['correlation'].mean()
    min_r = position_corr_df['correlation'].min()
    max_r = position_corr_df['correlation'].max()
    sig_count = (position_corr_df['p_value'] < 0.05).sum()

    print(f"Mean Pearson r = {mean_r:.3f}")
    print(f"Range: {min_r:.3f} to {max_r:.3f}")
    print(f"Significant dimensions: {sig_count}/{len(position_corr_df)} (p<0.05)")

    print("\nBy dimension type:")
    causal_r = position_corr_df[position_corr_df['dimension'] < K]['correlation'].mean()
    aux_r = position_corr_df[position_corr_df['dimension'] >= K]['correlation'].mean()
    print(f"  Causal dimension (dim 0): r = {causal_r:.3f}")
    print(f"  Auxiliary dimensions (dims 1-{L}): mean r = {aux_r:.3f}")

    # Direction similarities
    print("\n2. DIRECTION VECTOR ANALYSIS")
    print("-" * 80)
    mean_cos = direction_sim_df['cosine_similarity'].mean()
    min_cos = direction_sim_df['cosine_similarity'].min()
    max_cos = direction_sim_df['cosine_similarity'].max()

    print(f"Mean cosine similarity = {mean_cos:.3f}")
    print(f"Range: {min_cos:.3f} to {max_cos:.3f}")
    print(f"High similarity (>0.9): {(direction_sim_df['cosine_similarity'] > 0.9).sum()}/{len(direction_sim_df)}")

    print("\nMagnitude scaling:")
    mean_mag_ratio = direction_sim_df['magnitude_ratio'].mean()
    median_mag_ratio = direction_sim_df['magnitude_ratio'].median()
    print(f"  Mean magnitude ratio (Full/Half) = {mean_mag_ratio:.3f}")
    print(f"  Median magnitude ratio = {median_mag_ratio:.3f}")

    print("\nBy frequency:")
    for freq in sorted(direction_sim_df['frequency'].unique(), key=lambda x: int(x)):
        freq_data = direction_sim_df[direction_sim_df['frequency'] == freq]
        mean_cos_freq = freq_data['cosine_similarity'].mean()
        mean_mag_freq = freq_data['magnitude_ratio'].mean()
        print(f"  {freq} Hz: cosine similarity = {mean_cos_freq:.3f}, mag ratio = {mean_mag_freq:.3f}")

    # Dimension-wise direction correlations
    print("\n3. DIRECTION COMPONENT CORRELATIONS (By Dimension)")
    print("-" * 80)
    for _, row in dimwise_corr_df.iterrows():
        dim = int(row['dimension'])
        dim_type = "Causal" if dim < K else "Auxiliary"
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row[
                                                                                                 'p_value'] < 0.05 else ''
        print(f"  Dimension {dim} ({dim_type}): r = {row['correlation']:.3f} (p = {row['p_value']:.4f}) {sig}")

    print("\n" + "=" * 80)

    # Numbers for manuscript
    print("\nFOR MANUSCRIPT:")
    print("-" * 80)
    print(f"\"Latent positions at full amplitude strongly correlated with half-amplitude")
    print(f"positions across all six dimensions (mean Pearson r = {mean_r:.3f}, range {min_r:.3f}")
    print(f"to {max_r:.3f}), indicating the model learned amplitude-invariant representations")
    print(f"of neural state. The direction vectors from OFF to each stimulation condition")
    print(f"maintained consistent orientation across amplitudes (mean cosine similarity = {mean_cos:.3f},")
    print(f"range {min_cos:.3f} to {max_cos:.3f}), though magnitudes scaled with amplitude as expected")
    print(f"(mean ratio = {mean_mag_ratio:.3f}).\"")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_geometry_analysis(config: Dict):
    """Run complete latent space geometry analysis"""

    os.makedirs(config['output_dir'], exist_ok=True)

    # Extract model parameters
    model_params = extract_parameters(config['model_path'])
    K, L = model_params['K'], model_params['L']
    z_dim = K + L

    print(f"\n{'=' * 60}")
    print(f"LATENT SPACE GEOMETRY ANALYSIS")
    print(f"Model: K={K}, L={L}, z_dim={z_dim}, λ={model_params['lambda']}")
    print(f"Frequencies: {config['frequencies']}")
    print(f"{'=' * 60}\n")

    # Base config for encoder
    base_config = {
        "vae_model": "vanilla_Encoder", "n_features": 70,
        "alpha_samples": 1, "beta_samples": 1,
        "include_classes": None, "save_dir": "./", "model_name": "vae_model"
    }

    # Load encoder
    encoder = load_encoder_for_params(base_config,
                                      ModelParams(0, K, L, model_params['lambda'], config['model_path']))

    # Load data for both amplitudes
    X_half, cond_half, subj_half = load_amplitude_data(
        config['halfamp_data_path'], config['frequencies'], "Half-Amplitude")
    X_full, cond_full, subj_full = load_amplitude_data(
        config['fullamp_data_path'], config['frequencies'], "Full-Amplitude")

    # Get latent representations
    print("\nComputing latent representations...")
    latent_half = get_latent_representations(encoder, X_half, subj_half)
    latent_full = get_latent_representations(encoder, X_full, subj_full)

    # Compute condition centroids
    centroids_half = compute_condition_centroids(latent_half, cond_half, subj_half, config['frequencies'])
    centroids_full = compute_condition_centroids(latent_full, cond_full, subj_full, config['frequencies'])

    # Analysis 1: Position correlations
    position_corr_df = compute_position_correlations(centroids_half, centroids_full, z_dim, config['frequencies'])

    # Analysis 2: Direction vectors
    directions_half = compute_direction_vectors(centroids_half, config['frequencies'])
    directions_full = compute_direction_vectors(centroids_full, config['frequencies'])
    direction_sim_df = analyze_direction_similarity(directions_half, directions_full, config['frequencies'])

    # Analysis 3: Dimension-wise direction correlations
    dimwise_corr_df = compute_dimension_wise_correlations(directions_half, directions_full, z_dim,
                                                          config['frequencies'])

    # Save results
    position_corr_df.to_csv(os.path.join(config['output_dir'], 'position_correlations.csv'), index=False)
    direction_sim_df.to_csv(os.path.join(config['output_dir'], 'direction_analysis.csv'), index=False)
    dimwise_corr_df.to_csv(os.path.join(config['output_dir'], 'dimwise_direction_correlations.csv'), index=False)

    # Quantitative analysis of direction similarities
    analyze_direction_similarities(direction_sim_df, config['frequencies'])

    # Generate plots
    print("\nGenerating visualizations...")
    plot_position_correlations(position_corr_df, K, L, config['output_dir'])
    plot_position_scatter(centroids_half, centroids_full, config['frequencies'], K, L, config['output_dir'])
    plot_direction_similarities(direction_sim_df, config['output_dir'])
    plot_latent_trajectories_2d(centroids_half, centroids_full, config['frequencies'], config['output_dir'])

    # Print summary
    print_summary_statistics(position_corr_df, direction_sim_df, dimwise_corr_df, K, L)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to: {config['output_dir']}/")
    print(f"{'=' * 60}")

    return position_corr_df, direction_sim_df, dimwise_corr_df, centroids_half, centroids_full


if __name__ == "__main__":
    position_corr, direction_sim, dimwise_corr, cents_half, cents_full = run_geometry_analysis(CONFIG)