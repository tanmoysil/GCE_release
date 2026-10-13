from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress
from scipy.io import loadmat
import torch
from sklearn.preprocessing import StandardScaler
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from gce.utils_results import *
from dcor import distance_correlation
from dcor.independence import distance_correlation_t_test
from config.paths import (
    FULLAMP_ALL_CONDITIONS_MAT, SCORES_FULLAMP_XLSX, BEST_ENCODER_PT
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data paths - FULL AMPLITUDE
    'data_path': FULLAMP_ALL_CONDITIONS_MAT,
    'scores_path': SCORES_FULLAMP_XLSX,
    'model_path': BEST_ENCODER_PT,
    'output_dir': './fullamp_correlation_analysis',

    # Frequencies to analyze
    'frequencies': [125, 110, 85, 55],

    # Dimension combinations to analyze (0-indexed)
    'dim_combinations': [
        (0, 1, 2, 3, 4, 5),  # causal + all auxiliary
        (1, 2),  # first two auxiliary
        (1, 2, 3, 4, 5),  # all auxiliary
    ],

    # Correlation methods
    'correlation_methods': ['pearson', 'spearman', 'kendall', 'dcor'],

    # Plotting
    'generate_plots': True,
    'plot_dpi': 600,
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

def load_data(data_path: str, frequencies: List[int]):
    """Load neural data for specified conditions"""
    print(f"Loading FULL-AMPLITUDE data from {data_path}")
    mat_data = loadmat(data_path)
    X_list, idx_list, condition_labels = [], [], []

    for freq in frequencies:
        freq_key = f'X_{freq}_cleaned'
        idx_key = f'idx_{freq}_cleaned'
        if freq_key in mat_data and idx_key in mat_data:
            X_list.append(mat_data[freq_key])
            idx_list.append(mat_data[idx_key].flatten())
            condition_labels.append(np.full(len(mat_data[freq_key]), str(freq)))
            print(f"  {freq} Hz (full-amp): {len(mat_data[freq_key])} samples")

    X_list.append(mat_data['X_off_cleaned'])
    idx_list.append(mat_data['idx_off_cleaned'].flatten())
    condition_labels.append(np.full(len(mat_data['X_off_cleaned']), 'off'))
    print(f"  OFF: {len(mat_data['X_off_cleaned'])} samples")

    return np.vstack(X_list), np.hstack(condition_labels), np.hstack(idx_list)


def load_normalized_scores(excel_path: str, frequencies: List[int]) -> Dict:
    """Load pre-calculated normalized clinical scores"""
    print(f"Loading FULL-AMPLITUDE normalized scores from {excel_path}")
    df = pd.read_excel(excel_path)
    scores = {}

    for _, row in df.iterrows():
        patient_num = row['PatNum']
        if patient_num not in scores:
            scores[patient_num] = {}
        for freq in frequencies:
            scores[patient_num][str(freq)] = {
                'total': row.get(f'Normalized_{freq}_Total', np.nan),
                'rigidity': row.get(f'Normalized_{freq}_Rigidity', np.nan),
                'brady': row.get(f'Normalized_{freq}_Brady', np.nan),
                'tremor': row.get(f'Normalized_{freq}_Tremor', np.nan)
            }

    print(f"Loaded normalized scores for {len(scores)} patients")
    return scores


# =============================================================================
# DISTANCE COMPUTATION
# =============================================================================

def compute_all_distances(latent_dims: np.ndarray, conditions: np.ndarray,
                          subject_indices: np.ndarray, frequencies: List[int],
                          K: int, L: int, dim_combinations: List[Tuple[int, ...]]) -> Dict:
    """Compute all distance metrics in latent space from OFF condition"""
    print("Computing distances in latent space...")
    z_dim = K + L
    distance_dict = {}

    for patient in np.unique(subject_indices):
        patient_mask = subject_indices == patient
        all_conditions = ['off'] + [str(f) for f in frequencies]
        condition_latent_means = {}

        for condition in all_conditions:
            combined_mask = patient_mask & (conditions == condition)
            if np.any(combined_mask):
                condition_latent_means[condition] = np.mean(latent_dims[combined_mask], axis=0)

        if 'off' not in condition_latent_means:
            continue

        off_centroid = condition_latent_means['off']
        distance_dict[patient] = {}

        for condition in condition_latent_means:
            cond_centroid = condition_latent_means[condition]
            dist = {}

            # Individual dimension distances
            for dim in range(z_dim):
                signed_dist = cond_centroid[dim] - off_centroid[dim]
                dist[f'dim_{dim}_signed'] = signed_dist
                dist[f'dim_{dim}_abs'] = np.abs(signed_dist)

            # Standard Euclidean norms
            dist['norm_all'] = np.linalg.norm(cond_centroid - off_centroid)
            dist['norm_aux'] = np.linalg.norm(cond_centroid[K:] - off_centroid[K:])
            if K > 0:
                dist['norm_causal'] = np.linalg.norm(cond_centroid[:K] - off_centroid[:K])

            # Custom dimension combinations
            for dims in dim_combinations:
                if any(d >= z_dim for d in dims):
                    continue
                dim_label = '_'.join(map(str, dims))
                signed_diffs = [cond_centroid[d] - off_centroid[d] for d in dims]
                dist[f'dims_{dim_label}_sum_signed'] = sum(signed_diffs)
                dist[f'dims_{dim_label}_sum_abs'] = sum(np.abs(d) for d in signed_diffs)
                dist[f'dims_{dim_label}_norm'] = np.linalg.norm(signed_diffs)

            distance_dict[patient][condition] = dist

    return distance_dict


def combine_distances_with_scores(distance_dict: Dict, normalized_scores: Dict,
                                  frequencies: List[int]) -> pd.DataFrame:
    """Combine latent distances with normalized clinical scores"""
    combined_data = []

    for patient in distance_dict:
        if patient not in normalized_scores:
            continue
        for freq in frequencies:
            freq_str = str(freq)
            if freq_str not in distance_dict[patient] or freq_str not in normalized_scores[patient]:
                continue

            row = {'patient': patient, 'condition': freq_str}
            for score_name, score_val in normalized_scores[patient][freq_str].items():
                row[f'norm_{score_name}'] = score_val
            row.update(distance_dict[patient][freq_str])
            combined_data.append(row)

    df = pd.DataFrame(combined_data)
    print(f"Created combined DataFrame with {len(df)} rows")
    return df


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlations(df: pd.DataFrame, distance_cols: List[str],
                           method: str = 'spearman') -> pd.DataFrame:
    """Calculate correlations between distance metrics and clinical scores"""
    clinical_measures = {
        'norm_total': 'Total Score',
        'norm_rigidity': 'Rigidity',
        'norm_brady': 'Bradykinesia',
        'norm_tremor': 'Tremor'
    }
    results = []

    for dist_col in distance_cols:
        for measure_col, measure_name in clinical_measures.items():
            if measure_col not in df.columns or dist_col not in df.columns:
                continue

            valid = df.dropna(subset=[measure_col, dist_col])
            if len(valid) < 3:
                continue

            x = np.asarray(valid[dist_col], dtype=np.float64)
            y = np.asarray(valid[measure_col], dtype=np.float64)

            if method == 'spearman':
                r, p = spearmanr(x, y)
            elif method == 'pearson':
                r, p = pearsonr(x, y)
            elif method == 'kendall':
                r, p = kendalltau(x, y)
            elif method == 'dcor':
                r = distance_correlation(x, y)
                p = distance_correlation_t_test(x, y).pvalue
            else:
                continue

            results.append({
                'distance_metric': dist_col,
                'clinical_measure': measure_name,
                'correlation': r,
                'p_value': p,
                'n': len(valid),
                'significant': p < 0.05
            })

    return pd.DataFrame(results)


def compute_both_correlations_with_ci(x, y):
    """Compute BOTH Pearson and Spearman correlations with Fisher's z 95% CI"""
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 3:
        return None

    n = len(x_valid)
    se_z = 1 / np.sqrt(n - 3)

    # Spearman correlation
    rho, p_spearman = spearmanr(x_valid, y_valid)
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    z_lower_rho = z_rho - 1.96 * se_z
    z_upper_rho = z_rho + 1.96 * se_z
    rho_lower = (np.exp(2 * z_lower_rho) - 1) / (np.exp(2 * z_lower_rho) + 1)
    rho_upper = (np.exp(2 * z_upper_rho) - 1) / (np.exp(2 * z_upper_rho) + 1)

    # Pearson correlation
    r, p_pearson = pearsonr(x_valid, y_valid)
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    z_lower_r = z_r - 1.96 * se_z
    z_upper_r = z_r + 1.96 * se_z
    r_lower = (np.exp(2 * z_lower_r) - 1) / (np.exp(2 * z_lower_r) + 1)
    r_upper = (np.exp(2 * z_upper_r) - 1) / (np.exp(2 * z_upper_r) + 1)

    return {
        'spearman': {
            'rho': rho,
            'rho_lower': rho_lower,
            'rho_upper': rho_upper,
            'p': p_spearman
        },
        'pearson': {
            'r': r,
            'r_lower': r_lower,
            'r_upper': r_upper,
            'p': p_pearson
        },
        'n': n
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation_matrix(df: pd.DataFrame, correlations_df: pd.DataFrame,
                            distance_col: str, frequencies: List[int],
                            output_dir: str, dpi: int = 150):
    """
    Create correlation matrix plot with patient labels and frequency colors
    Similar to the original plot_correlation_matrix style
    """
    measures = {
        'norm_total': 'Total Score',
        'norm_rigidity': 'Rigidity',
        'norm_brady': 'Bradykinesia',
        'norm_tremor': 'Tremor'
    }

    # Filter valid measures
    valid_measures = {col: name for col, name in measures.items()
                      if col in df.columns and df[col].notna().sum() >= 3}

    n_plots = len(valid_measures)
    if n_plots == 0:
        return None

    n_rows, n_cols = (1, n_plots) if n_plots <= 2 else (2, 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Colors and markers for frequencies
    cmap = plt.cm.tab10
    colors = {str(f): cmap(i) for i, f in enumerate(frequencies)}
    markers = {str(f): ['o', 's', '^', 'D'][i % 4] for i, f in enumerate(frequencies)}

    for i, (measure_col, measure_name) in enumerate(valid_measures.items()):
        ax = axes[i]

        # Get correlation for this measure
        corr_row = correlations_df[
            (correlations_df['clinical_measure'] == measure_name) &
            (correlations_df['distance_metric'] == distance_col)
            ]
        r = corr_row['correlation'].iloc[0] if len(corr_row) > 0 else None
        p = corr_row['p_value'].iloc[0] if len(corr_row) > 0 else None

        # Plot by condition
        for condition, group in df.groupby('condition'):
            valid_group = group.dropna(subset=[measure_col, distance_col])
            if len(valid_group) > 0:
                ax.scatter(valid_group[distance_col], valid_group[measure_col],
                           color=colors.get(condition, 'gray'),
                           marker=markers.get(condition, 'o'),
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                           label=f'{condition} Hz')

                # Patient labels
                for _, row in valid_group.iterrows():
                    ax.annotate(f"P{int(row['patient'])}",
                                xy=(row[distance_col], row[measure_col]),
                                xytext=(5, 0), textcoords='offset points',
                                fontsize=8, alpha=0.8)

        # Correlation annotation
        if r is not None and p is not None:
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.text(0.05, 0.95, f"r = {r:.3f}{sig}\np = {p:.4f}",
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Background color for significant correlations
            if p < 0.05:
                ax.set_facecolor('#e6ffe6' if r < 0 else '#ffe6e6')

        # Trendline
        trend_data = df.dropna(subset=[measure_col, distance_col])
        if len(trend_data) >= 3:
            slope, intercept, _, _, _ = linregress(
                trend_data[distance_col], trend_data[measure_col])
            x_vals = np.array([trend_data[distance_col].min(),
                               trend_data[distance_col].max()])
            ax.plot(x_vals, intercept + slope * x_vals, 'k--', alpha=0.6, linewidth=1.5)

        ax.set_title(measure_name, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{distance_col}', fontsize=11)
        ax.set_ylabel(f'Normalized {measure_name}', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            ax.legend(loc='lower right', fontsize=9)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Clean title
    title_metric = distance_col.replace('_', ' ').replace('dims', 'Dims').replace('norm', 'Norm')
    plt.suptitle(f'Full-Amp: Latent Distance vs Clinical Scores\nMetric: {title_metric}',
                 y=1.02, fontsize=14, fontweight='bold')

    safe_name = distance_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'fullamp_correlation_matrix_{safe_name}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'fullamp_correlation_matrix_{safe_name}.pdf'),
                bbox_inches='tight')
    plt.close()

    return fig


def plot_correlation_by_condition(df: pd.DataFrame, distance_col: str,
                                  output_dir: str, dpi: int = 150):
    """Plot correlations separately for each frequency condition"""
    conditions = sorted(df['condition'].unique(), key=lambda x: int(x), reverse=True)
    n_conditions = len(conditions)

    fig, axes = plt.subplots(n_conditions, 4, figsize=(16, 4 * n_conditions))
    if n_conditions == 1:
        axes = axes.reshape(1, -1)

    clinical_measures = ['norm_total', 'norm_rigidity', 'norm_brady', 'norm_tremor']
    measure_names = ['Total', 'Rigidity', 'Bradykinesia', 'Tremor']

    for i, condition in enumerate(conditions):
        cond_data = df[df['condition'] == condition]

        for j, (measure, name) in enumerate(zip(clinical_measures, measure_names)):
            ax = axes[i, j]
            valid = cond_data.dropna(subset=[measure, distance_col])

            if len(valid) >= 3:
                x, y = valid[distance_col], valid[measure]

                # Scatter with patient colors
                patients = valid['patient'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(patients)))
                patient_colors = dict(zip(patients, colors))

                for patient in patients:
                    mask = valid['patient'] == patient
                    ax.scatter(x[mask], y[mask], c=[patient_colors[patient]],
                               s=80, alpha=0.7, edgecolors='black', label=f'P{int(patient)}')

                # Regression line
                z = np.polyfit(x, y, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p_line(x_line), 'r--', linewidth=2)

                r, pval = spearmanr(x, y)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                ax.set_title(f'{condition}Hz - {name}\nρ={r:.3f}{sig}')

                if pval < 0.05:
                    ax.set_facecolor('#e6ffe6' if r < 0 else '#ffe6e6')
            else:
                ax.set_title(f'{condition}Hz - {name}\n(n<3)')

            ax.set_xlabel(distance_col)
            ax.set_ylabel('Normalized Score')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = distance_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'fullamp_by_condition_{safe_name}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    return fig


def plot_correlation_heatmap(results_df: pd.DataFrame, output_dir: str,
                             method: str, dpi: int = 150):
    """Create heatmap of all correlations"""
    if results_df.empty:
        return

    pivot = results_df.pivot(index='distance_metric', columns='clinical_measure', values='correlation')

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add values with significance
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                mask = (results_df['distance_metric'] == pivot.index[i]) & \
                       (results_df['clinical_measure'] == pivot.columns[j])
                p_val = results_df.loc[mask, 'p_value'].values[0] if mask.any() else 1
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(f'Full-Amp {method.capitalize()} Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fullamp_heatmap_{method}.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_scatter_grid(df: pd.DataFrame, correlations_df: pd.DataFrame,
                      distance_col: str, frequencies: List[int],
                      output_dir: str, dpi: int = 150):
    """
    Create scatter plot with patient colors (alternative view)
    """
    measures = {
        'norm_total': 'Total Score',
        'norm_rigidity': 'Rigidity',
        'norm_brady': 'Bradykinesia',
        'norm_tremor': 'Tremor'
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Patient colors
    patients = df['patient'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(patients)))
    patient_color_map = dict(zip(patients, colors))

    for idx, (measure_col, measure_name) in enumerate(measures.items()):
        ax = axes[idx]

        if measure_col not in df.columns:
            ax.set_visible(False)
            continue

        valid_data = df.dropna(subset=[measure_col, distance_col])

        if len(valid_data) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(measure_name)
            continue

        x = valid_data[distance_col]
        y = valid_data[measure_col]

        for patient in patients:
            mask = valid_data['patient'] == patient
            ax.scatter(x[mask], y[mask],
                       c=[patient_color_map[patient]],
                       label=f'P{int(patient)}',
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)

        # Correlations
        r_pearson, p_pearson = pearsonr(x, y)
        r_spearman, p_spearman = spearmanr(x, y)

        textstr = f'Pearson: r={r_pearson:.3f}, p={p_pearson:.4f}\nSpearman: ρ={r_spearman:.3f}, p={p_spearman:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if p_spearman < 0.05:
            ax.set_facecolor('#e6ffe6' if r_spearman < 0 else '#ffe6e6')

        ax.set_xlabel(f'{distance_col}')
        ax.set_ylabel(f'Normalized {measure_name}')
        ax.set_title(measure_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5))

    plt.tight_layout()
    safe_name = distance_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'fullamp_scatter_patients_{safe_name}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# NEW: DIMENSION-CLINICAL SCORE CORRELATION PLOTS (DUAL SPEARMAN + PEARSON)
# =============================================================================

def plot_dimensions_vs_total_forest(combined_df: pd.DataFrame, K: int, L: int,
                                     output_dir: str, dpi: int = 600):
    """
    Single forest plot: All dimensions vs Total Score (dual Spearman + Pearson annotations)
    """
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.weight'] = 'medium'

    # Okabe-Ito colors
    blue_color = (0/255, 114/255, 178/255)      # Blue (Causal)
    orange_color = (230/255, 159/255, 0/255)    # Orange (Auxiliary)
    green_color = (0/255, 158/255, 115/255)     # Bluish green (Combined)

    # Compute combined auxiliary (sum of dim_1 + dim_2)
    if 'dim_1_signed' in combined_df.columns and 'dim_2_signed' in combined_df.columns:
        combined_df['combined_aux_12_signed'] = (
            combined_df['dim_1_signed'] + combined_df['dim_2_signed']
        )

    z_dim = K + L
    results = []

    # Compute BOTH correlations for each dimension
    for dim in range(z_dim):
        dim_col = f'dim_{dim}_signed'
        if dim_col not in combined_df.columns:
            continue

        corr_data = compute_both_correlations_with_ci(
            combined_df[dim_col].values,
            combined_df['norm_total'].values
        )

        if corr_data is not None:
            results.append({
                'dim': dim,
                'label': f'Dim {dim}',
                'spearman': corr_data['spearman'],
                'pearson': corr_data['pearson'],
                'n': corr_data['n'],
                'is_causal': dim < K,
                'is_combined': False
            })

    # Add combined Dim 1+2 to results
    if 'combined_aux_12_signed' in combined_df.columns:
        corr_data = compute_both_correlations_with_ci(
            combined_df['combined_aux_12_signed'].values,
            combined_df['norm_total'].values
        )
        if corr_data is not None:
            results.append({
                'dim': 'combined_12',
                'label': 'Dim 1+2 (Combined)',
                'spearman': corr_data['spearman'],
                'pearson': corr_data['pearson'],
                'n': corr_data['n'],
                'is_causal': False,
                'is_combined': True
            })

    if not results:
        print("No valid correlations found for forest plot")
        return

    # Create single forest plot with dual annotations
    fig, ax = plt.subplots(figsize=(13, 7))

    n_dims = len(results)
    y_positions = np.arange(n_dims)[::-1]  # Reverse so Dim 0 is at top

    for i, res in enumerate(results):
        y_pos = y_positions[i]

        # Determine color, marker, and size based on dimension type
        if res['is_combined']:
            color = green_color
            marker = '^'  # Triangle
            marker_size = 200
        elif res['is_causal']:
            color = blue_color
            marker = 'o'
            marker_size = 180
        else:
            color = orange_color
            marker = 's'
            marker_size = 130

        # Use Spearman for the point + error bar (primary method)
        sp = res['spearman']
        rho = sp['rho']
        rho_lower = sp['rho_lower']
        rho_upper = sp['rho_upper']

        # Point
        ax.scatter(rho, y_pos, s=marker_size, color=color, alpha=0.9,
                  marker=marker, edgecolors='black', linewidth=2.5, zorder=3)

        # Error bar
        xerr_lower = rho - rho_lower
        xerr_upper = rho_upper - rho
        ax.errorbar(rho, y_pos, xerr=[[xerr_lower], [xerr_upper]],
                   fmt='none', ecolor='black', capsize=7, capthick=3,
                   linewidth=2.5, alpha=0.7, zorder=2)

        # Dual annotation (2 lines: Spearman + Pearson)
        pe = res['pearson']
        p_str_sp = "p<0.001" if sp['p'] < 0.001 else f"p={sp['p']:.3f}"
        p_str_pe = "p<0.001" if pe['p'] < 0.001 else f"p={pe['p']:.3f}"

        annotation = (f"ρ={rho:.3f} [{rho_lower:.3f}, {rho_upper:.3f}] ({p_str_sp})\n"
                     f"r={pe['r']:.3f} [{pe['r_lower']:.3f}, {pe['r_upper']:.3f}] ({p_str_pe})")
        ax.text(1.05, y_pos, annotation, ha='left', va='center',
               fontsize=10, fontweight='medium', fontfamily='Roboto')

    # Reference line at 0
    ax.axvline(0, color=(160/255, 160/255, 160/255), linewidth=2,
              linestyle='--', alpha=0.6, zorder=1)

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([res['label'] for res in results],
                       fontsize=12, fontweight='medium', fontfamily='Roboto')
    ax.set_xlabel('Correlation Coefficient', fontsize=15,
                 fontweight='medium', fontfamily='Roboto')
    ax.set_title('Latent Dimensions vs Total Clinical Score\n(Spearman ρ and Pearson r with 95% Confidence Intervals)',
                fontsize=14, fontweight='bold', fontfamily='Roboto', pad=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-1, 1.25)  # Extra space for annotations

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=blue_color,
               markersize=12, markeredgecolor='black', markeredgewidth=2, label='Causal Dimension'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=orange_color,
               markersize=11, markeredgecolor='black', markeredgewidth=2, label='Auxiliary Dimensions'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=green_color,
               markersize=12, markeredgecolor='black', markeredgewidth=2, label='Combined (1+2)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12,
             framealpha=0.95, edgecolor='black',
             prop={'family': 'Roboto', 'weight': 'medium'})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_correlations_total_forest.png'),
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'dimension_correlations_total_forest.svg'),
                bbox_inches='tight')
    plt.close()

    print(f"✓ Saved forest plot: dimension_correlations_total_forest")


def plot_dimensions_vs_total_detailed(combined_df: pd.DataFrame, K: int, L: int,
                                       frequencies: List[int], output_dir: str,
                                       dpi: int = 600):
    """
    Detailed scatter subplots: All dimensions + Combined (1+2) vs Total Score (2×4 grid for 7 plots)
    Shows both Spearman and Pearson in annotations
    """
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.weight'] = 'medium'

    # Okabe-Ito colors for frequencies
    freq_colors = {
        '125': (230/255, 159/255, 0/255),      # Orange
        '110': (86/255, 180/255, 233/255),     # Sky blue
        '85': (0/255, 158/255, 115/255),       # Bluish green
        '55': (213/255, 94/255, 0/255),        # Vermillion
    }
    freq_markers = {
        '125': 'o', '110': 's', '85': '^', '55': 'D'
    }

    # Compute combined auxiliary (sum of dim_1 + dim_2)
    if 'dim_1_signed' in combined_df.columns and 'dim_2_signed' in combined_df.columns:
        combined_df['combined_aux_12_signed'] = (
            combined_df['dim_1_signed'] + combined_df['dim_2_signed']
        )

    z_dim = K + L
    # Create 2×4 grid to accommodate 6 individual dims + 1 combined
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for dim in range(z_dim):
        ax = axes[dim]
        dim_col = f'dim_{dim}_signed'

        if dim_col not in combined_df.columns:
            ax.axis('off')
            continue

        # Plot points by frequency
        for freq in frequencies:
            freq_str = str(freq)
            freq_data = combined_df[combined_df['condition'] == freq_str]
            valid = freq_data.dropna(subset=[dim_col, 'norm_total'])

            if len(valid) > 0:
                ax.scatter(valid[dim_col], valid['norm_total'],
                          c=[freq_colors[freq_str]], marker=freq_markers[freq_str],
                          s=120, alpha=0.8, edgecolors='black', linewidth=1.5,
                          label=f'{freq} Hz')

                # Optional: Add patient labels
                for _, row in valid.iterrows():
                    ax.annotate(f"P{int(row['patient'])}",
                               xy=(row[dim_col], row['norm_total']),
                               xytext=(3, 3), textcoords='offset points',
                               fontsize=7, alpha=0.6, fontfamily='Roboto')

        # Compute BOTH correlations
        corr_data = compute_both_correlations_with_ci(
            combined_df[dim_col].values,
            combined_df['norm_total'].values
        )

        if corr_data is not None:
            valid_data = combined_df.dropna(subset=[dim_col, 'norm_total'])

            if len(valid_data) >= 3:
                # Regression line (using actual linear regression for both)
                slope, intercept, _, _, _ = linregress(
                    valid_data[dim_col], valid_data['norm_total']
                )
                x_vals = np.array([valid_data[dim_col].min(),
                                  valid_data[dim_col].max()])

                # Both regression lines (same line since both use linear regression)
                # But show with different styles to indicate different correlations
                ax.plot(x_vals, intercept + slope * x_vals,
                       'k-', linewidth=2.5, alpha=0.8, zorder=1, label='Regression')

            # Annotation with BOTH correlations
            sp = corr_data['spearman']
            pe = corr_data['pearson']

            p_str_sp = "p<0.001" if sp['p'] < 0.001 else f"p={sp['p']:.3f}"
            p_str_pe = "p<0.001" if pe['p'] < 0.001 else f"p={pe['p']:.3f}"

            annotation = (f"ρ={sp['rho']:.3f} [{sp['rho_lower']:.3f}, {sp['rho_upper']:.3f}] ({p_str_sp})\n"
                         f"r={pe['r']:.3f} [{pe['r_lower']:.3f}, {pe['r_upper']:.3f}] ({p_str_pe})")

            ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                   fontsize=9, fontweight='medium', fontfamily='Roboto',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white',
                            alpha=0.9, edgecolor='black', linewidth=1.5))

        # Formatting
        is_causal = dim < K
        title_type = "Causal" if is_causal else "Auxiliary"
        ax.set_title(f'Dim {dim} ({title_type}) vs Total Score',
                    fontsize=13, fontweight='bold', fontfamily='Roboto')
        ax.set_xlabel(f'Dim {dim} (signed distance from OFF)',
                     fontsize=11, fontweight='medium', fontfamily='Roboto')
        ax.set_ylabel('Normalized Total Score',
                     fontsize=11, fontweight='medium', fontfamily='Roboto')
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(alpha=0.3, zorder=0)

        if dim == 0:  # Only show legend once
            ax.legend(loc='lower right', fontsize=9,
                     prop={'family': 'Roboto', 'weight': 'medium'})

    # Add 7th subplot for Combined Dim 1+2
    if 'combined_aux_12_signed' in combined_df.columns:
        ax = axes[z_dim]
        dim_col = 'combined_aux_12_signed'

        # Plot points by frequency
        for freq in frequencies:
            freq_str = str(freq)
            freq_data = combined_df[combined_df['condition'] == freq_str]
            valid = freq_data.dropna(subset=[dim_col, 'norm_total'])

            if len(valid) > 0:
                ax.scatter(valid[dim_col], valid['norm_total'],
                          c=[freq_colors[freq_str]], marker=freq_markers[freq_str],
                          s=120, alpha=0.8, edgecolors='black', linewidth=1.5,
                          label=f'{freq} Hz')

                # Optional: Add patient labels
                for _, row in valid.iterrows():
                    ax.annotate(f"P{int(row['patient'])}",
                               xy=(row[dim_col], row['norm_total']),
                               xytext=(3, 3), textcoords='offset points',
                               fontsize=7, alpha=0.6, fontfamily='Roboto')

        # Compute BOTH correlations
        corr_data = compute_both_correlations_with_ci(
            combined_df[dim_col].values,
            combined_df['norm_total'].values
        )

        if corr_data is not None:
            valid_data = combined_df.dropna(subset=[dim_col, 'norm_total'])

            if len(valid_data) >= 3:
                # Regression line
                slope, intercept, _, _, _ = linregress(
                    valid_data[dim_col], valid_data['norm_total']
                )
                x_vals = np.array([valid_data[dim_col].min(),
                                  valid_data[dim_col].max()])

                ax.plot(x_vals, intercept + slope * x_vals,
                       'k-', linewidth=2.5, alpha=0.8, zorder=1, label='Regression')

            # Annotation with BOTH correlations
            sp = corr_data['spearman']
            pe = corr_data['pearson']

            p_str_sp = "p<0.001" if sp['p'] < 0.001 else f"p={sp['p']:.3f}"
            p_str_pe = "p<0.001" if pe['p'] < 0.001 else f"p={pe['p']:.3f}"

            annotation = (f"ρ={sp['rho']:.3f} [{sp['rho_lower']:.3f}, {sp['rho_upper']:.3f}] ({p_str_sp})\n"
                         f"r={pe['r']:.3f} [{pe['r_lower']:.3f}, {pe['r_upper']:.3f}] ({p_str_pe})")

            ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                   fontsize=9, fontweight='medium', fontfamily='Roboto',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white',
                            alpha=0.9, edgecolor='black', linewidth=1.5))

        # Formatting
        ax.set_title(f'Dim 1+2 (Combined) vs Total Score',
                    fontsize=13, fontweight='bold', fontfamily='Roboto')
        ax.set_xlabel(f'Dim 1+2 (signed distance from OFF)',
                     fontsize=11, fontweight='medium', fontfamily='Roboto')
        ax.set_ylabel('Normalized Total Score',
                     fontsize=11, fontweight='medium', fontfamily='Roboto')
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(alpha=0.3, zorder=0)

    # Turn off the 8th subplot (last one in 2×4 grid)
    axes[7].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_correlations_total_detailed.png'),
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'dimension_correlations_total_detailed.svg'),
                bbox_inches='tight')
    plt.close()

    print(f"✓ Saved detailed scatter grid: dimension_correlations_total_detailed")


def plot_auxiliary_vs_symptoms_forest(combined_df: pd.DataFrame, K: int, L: int,
                                       output_dir: str, dpi: int = 600):
    """
    Single forest plot: Auxiliary dimensions vs symptoms (dual Spearman + Pearson annotations)
    9 rows total: 3 symptoms × 3 dims (grouped by symptom)
    """
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.weight'] = 'medium'

    # Okabe-Ito colors for dimensions
    dim_colors = {
        'Dim 1': (0/255, 114/255, 178/255),         # Blue
        'Dim 2': (230/255, 159/255, 0/255),          # Orange
        'Combined (1+2)': (0/255, 158/255, 115/255), # Bluish green
    }

    # Symptom markers
    symptom_markers = {
        'Bradykinesia': 'o',
        'Rigidity': 's',
        'Tremor': '^'
    }

    # Compute combined auxiliary (sum of dim_1 + dim_2)
    if 'dim_1_signed' in combined_df.columns and 'dim_2_signed' in combined_df.columns:
        combined_df['combined_aux_12_signed'] = (
            combined_df['dim_1_signed'] + combined_df['dim_2_signed']
        )

    # Define correlations to compute (GROUPED BY SYMPTOM)
    dimensions = [
        ('dim_1_signed', 'Dim 1'),
        ('dim_2_signed', 'Dim 2'),
        ('combined_aux_12_signed', 'Combined (1+2)')
    ]
    symptoms = [
        ('norm_brady', 'Bradykinesia'),
        ('norm_rigidity', 'Rigidity'),
        ('norm_tremor', 'Tremor')
    ]

    results = []
    # Loop by SYMPTOM first, then dimension (for better grouping)
    for symptom_col, symptom_label in symptoms:
        for dim_col, dim_label in dimensions:
            if dim_col not in combined_df.columns or symptom_col not in combined_df.columns:
                continue

            corr_data = compute_both_correlations_with_ci(
                combined_df[dim_col].values,
                combined_df[symptom_col].values
            )

            if corr_data is not None:
                results.append({
                    'dim_label': dim_label,
                    'symptom_label': symptom_label,
                    'spearman': corr_data['spearman'],
                    'pearson': corr_data['pearson'],
                    'n': corr_data['n']
                })

    if not results:
        print("No valid correlations found for auxiliary vs symptoms forest plot")
        return

    # Create single forest plot with dual annotations
    fig, ax = plt.subplots(figsize=(14, 10))

    n_rows = len(results)
    y_positions = np.arange(n_rows)[::-1]  # Reverse for top-to-bottom

    for i, res in enumerate(results):
        y_pos = y_positions[i]
        color = dim_colors[res['dim_label']]
        marker = symptom_markers[res['symptom_label']]

        # Marker size varies by dimension type
        if res['dim_label'] == 'Combined (1+2)':
            marker_size = 200
        else:
            marker_size = 150

        # Use Spearman for the point + error bar (primary method)
        sp = res['spearman']
        rho = sp['rho']
        rho_lower = sp['rho_lower']
        rho_upper = sp['rho_upper']

        # Point
        ax.scatter(rho, y_pos, s=marker_size, color=color, alpha=0.9,
                  marker=marker, edgecolors='black', linewidth=2.5, zorder=3)

        # Error bar
        xerr_lower = rho - rho_lower
        xerr_upper = rho_upper - rho
        ax.errorbar(rho, y_pos, xerr=[[xerr_lower], [xerr_upper]],
                   fmt='none', ecolor='black', capsize=7, capthick=3,
                   linewidth=2.5, alpha=0.7, zorder=2)

        # Dual annotation (2 lines: Spearman + Pearson)
        pe = res['pearson']
        p_str_sp = "p<0.001" if sp['p'] < 0.001 else f"p={sp['p']:.3f}"
        p_str_pe = "p<0.001" if pe['p'] < 0.001 else f"p={pe['p']:.3f}"

        annotation = (f"ρ={rho:.3f} [{rho_lower:.3f}, {rho_upper:.3f}] ({p_str_sp})\n"
                     f"r={pe['r']:.3f} [{pe['r_lower']:.3f}, {pe['r_upper']:.3f}] ({p_str_pe})")
        ax.text(1.05, y_pos, annotation, ha='left', va='center',
               fontsize=9, fontweight='medium', fontfamily='Roboto')

    # Reference line at 0
    ax.axvline(0, color=(160/255, 160/255, 160/255), linewidth=2,
              linestyle='--', alpha=0.6, zorder=1)

    # Y-axis labels (grouped by symptom)
    y_labels = [f"{res['symptom_label']}: {res['dim_label']}" for res in results]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='medium', fontfamily='Roboto')

    # X-axis
    ax.set_xlabel('Correlation Coefficient', fontsize=14,
                 fontweight='medium', fontfamily='Roboto')
    ax.set_title('Auxiliary Dimensions vs Individual Symptoms\n(Spearman ρ and Pearson r with 95% Confidence Intervals)',
                fontsize=14, fontweight='bold', fontfamily='Roboto', pad=15)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-1, 1.3)  # Extra space for annotations

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []

    # Dimension colors
    for dim_label, color in dim_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label=dim_label)
        )

    # Symptom markers (in gray to show shape)
    gray = (0.5, 0.5, 0.5)
    for symptom, marker in symptom_markers.items():
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor=gray,
                   markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label=symptom)
        )

    ax.legend(handles=legend_elements, loc='lower right', ncol=2, fontsize=11,
             framealpha=0.95, edgecolor='black',
             prop={'family': 'Roboto', 'weight': 'medium'})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auxiliary_correlations_symptoms_forest.png'),
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'auxiliary_correlations_symptoms_forest.svg'),
                bbox_inches='tight')
    plt.close()

    print(f"✓ Saved forest plot: auxiliary_correlations_symptoms_forest")


def plot_auxiliary_vs_symptoms_detailed(combined_df: pd.DataFrame, K: int, L: int,
                                          frequencies: List[int], output_dir: str,
                                          dpi: int = 600):
    """
    Detailed scatter subplots: Symptoms (rows) vs Auxiliary dims (cols)
    3×3 grid: Rows = symptoms, Columns = dimensions
    Shows both Spearman and Pearson in annotations
    """
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.weight'] = 'medium'

    # Okabe-Ito colors for frequencies
    freq_colors = {
        '125': (230/255, 159/255, 0/255),      # Orange
        '110': (86/255, 180/255, 233/255),     # Sky blue
        '85': (0/255, 158/255, 115/255),       # Bluish green
        '55': (213/255, 94/255, 0/255),        # Vermillion
    }
    freq_markers = {
        '125': 'o', '110': 's', '85': '^', '55': 'D'
    }

    # Compute combined auxiliary
    if 'dim_1_signed' in combined_df.columns and 'dim_2_signed' in combined_df.columns:
        combined_df['combined_aux_12_signed'] = (
            combined_df['dim_1_signed'] + combined_df['dim_2_signed']
        )

    # Define grid: Rows = symptoms, Columns = dimensions
    symptoms = [
        ('norm_brady', 'Bradykinesia'),
        ('norm_rigidity', 'Rigidity'),
        ('norm_tremor', 'Tremor')
    ]
    dimensions = [
        ('dim_1_signed', 'Dim 1'),
        ('dim_2_signed', 'Dim 2'),
        ('combined_aux_12_signed', 'Combined (1+2)')
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row, (symptom_col, symptom_label) in enumerate(symptoms):
        for col, (dim_col, dim_label) in enumerate(dimensions):
            ax = axes[row, col]

            if dim_col not in combined_df.columns or symptom_col not in combined_df.columns:
                ax.axis('off')
                continue

            # Plot points by frequency
            for freq in frequencies:
                freq_str = str(freq)
                freq_data = combined_df[combined_df['condition'] == freq_str]
                valid = freq_data.dropna(subset=[dim_col, symptom_col])

                if len(valid) > 0:
                    ax.scatter(valid[dim_col], valid[symptom_col],
                              c=[freq_colors[freq_str]], marker=freq_markers[freq_str],
                              s=100, alpha=0.8, edgecolors='black', linewidth=1.5,
                              label=f'{freq} Hz')

                    # Optional: Patient labels
                    for _, prow in valid.iterrows():
                        ax.annotate(f"P{int(prow['patient'])}",
                                   xy=(prow[dim_col], prow[symptom_col]),
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=7, alpha=0.6, fontfamily='Roboto')

            # Compute BOTH correlations
            corr_data = compute_both_correlations_with_ci(
                combined_df[dim_col].values,
                combined_df[symptom_col].values
            )

            if corr_data is not None:
                valid_data = combined_df.dropna(subset=[dim_col, symptom_col])

                if len(valid_data) >= 3:
                    # Regression line
                    slope, intercept, _, _, _ = linregress(
                        valid_data[dim_col], valid_data[symptom_col]
                    )
                    x_vals = np.array([valid_data[dim_col].min(),
                                      valid_data[dim_col].max()])
                    ax.plot(x_vals, intercept + slope * x_vals,
                           'k-', linewidth=2.5, alpha=0.8, zorder=1, label='Regression')

                # Annotation with BOTH correlations
                sp = corr_data['spearman']
                pe = corr_data['pearson']

                p_str_sp = "p<0.001" if sp['p'] < 0.001 else f"p={sp['p']:.3f}"
                p_str_pe = "p<0.001" if pe['p'] < 0.001 else f"p={pe['p']:.3f}"

                annotation = (f"ρ={sp['rho']:.3f} [{sp['rho_lower']:.3f}, {sp['rho_upper']:.3f}] ({p_str_sp})\n"
                             f"r={pe['r']:.3f} [{pe['r_lower']:.3f}, {pe['r_upper']:.3f}] ({p_str_pe})")

                ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                       fontsize=8, fontweight='medium', fontfamily='Roboto',
                       va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.9, edgecolor='black', linewidth=1.5))

            # Formatting
            ax.set_title(f'{symptom_label}: {dim_label}',
                        fontsize=12, fontweight='bold', fontfamily='Roboto')
            ax.set_xlabel(f'{dim_label} (signed distance)',
                         fontsize=10, fontweight='medium', fontfamily='Roboto')
            ax.set_ylabel(f'Normalized {symptom_label}',
                         fontsize=10, fontweight='medium', fontfamily='Roboto')
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(alpha=0.3, zorder=0)

            if row == 0 and col == 0:  # Legend only on first subplot
                ax.legend(loc='lower right', fontsize=8,
                         prop={'family': 'Roboto', 'weight': 'medium'})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auxiliary_correlations_symptoms_detailed.png'),
                dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'auxiliary_correlations_symptoms_detailed.svg'),
                bbox_inches='tight')
    plt.close()

    print(f"✓ Saved detailed scatter grid: auxiliary_correlations_symptoms_detailed")


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_results_summary(all_results: Dict, model_info: Dict, dim_combinations: List[Tuple]):
    """Print formatted summary of all correlation results (SIGNED ONLY)"""
    K, L, z_dim = model_info['K'], model_info['L'], model_info['z_dim']

    print("\n" + "=" * 90)
    print(f"FULL-AMPLITUDE ANALYSIS (SIGNED DIMENSIONS ONLY)")
    print(f"MODEL: K={K} (causal), L={L} (auxiliary), z_dim={z_dim}")
    print("=" * 90)

    for method, results_df in all_results.items():
        print(f"\n{'=' * 90}")
        print(f"{method.upper()} CORRELATIONS")
        print("=" * 90)

        if results_df.empty:
            print("No results")
            continue

        categories = {
            'Individual Dimensions (Signed)': [f'dim_{i}_signed' for i in range(z_dim)],
            'Combined Dimensions (Sum Signed)': [f'dims_{"_".join(map(str, d))}_sum_signed' for d in dim_combinations],
        }

        for measure in results_df['clinical_measure'].unique():
            print(f"\n{measure}:")
            print("-" * 80)
            measure_df = results_df[results_df['clinical_measure'] == measure]

            for cat_name, cat_prefixes in categories.items():
                cat_results = []
                for _, row in measure_df.iterrows():
                    if any(row['distance_metric'].startswith(p) or row['distance_metric'] == p
                           for p in cat_prefixes):
                        cat_results.append(row)

                if cat_results:
                    print(f"\n  {cat_name}:")
                    for row in cat_results:
                        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row[
                            'significant'] else ''
                        print(
                            f"    {row['distance_metric']:35s}: r={row['correlation']:7.3f}, p={row['p_value']:.4f} {sig}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis(config: Dict):
    """Run complete correlation analysis pipeline"""

    os.makedirs(config['output_dir'], exist_ok=True)

    # Model parameters
    model_params = extract_parameters(config['model_path'])
    K, L = model_params['K'], model_params['L']
    z_dim = K + L

    print(f"\n{'=' * 60}")
    print(f"FULL-AMPLITUDE STIMULATION ANALYSIS")
    print(f"Model: K={K}, L={L}, z_dim={z_dim}, λ={model_params['lambda']}")
    print(f"Frequencies: {config['frequencies']}")
    print(f"Dimension combinations: {config['dim_combinations']}")
    print(f"{'=' * 60}\n")

    # Validate combinations
    valid_combinations = [d for d in config['dim_combinations'] if all(dim < z_dim for dim in d)]

    base_config = {
        "vae_model": "vanilla_Encoder", "n_features": 70,
        "alpha_samples": 1, "beta_samples": 1,
        "include_classes": None, "save_dir": "./", "model_name": "vae_model"
    }

    # Load encoder and data
    encoder = load_encoder_for_params(base_config,
                                      ModelParams(0, K, L, model_params['lambda'], config['model_path']))
    X_all, conditions, subject_indices = load_data(config['data_path'], config['frequencies'])
    normalized_scores = load_normalized_scores(config['scores_path'], config['frequencies'])
    latent_dims = get_latent_representations(encoder, X_all, subject_indices)

    # Compute distances
    distance_dict = compute_all_distances(
        latent_dims, conditions, subject_indices,
        config['frequencies'], K, L, valid_combinations
    )
    combined_df = combine_distances_with_scores(distance_dict, normalized_scores, config['frequencies'])

    if len(combined_df) == 0:
        print("\nERROR: No matching data found!")
        return None, None, None

    # ==========================================================================
    # BUILD LIST OF SIGNED METRICS ONLY (NO ABS OR NORM)
    # ==========================================================================

    distance_cols = []

    # 1. Individual dimensions - SIGNED ONLY
    for dim in range(z_dim):
        distance_cols.append(f'dim_{dim}_signed')

    # 2. Combined dimension metrics - SUM_SIGNED ONLY
    for dims in valid_combinations:
        dim_label = '_'.join(map(str, dims))
        distance_cols.append(f'dims_{dim_label}_sum_signed')

    print(f"\nTotal metrics to analyze (SIGNED ONLY): {len(distance_cols)}")
    print("Metrics:")
    for col in distance_cols:
        print(f"  - {col}")

    # Calculate correlations
    all_results = {}
    for method in config['correlation_methods']:
        print(f"Calculating {method} correlations...")
        all_results[method] = calculate_correlations(combined_df, distance_cols, method)

    # Save data
    combined_df.to_csv(os.path.join(config['output_dir'], 'fullamp_combined_data.csv'), index=False)
    for method, results_df in all_results.items():
        results_df.to_csv(os.path.join(config['output_dir'], f'fullamp_{method}_correlations.csv'), index=False)

    # ==========================================================================
    # GENERATE PLOTS FOR ALL METRICS
    # ==========================================================================

    if config['generate_plots']:
        print("\n" + "=" * 60)
        print("Generating plots for ALL metrics...")
        print("=" * 60)

        # Plot EVERY metric
        for i, metric in enumerate(distance_cols):
            print(f"\n[{i + 1}/{len(distance_cols)}] Plotting: {metric}")

            if metric not in combined_df.columns:
                print(f"  Skipping - column not found")
                continue

            # 1. Correlation matrix style (frequency colors)
            plot_correlation_matrix(combined_df, all_results['spearman'], metric,
                                    config['frequencies'], config['output_dir'], config['plot_dpi'])

            # 2. By condition plots
            plot_correlation_by_condition(combined_df, metric, config['output_dir'], config['plot_dpi'])

            # 3. Patient scatter plots
            plot_scatter_grid(combined_df, all_results['spearman'], metric,
                              config['frequencies'], config['output_dir'], config['plot_dpi'])

        # Heatmaps for all methods
        print("\nGenerating heatmaps...")
        for method in config['correlation_methods']:
            if method in all_results:
                plot_correlation_heatmap(all_results[method], config['output_dir'], method, config['plot_dpi'])

    model_info = {'K': K, 'L': L, 'z_dim': z_dim}
    print_results_summary(all_results, model_info, valid_combinations)

    # ==========================================================================
    # GENERATE NEW HYBRID PLOTS (DUAL SPEARMAN + PEARSON)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Generating Dual Correlation Plots (Spearman + Pearson)")
    print("=" * 60)

    # Plot 1: All dimensions vs Total Score
    print("\n[Plot 1] All dimensions vs Total Score:")
    plot_dimensions_vs_total_forest(combined_df, K, L, config['output_dir'], config['plot_dpi'])
    plot_dimensions_vs_total_detailed(combined_df, K, L, config['frequencies'],
                                       config['output_dir'], config['plot_dpi'])

    # Plot 2: Auxiliary dimensions vs Individual symptoms
    print("\n[Plot 2] Auxiliary dimensions vs Individual symptoms:")
    plot_auxiliary_vs_symptoms_forest(combined_df, K, L, config['output_dir'], config['plot_dpi'])
    plot_auxiliary_vs_symptoms_detailed(combined_df, K, L, config['frequencies'],
                                         config['output_dir'], config['plot_dpi'])

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {config['output_dir']}/")
    print(f"Total plots generated: {len(distance_cols) * 3 + len(config['correlation_methods'])} + 4 new dual correlation plots")
    print(f"{'=' * 60}")

    return combined_df, all_results, model_info


if __name__ == "__main__":
    combined_df, correlation_results, model_info = run_full_analysis(CONFIG)