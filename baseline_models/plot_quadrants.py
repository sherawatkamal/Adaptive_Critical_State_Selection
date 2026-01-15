#!/usr/bin/env python3
"""
Enrichment Analysis for Teachable Moments
Validates that uncertainty-based selection beats random sampling
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import argparse

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(csv_path: str) -> pd.DataFrame:
    """Load processed data with quadrant assignments."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points")
    return df

def compute_enrichment_metrics(df: pd.DataFrame, u_column: str = 'U', 
                               l_column: str = 'L', l_threshold: float = 75):
    """
    Compute enrichment of recoverable states when selecting by uncertainty.
    
    Returns:
        dict with AUC, precision@k, and enrichment ratios
    """
    # Define recoverable as L > threshold
    df['recoverable'] = (df[l_column] > l_threshold).astype(int)
    
    baseline_rate = df['recoverable'].mean()
    
    # Sort by uncertainty (descending - highest U first)
    df_sorted = df.sort_values(u_column, ascending=False).reset_index(drop=True)
    
    metrics = {
        'baseline_rate': baseline_rate,
        'n_total': len(df),
        'n_recoverable': df['recoverable'].sum(),
    }
    
    # Top-K precision (selecting by highest uncertainty)
    for k_pct in [10, 20, 30, 50]:
        k = int(len(df) * k_pct / 100)
        top_k_recoverable = df_sorted.head(k)['recoverable'].sum()
        top_k_rate = top_k_recoverable / k
        enrichment = top_k_rate / baseline_rate if baseline_rate > 0 else 0
        
        metrics[f'top_{k_pct}pct_rate'] = top_k_rate
        metrics[f'top_{k_pct}pct_enrichment'] = enrichment
        metrics[f'top_{k_pct}pct_count'] = int(top_k_recoverable)
    
    # Middle-K selection (advisor's "Goldilocks zone" hypothesis)
    for low_pct, high_pct in [(30, 50), (40, 60), (35, 65)]:
        low_idx = int(len(df) * low_pct / 100)
        high_idx = int(len(df) * high_pct / 100)
        
        mid_k_recoverable = df_sorted.iloc[low_idx:high_idx]['recoverable'].sum()
        mid_k_size = high_idx - low_idx
        mid_k_rate = mid_k_recoverable / mid_k_size if mid_k_size > 0 else 0
        enrichment = mid_k_rate / baseline_rate if baseline_rate > 0 else 0
        
        metrics[f'mid_{low_pct}_{high_pct}pct_rate'] = mid_k_rate
        metrics[f'mid_{low_pct}_{high_pct}pct_enrichment'] = enrichment
        metrics[f'mid_{low_pct}_{high_pct}pct_count'] = int(mid_k_recoverable)
    
    # AUC for uncertainty as predictor of recoverability
    if len(df['recoverable'].unique()) > 1:
        try:
            auc_score = roc_auc_score(df['recoverable'], df[u_column])
            metrics['auc'] = auc_score
        except:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5
    
    # Spearman correlation
    corr, p_value = stats.spearmanr(df[u_column], df[l_column])
    metrics['spearman_corr'] = corr
    metrics['spearman_pvalue'] = p_value
    
    return metrics, df

def plot_binned_bar_chart(df: pd.DataFrame, output_path: str, 
                          u_column: str = 'U', l_column: str = 'L',
                          l_threshold: float = 75):
    """
    Binned bar chart showing P(Recoverable) vs Uncertainty bins.
    This is the "hill" plot the advisor wants to see.
    """
    # Create uncertainty bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    df['U_bin'] = pd.cut(df[u_column], bins=bins, labels=bin_labels, include_lowest=True)
    df['recoverable'] = (df[l_column] > l_threshold).astype(int)
    
    # Compute stats per bin
    bin_stats = []
    for bin_label in bin_labels:
        bin_data = df[df['U_bin'] == bin_label]
        if len(bin_data) > 0:
            n_total = len(bin_data)
            n_recoverable = bin_data['recoverable'].sum()
            rate = n_recoverable / n_total
            
            # Standard error for error bars
            se = np.sqrt(rate * (1 - rate) / n_total) if n_total > 0 else 0
            
            bin_stats.append({
                'bin': bin_label,
                'rate': rate,
                'se': se,
                'n_total': n_total,
                'n_recoverable': int(n_recoverable)
            })
    
    bin_df = pd.DataFrame(bin_stats)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(bin_df))
    bars = ax.bar(x_pos, bin_df['rate'], yerr=bin_df['se'], 
                   capsize=5, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Color the bars by height (highlight the peak)
    max_idx = bin_df['rate'].idxmax()
    for i, bar in enumerate(bars):
        if i == max_idx:
            bar.set_color('green')
            bar.set_alpha(0.9)
    
    # Add value labels on bars
    for i, (rate, n_rec, n_tot) in enumerate(zip(bin_df['rate'], 
                                                   bin_df['n_recoverable'], 
                                                   bin_df['n_total'])):
        ax.text(i, rate + bin_df.iloc[i]['se'] + 0.02, 
               f'{100*rate:.1f}%\n({n_rec}/{n_tot})', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_df['bin'], fontsize=11)
    ax.set_xlabel('Uncertainty Bins', fontsize=13, fontweight='bold')
    ax.set_ylabel('P(Recoverable | L > 75)', fontsize=13, fontweight='bold')
    ax.set_title('The "Goldilocks Zone": Recoverability vs Uncertainty\nShowing the Inverted-U Hypothesis', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add baseline horizontal line
    baseline = df['recoverable'].mean()
    ax.axhline(baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Baseline (Random): {100*baseline:.1f}%')
    
    ax.set_ylim(0, max(bin_df['rate'].max() * 1.2, baseline * 1.5))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return bin_df

def plot_enrichment_curves(df: pd.DataFrame, output_path: str,
                          u_column: str = 'U', l_column: str = 'L',
                          l_threshold: float = 75):
    """
    Cumulative enrichment curves showing yield vs selection size.
    """
    df['recoverable'] = (df[l_column] > l_threshold).astype(int)
    baseline_rate = df['recoverable'].mean()
    
    # Sort by uncertainty (descending)
    df_sorted_u = df.sort_values(u_column, ascending=False).reset_index(drop=True)
    
    # Sort by recoverability (for oracle)
    df_sorted_oracle = df.sort_values(l_column, ascending=False).reset_index(drop=True)
    
    # Compute cumulative precision
    n_points = 100
    x = np.linspace(0, len(df), n_points, dtype=int)
    
    precision_u = []
    precision_oracle = []
    precision_random = []
    
    for k in x:
        if k == 0:
            precision_u.append(0)
            precision_oracle.append(0)
            precision_random.append(0)
        else:
            # Uncertainty-based selection
            p_u = df_sorted_u.head(k)['recoverable'].sum() / k
            precision_u.append(p_u)
            
            # Oracle (upper bound)
            p_oracle = df_sorted_oracle.head(k)['recoverable'].sum() / k
            precision_oracle.append(p_oracle)
            
            # Random (baseline)
            precision_random.append(baseline_rate)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pct = 100 * x / len(df)
    
    ax.plot(x_pct, precision_oracle, 'g-', linewidth=3, label='Oracle (L-sorted)', alpha=0.8)
    ax.plot(x_pct, precision_u, 'b-', linewidth=3, label='Uncertainty-based', alpha=0.8)
    ax.plot(x_pct, precision_random, 'r--', linewidth=2, label='Random baseline', alpha=0.8)
    
    # Fill area between uncertainty and random
    ax.fill_between(x_pct, precision_u, precision_random, 
                    where=(np.array(precision_u) >= np.array(precision_random)),
                    alpha=0.3, color='blue', label='Enrichment gain')
    
    ax.set_xlabel('% of States Selected', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision (% Recoverable in Selection)', fontsize=13, fontweight='bold')
    ax.set_title('Enrichment Analysis: Does Uncertainty Predict Recoverability?', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Add annotation for AUC-like metric
    auc_uncertainty = np.trapz(precision_u, x_pct / 100)
    auc_random = baseline_rate
    lift = (auc_uncertainty - auc_random) / auc_random if auc_random > 0 else 0
    
    ax.text(0.98, 0.02, f'AUC (Uncertainty): {auc_uncertainty:.3f}\n'
                        f'AUC (Random): {auc_random:.3f}\n'
                        f'Lift: {100*lift:.1f}%',
           transform=ax.transAxes, fontsize=10, 
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_2d_heatmap(df: pd.DataFrame, output_path: str,
                   u_column: str = 'U', l_column: str = 'L'):
    """
    2D histogram heatmap showing density of (U, L) combinations.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    h, xedges, yedges, im = ax.hist2d(df[u_column], df[l_column], 
                                       bins=[20, 20], cmap='YlOrRd', 
                                       cmin=1)
    
    ax.set_xlabel('Uncertainty (U)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recoverability (L)', fontsize=13, fontweight='bold')
    ax.set_title('2D Density Heatmap: The "Hill" Visualization', 
                fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold')
    
    # Add quadrant lines
    u_thresh = df[u_column].median()
    l_thresh = df[l_column].median()
    ax.axvline(u_thresh, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(l_thresh, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def identify_case_studies(df: pd.DataFrame, output_path: str,
                         u_column: str = 'U', l_column: str = 'L',
                         l_threshold: float = 75):
    """
    Identify exemplar cases for qualitative analysis.
    As requested: 5 High U + Unrecoverable (Noise) vs 5 High U + Recoverable (Teachable)
    """
    df['recoverable'] = df[l_column] > l_threshold
    
    # Define "high U" as top 30%
    u_75 = df[u_column].quantile(0.70)
    high_u = df[df[u_column] > u_75]
    
    # High U + Unrecoverable (Noise)
    noise_cases = high_u[~high_u['recoverable']].nlargest(5, u_column)
    
    # High U + Recoverable (Teachable)
    teachable_cases = high_u[high_u['recoverable']].nlargest(5, u_column)
    
    case_studies = {
        'noise_high_u_unrecoverable': noise_cases[['task_id', 'state_idx', u_column, 
                                                    l_column, 'improvement', 'success']].to_dict('records'),
        'teachable_high_u_recoverable': teachable_cases[['task_id', 'state_idx', u_column, 
                                                          l_column, 'improvement', 'success']].to_dict('records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(case_studies, f, indent=2)
    
    print(f"✓ Saved case studies: {output_path}")
    print(f"\nNoise cases (High U + Unrecoverable):")
    for i, case in enumerate(case_studies['noise_high_u_unrecoverable'], 1):
        print(f"  {i}. Task {case['task_id']}, Step {case['state_idx']}: "
              f"U={case[u_column]:.3f}, L={case[l_column]:.1f}")
    
    print(f"\nTeachable cases (High U + Recoverable):")
    for i, case in enumerate(case_studies['teachable_high_u_recoverable'], 1):
        print(f"  {i}. Task {case['task_id']}, Step {case['state_idx']}: "
              f"U={case[u_column]:.3f}, L={case[l_column]:.1f}")
    
    return case_studies

def print_enrichment_summary(metrics: dict):
    """Print formatted enrichment metrics."""
    print("\n" + "="*70)
    print("ENRICHMENT ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total states:          {metrics['n_total']:,}")
    print(f"Recoverable states:    {metrics['n_recoverable']:,}")
    print(f"Baseline rate:         {100*metrics['baseline_rate']:.2f}%")
    print(f"\nAUC (U predicts L):    {metrics['auc']:.3f}")
    print(f"Spearman correlation:  {metrics['spearman_corr']:.3f} (p={metrics['spearman_pvalue']:.2e})")
    
    print("\n" + "-"*70)
    print("TOP-K SELECTION (Highest Uncertainty)")
    print("-"*70)
    for k in [10, 20, 30, 50]:
        rate = metrics[f'top_{k}pct_rate']
        enrichment = metrics[f'top_{k}pct_enrichment']
        count = metrics[f'top_{k}pct_count']
        print(f"Top {k:2d}%: {100*rate:5.1f}% recoverable ({count:3d} states) "
              f"| Enrichment: {enrichment:.2f}x")
    
    print("\n" + "-"*70)
    print("MIDDLE-K SELECTION (Goldilocks Zone)")
    print("-"*70)
    for low, high in [(30, 50), (40, 60), (35, 65)]:
        rate = metrics[f'mid_{low}_{high}pct_rate']
        enrichment = metrics[f'mid_{low}_{high}pct_enrichment']
        count = metrics[f'mid_{low}_{high}pct_count']
        print(f"Mid {low}-{high}%: {100*rate:5.1f}% recoverable ({count:3d} states) "
              f"| Enrichment: {enrichment:.2f}x")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Enrichment analysis for teachable moments')
    parser.add_argument('--data', required=True, help='Path to data_with_quadrants.csv')
    parser.add_argument('--output-dir', default='./enrichment_analysis')
    parser.add_argument('--l-threshold', type=float, default=75,
                       help='Threshold for defining recoverable states')
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    
    print("\nComputing enrichment metrics...")
    metrics, df_labeled = compute_enrichment_metrics(df, l_threshold=args.l_threshold)
    
    print_enrichment_summary(metrics)
    
    # Save metrics
    with open(out / 'enrichment_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {out / 'enrichment_metrics.json'}")
    
    print("\nGenerating plots...")
    
    # 1. The "Hill" plot (binned bar chart)
    bin_df = plot_binned_bar_chart(df_labeled, out / 'binned_bar_chart.png',
                                    l_threshold=args.l_threshold)
    
    # 2. Enrichment curves
    plot_enrichment_curves(df_labeled, out / 'enrichment_curves.png',
                          l_threshold=args.l_threshold)
    
    # 3. 2D heatmap
    plot_2d_heatmap(df_labeled, out / '2d_heatmap.png')
    
    # 4. Case studies
    case_studies = identify_case_studies(df_labeled, out / 'case_studies.json',
                                        l_threshold=args.l_threshold)
    
    print(f"\n✓ All outputs saved to: {out}")
    print("\nGenerated files:")
    print("  1. binned_bar_chart.png - The 'hill' showing Goldilocks zone")
    print("  2. enrichment_curves.png - Cumulative precision vs selection size")
    print("  3. 2d_heatmap.png - Density heatmap")
    print("  4. enrichment_metrics.json - All numeric metrics")
    print("  5. case_studies.json - Exemplar states for qualitative analysis")

if __name__ == '__main__':
    main()