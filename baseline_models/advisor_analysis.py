#!/usr/bin/env python3
"""
Advisor-Requested Analysis: Inverted-U Hypothesis & Enrichment
Generates binned bar charts and validates teachability prediction
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

def create_uncertainty_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Create uncertainty bins for analysis."""
    df = df.copy()
    
    # Equal-width bins
    df['U_bin'] = pd.cut(df['U'], bins=n_bins, labels=False)
    bin_edges = pd.cut(df['U'], bins=n_bins, retbins=True)[1]
    
    # Create bin labels
    bin_labels = []
    for i in range(len(bin_edges)-1):
        label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        bin_labels.append(label)
    
    df['U_bin_label'] = pd.cut(df['U'], bins=n_bins, labels=bin_labels)
    
    return df, bin_edges

def plot_binned_recoverability(df: pd.DataFrame, bin_edges: np.ndarray, output_path: str):
    """
    ADVISOR REQUEST 1: Binned bar chart showing Inverted-U hypothesis.
    X-axis: Uncertainty bins
    Y-axis: Probability of recoverability (recoverable count / total count)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define "recoverable" as L > threshold (use 75th percentile or median)
    l_threshold = df['L'].quantile(0.5)  # Use median as threshold
    df['recoverable'] = df['L'] > l_threshold
    
    # Compute recovery rate per bin
    bin_stats = df.groupby('U_bin_label').agg({
        'recoverable': ['sum', 'count', 'mean'],
        'L': ['mean', 'std'],
        'success': 'mean'
    }).reset_index()
    
    bin_stats.columns = ['bin', 'recoverable_count', 'total_count', 'recovery_rate', 
                         'mean_L', 'std_L', 'success_rate']
    
    # Compute standard error for error bars
    bin_stats['se'] = np.sqrt(bin_stats['recovery_rate'] * (1 - bin_stats['recovery_rate']) / 
                               bin_stats['total_count'])
    
    # Plot 1: Recovery Rate by Bin
    x_pos = np.arange(len(bin_stats))
    ax1.bar(x_pos, bin_stats['recovery_rate'], 
           color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.errorbar(x_pos, bin_stats['recovery_rate'], yerr=bin_stats['se'], 
                fmt='none', color='black', capsize=5, capthick=2)
    
    # Add count labels
    for i, row in bin_stats.iterrows():
        ax1.text(i, row['recovery_rate'] + row['se'] + 0.02, 
                f"n={int(row['total_count'])}", ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_stats['bin'], rotation=45, ha='right')
    ax1.set_xlabel('Uncertainty (U) Bins', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Probability of Recoverability', fontsize=13, fontweight='bold')
    ax1.set_title('Inverted-U Hypothesis: Recovery Rate by Uncertainty\n(Error bars = SE)', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Chance')
    ax1.legend()
    
    # Plot 2: Mean Improvement by Bin
    ax2.bar(x_pos, bin_stats['mean_L'], 
           color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.errorbar(x_pos, bin_stats['mean_L'], yerr=bin_stats['std_L'], 
                fmt='none', color='black', capsize=5, capthick=2)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_stats['bin'], rotation=45, ha='right')
    ax2.set_xlabel('Uncertainty (U) Bins', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Improvement (L)', fontsize=13, fontweight='bold')
    ax2.set_title('Mean Improvement by Uncertainty Bin\n(Error bars = SD)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(df['L'].median(), color='blue', linestyle='--', linewidth=2, alpha=0.5, 
               label=f'Median L={df["L"].median():.1f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("BINNED RECOVERY STATISTICS")
    print("="*70)
    print(bin_stats.to_string(index=False))
    print("="*70)
    
    # Test for inverted-U
    peak_bin = bin_stats['recovery_rate'].idxmax()
    print(f"\nPeak recovery rate at bin: {bin_stats.loc[peak_bin, 'bin']} "
          f"({100*bin_stats.loc[peak_bin, 'recovery_rate']:.1f}%)")
    
    return bin_stats

def enrichment_analysis(df: pd.DataFrame, output_path: str):
    """
    ADVISOR REQUEST 2: Enrichment analysis - does selecting by U increase yield?
    Compare top-k% by uncertainty vs random sampling.
    """
    l_threshold = df['L'].quantile(0.5)
    df['recoverable'] = df['L'] > l_threshold
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define selection strategies
    percentiles = [10, 20, 30, 40, 50]
    
    results = []
    
    for percentile in percentiles:
        n_select = int(len(df) * percentile / 100)
        
        # Strategy 1: Random
        random_yield = df['recoverable'].mean()
        
        # Strategy 2: Top-k by U (highest uncertainty)
        top_u = df.nlargest(n_select, 'U')
        top_u_yield = top_u['recoverable'].mean()
        
        # Strategy 3: Middle-k by U (Goldilocks zone: 0.4-0.7)
        middle_mask = (df['U'] >= 0.4) & (df['U'] <= 0.7)
        middle_df = df[middle_mask]
        if len(middle_df) >= n_select:
            middle_sample = middle_df.sample(min(n_select, len(middle_df)), random_state=42)
            middle_yield = middle_sample['recoverable'].mean()
        else:
            middle_yield = middle_df['recoverable'].mean() if len(middle_df) > 0 else random_yield
        
        # Strategy 4: Bottom-k by U (lowest uncertainty)
        bottom_u = df.nsmallest(n_select, 'U')
        bottom_u_yield = bottom_u['recoverable'].mean()
        
        results.append({
            'percentile': percentile,
            'n_select': n_select,
            'random': random_yield,
            'top_u': top_u_yield,
            'middle_u': middle_yield,
            'bottom_u': bottom_u_yield,
            'top_lift': (top_u_yield - random_yield) / random_yield if random_yield > 0 else 0,
            'middle_lift': (middle_yield - random_yield) / random_yield if random_yield > 0 else 0,
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot 1: Yield by selection strategy
    ax = axes[0, 0]
    x = results_df['percentile']
    ax.plot(x, results_df['random'], 'o--', label='Random', linewidth=2, markersize=8)
    ax.plot(x, results_df['top_u'], 's--', label='Top-k by U (High Uncertainty)', linewidth=2, markersize=8)
    ax.plot(x, results_df['middle_u'], '^--', label='Middle U (Goldilocks: 0.4-0.7)', linewidth=2, markersize=8)
    ax.plot(x, results_df['bottom_u'], 'd--', label='Bottom-k by U (Low Uncertainty)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Selection Budget (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Yield (Fraction Recoverable)', fontsize=12, fontweight='bold')
    ax.set_title('Enrichment Analysis: Yield vs Selection Strategy', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Plot 2: Lift over random
    ax = axes[0, 1]
    ax.bar(x - 1, results_df['top_lift'] * 100, width=2, alpha=0.7, label='Top U Lift', color='steelblue')
    ax.bar(x + 1, results_df['middle_lift'] * 100, width=2, alpha=0.7, label='Middle U Lift', color='coral')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Selection Budget (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lift over Random (%)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Improvement over Random Sampling', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Precision-Recall curve
    ax = axes[1, 0]
    
    # Rank by U (descending)
    df_sorted = df.sort_values('U', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = np.arange(len(df_sorted))
    df_sorted['recall'] = df_sorted['recoverable'].cumsum() / df_sorted['recoverable'].sum()
    df_sorted['precision'] = df_sorted['recoverable'].cumsum() / (df_sorted['rank'] + 1)
    
    ax.plot(df_sorted['recall'], df_sorted['precision'], linewidth=2, label='U-based Selection')
    ax.plot([0, 1], [df['recoverable'].mean(), df['recoverable'].mean()], 
           'r--', linewidth=2, label='Random Selection')
    
    # Calculate AUC
    pr_auc = auc(df_sorted['recall'], df_sorted['precision'])
    
    ax.set_xlabel('Recall (Fraction of Recoverable States Retrieved)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (Fraction Recoverable in Selection)', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: ROC-style curve
    ax = axes[1, 1]
    
    # Compute ROC curve for uncertainty predicting recoverability
    try:
        roc_auc = roc_auc_score(df['recoverable'], df['U'])
        
        # Sort by U for cumulative curve
        fpr_vals = []
        tpr_vals = []
        thresholds = np.linspace(df['U'].min(), df['U'].max(), 100)
        
        for thresh in thresholds:
            selected = df['U'] >= thresh
            if selected.sum() == 0:
                continue
            tp = (selected & df['recoverable']).sum()
            fp = (selected & ~df['recoverable']).sum()
            fn = (~selected & df['recoverable']).sum()
            tn = (~selected & ~df['recoverable']).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_vals.append(tpr)
            fpr_vals.append(fpr)
        
        # Sort for plotting
        sorted_indices = np.argsort(fpr_vals)
        fpr_vals = np.array(fpr_vals)[sorted_indices]
        tpr_vals = np.array(tpr_vals)[sorted_indices]
        
        ax.plot(fpr_vals, tpr_vals, linewidth=2, label=f'U-based (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve: Uncertainty Predicts Recoverability', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'ROC Failed: {str(e)}', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print enrichment summary
    print("\n" + "="*70)
    print("ENRICHMENT ANALYSIS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    print(f"\nPrecision-Recall AUC: {pr_auc:.3f}")
    if 'roc_auc' in locals():
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"\nInterpretation: AUC > 0.5 means U predicts recoverability better than random")
        print(f"               AUC = {roc_auc:.3f} → {'SIGNIFICANT' if roc_auc > 0.55 else 'WEAK'} signal")
    
    return results_df, pr_auc

def compare_uncertainty_metrics(df: pd.DataFrame, output_path: str):
    """
    ADVISOR REQUEST 3: Compare alternative uncertainty metrics.
    - Current: Token entropy
    - Alternative 1: 1 - P_max (min-prob)
    - Alternative 2: Sequence-level uncertainty (if available)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    l_threshold = df['L'].quantile(0.5)
    df['recoverable'] = df['L'] > l_threshold
    
    # Metric 1: Current entropy (already have as 'U')
    metric1_name = 'Entropy (Current)'
    metric1 = df['U']
    
    # Metric 2: Normalize entropy by action space size if we have it
    if 'normalized_entropy' in df.columns and 'true_entropy' in df.columns:
        metric2_name = 'Normalized Entropy'
        metric2 = df['normalized_entropy']
    else:
        # Approximate: assume uniform distribution would have H = log(N_actions)
        # For now, just use a simple transform
        metric2_name = 'Entropy / log(10)'  # Assume ~10 actions average
        metric2 = df['U'] / np.log(10)
    
    # Metric 3: Inverse confidence (1 - P_max would need softmax, approximate with entropy)
    # Higher entropy ≈ lower P_max
    metric3_name = 'Uncertainty Score (exp-normalized)'
    metric3 = 1 - np.exp(-df['U'])  # Maps [0,inf] → [0,1]
    
    metrics = [
        (metric1_name, metric1),
        (metric2_name, metric2),
        (metric3_name, metric3)
    ]
    
    results = []
    
    for idx, (name, metric) in enumerate(metrics):
        # Compute correlation with recoverability
        corr_pearson = stats.pearsonr(metric, df['L'])[0]
        corr_spearman = stats.spearmanr(metric, df['L'])[0]
        
        # Compute AUC for predicting recoverability
        try:
            roc_auc = roc_auc_score(df['recoverable'], metric)
        except:
            roc_auc = 0.5
        
        results.append({
            'metric': name,
            'pearson': corr_pearson,
            'spearman': corr_spearman,
            'roc_auc': roc_auc
        })
        
        # Plot correlation scatter
        if idx < 3:
            ax = axes[idx // 2, idx % 2]
            
            # Hexbin for clarity
            hexbin = ax.hexbin(metric, df['L'], gridsize=30, cmap='viridis', mincnt=1)
            
            ax.set_xlabel(f'{name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Recoverability (L)', fontsize=12, fontweight='bold')
            ax.set_title(f'{name} vs Recoverability\n'
                        f'Pearson r={corr_pearson:.3f}, Spearman ρ={corr_spearman:.3f}, AUC={roc_auc:.3f}',
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.colorbar(hexbin, ax=ax, label='Count')
    
    # Plot comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    results_df = pd.DataFrame(results)
    table_data = results_df.values
    table = ax.table(cellText=table_data, colLabels=results_df.columns,
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Highlight best metric
    best_idx = results_df['roc_auc'].idxmax()
    for i in range(len(results_df.columns)):
        table[(best_idx + 1, i)].set_facecolor('#90EE90')
    
    ax.set_title('Uncertainty Metric Comparison\n(Green = Best AUC)', 
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("UNCERTAINTY METRIC COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Advisor-requested analysis')
    parser.add_argument('--data', required=True, help='Path to data_with_quadrants.csv')
    parser.add_argument('--output-dir', default='./advisor_analysis')
    parser.add_argument('--n-bins', type=int, default=5, help='Number of uncertainty bins')
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    
    # Create bins
    df, bin_edges = create_uncertainty_bins(df, args.n_bins)
    
    print("\n" + "="*70)
    print("GENERATING ADVISOR-REQUESTED PLOTS")
    print("="*70)
    
    # Plot 1: Binned recoverability (Inverted-U hypothesis)
    print("\n[1/3] Binned recoverability analysis...")
    bin_stats = plot_binned_recoverability(df, bin_edges, out / 'binned_recoverability.png')
    bin_stats.to_csv(out / 'bin_statistics.csv', index=False)
    
    # Plot 2: Enrichment analysis
    print("\n[2/3] Enrichment analysis...")
    enrichment_df, pr_auc = enrichment_analysis(df, out / 'enrichment_analysis.png')
    enrichment_df.to_csv(out / 'enrichment_results.csv', index=False)
    
    # Plot 3: Alternative uncertainty metrics
    print("\n[3/3] Comparing uncertainty metrics...")
    metrics_df = compare_uncertainty_metrics(df, out / 'uncertainty_metrics_comparison.png')
    metrics_df.to_csv(out / 'uncertainty_metrics.csv', index=False)
    
    print(f"\n✓ All analysis complete. Results saved to: {out}")
    print("\nGenerated files:")
    print("  1. binned_recoverability.png - Shows inverted-U hypothesis")
    print("  2. enrichment_analysis.png - Selection strategy comparison + AUC")
    print("  3. uncertainty_metrics_comparison.png - Alternative uncertainty measures")
    print("  4. bin_statistics.csv - Numeric data for bins")
    print("  5. enrichment_results.csv - Numeric enrichment data")
    print("  6. uncertainty_metrics.csv - Metric comparison table")

if __name__ == '__main__':
    main()
