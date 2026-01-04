#!/usr/bin/env python3
"""
Multi-Trial Comparison: Baseline vs Entropy

Runs multiple trials of both strategies and computes:
- Mean ± Std for each metric
- 95% Confidence Intervals
- Statistical significance (t-test)

Usage:
    python multi_trial_comparison.py --num_trials 5 --budget 750
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from scipy import stats
from datetime import datetime


def run_trial(strategy, trial_num, args):
    """Run a single trial and return results"""
    output_dir = f"./eef_trial_{strategy}_{trial_num}"
    
    cmd = [
        "python", "eef_complete_updated.py",
        "--failure_data", args.failure_data,
        "--strategy", strategy,
        "--M", str(args.M),
        "--num_trajectories", str(args.num_trajectories),
        "--simulation_budget", str(args.budget),
        "--output_dir", output_dir,
    ]
    
    print(f"  Running {strategy} trial {trial_num}...")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:200]}")
        return None
    
    # Find and load stats file
    stats_files = [f for f in os.listdir(output_dir) if f.startswith("stats_")]
    if not stats_files:
        print(f"    ERROR: No stats file found")
        return None
    
    with open(os.path.join(output_dir, stats_files[0])) as f:
        trial_stats = json.load(f)
    
    return {
        'full_success': trial_stats.get('full_success_count', 0),
        'improvements': trial_stats.get('improvement_count', 0),
        'total_beneficial': trial_stats.get('total_beneficial', 0),
        'simulations': trial_stats.get('simulations_run', 0),
        'recovery_rate': trial_stats.get('recovery_rate', 0) * 100,
    }


def compute_stats(values):
    """Compute mean, std, and 95% CI"""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    ci_95 = 1.96 * se
    return {
        'mean': mean,
        'std': std,
        'ci_95': ci_95,
        'min': np.min(values),
        'max': np.max(values),
    }


def run_ttest(baseline_vals, entropy_vals):
    """Run independent t-test"""
    t_stat, p_value = stats.ttest_ind(entropy_vals, baseline_vals)
    return t_stat, p_value


def main():
    parser = argparse.ArgumentParser(description="Multi-Trial Comparison")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--budget", type=int, default=750)
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--failure_data", type=str, default="./failure_trajectories.json")
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-TRIAL COMPARISON: BASELINE vs ENTROPY")
    print("="*70)
    print(f"  Trials: {args.num_trials}")
    print(f"  Budget per trial: {args.budget}")
    print(f"  Trajectories: {args.num_trajectories}")
    print("="*70)
    
    # Run trials
    baseline_results = []
    entropy_results = []
    
    for trial in range(1, args.num_trials + 1):
        print(f"\n--- Trial {trial}/{args.num_trials} ---")
        
        # Baseline
        b_result = run_trial("baseline", trial, args)
        if b_result:
            baseline_results.append(b_result)
            print(f"    Baseline: {b_result['full_success']} success, {b_result['total_beneficial']} beneficial")
        
        # Entropy
        e_result = run_trial("entropy", trial, args)
        if e_result:
            entropy_results.append(e_result)
            print(f"    Entropy:  {e_result['full_success']} success, {e_result['total_beneficial']} beneficial")
    
    if len(baseline_results) < 2 or len(entropy_results) < 2:
        print("\nERROR: Not enough successful trials for statistical analysis")
        return
    
    # Compute statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    metrics = ['full_success', 'improvements', 'total_beneficial', 'recovery_rate']
    metric_names = ['Full Success (r=100)', 'Improvements', 'Total Beneficial', 'Recovery Rate (%)']
    
    results_summary = {
        'baseline': {},
        'entropy': {},
        'comparison': {},
    }
    
    print(f"\n{'Metric':<25} {'Baseline':>20} {'Entropy':>20} {'p-value':>10}")
    print("-"*75)
    
    for metric, name in zip(metrics, metric_names):
        b_vals = [r[metric] for r in baseline_results]
        e_vals = [r[metric] for r in entropy_results]
        
        b_stats = compute_stats(b_vals)
        e_stats = compute_stats(e_vals)
        
        t_stat, p_value = run_ttest(b_vals, e_vals)
        
        # Store results
        results_summary['baseline'][metric] = b_stats
        results_summary['entropy'][metric] = e_stats
        results_summary['comparison'][metric] = {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
        
        # Format for display
        b_str = f"{b_stats['mean']:.1f} ± {b_stats['std']:.1f}"
        e_str = f"{e_stats['mean']:.1f} ± {e_stats['std']:.1f}"
        sig = "*" if p_value < 0.05 else ""
        
        print(f"{name:<25} {b_str:>20} {e_str:>20} {p_value:>9.4f}{sig}")
    
    print("-"*75)
    print("* p < 0.05 (statistically significant)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    b_success = results_summary['baseline']['full_success']['mean']
    e_success = results_summary['entropy']['full_success']['mean']
    success_diff = (e_success - b_success) / b_success * 100 if b_success > 0 else 0
    
    b_beneficial = results_summary['baseline']['total_beneficial']['mean']
    e_beneficial = results_summary['entropy']['total_beneficial']['mean']
    beneficial_diff = (e_beneficial - b_beneficial) / b_beneficial * 100 if b_beneficial > 0 else 0
    
    print(f"\nFull Success:     Entropy {e_success:.1f} vs Baseline {b_success:.1f} ({success_diff:+.1f}%)")
    print(f"Total Beneficial: Entropy {e_beneficial:.1f} vs Baseline {b_beneficial:.1f} ({beneficial_diff:+.1f}%)")
    
    p_success = results_summary['comparison']['full_success']['p_value']
    p_beneficial = results_summary['comparison']['total_beneficial']['p_value']
    
    print(f"\nStatistical Significance:")
    print(f"  Full Success:     p = {p_success:.4f} {'✓ SIGNIFICANT' if p_success < 0.05 else '✗ not significant'}")
    print(f"  Total Beneficial: p = {p_beneficial:.4f} {'✓ SIGNIFICANT' if p_beneficial < 0.05 else '✗ not significant'}")
    
    # Winner
    print("\n" + "="*70)
    if e_success > b_success and p_success < 0.05:
        print("🎉 ENTROPY WINS with statistical significance!")
    elif e_success > b_success:
        print("📈 ENTROPY appears better but not statistically significant")
        print("   Consider running more trials")
    elif b_success > e_success and p_success < 0.05:
        print("📉 BASELINE wins with statistical significance")
    else:
        print("🤷 No clear winner - results are within noise")
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./comparison_results_{timestamp}.json"
    
    full_results = {
        'config': {
            'num_trials': args.num_trials,
            'budget': args.budget,
            'num_trajectories': args.num_trajectories,
            'M': args.M,
        },
        'raw_results': {
            'baseline': baseline_results,
            'entropy': entropy_results,
        },
        'statistics': results_summary,
    }
    
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n✓ Full results saved to {output_file}")


if __name__ == "__main__":
    main()