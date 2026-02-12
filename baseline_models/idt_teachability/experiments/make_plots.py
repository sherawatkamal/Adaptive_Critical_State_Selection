#!/usr/bin/env python3
"""
Generate matplotlib plots for IDT experiments.

Saves to baseline_models/idt_teachability/outputs/plots/
"""

from __future__ import annotations

import os
import json
import glob
from typing import List, Dict, Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import numpy as np


def _find_latest_stats(output_dir: str, pattern: str, prefer_n_traj: int | None = None) -> List[Dict]:
    """Find patch_stats files matching pattern. If prefer_n_traj is set, prefer files with that N_traj_total."""
    base = os.path.join(output_dir, f"patch_stats_{pattern}*.json")
    files = sorted(glob.glob(base), key=os.path.getmtime, reverse=True)
    out = []
    for f in files:
        try:
            with open(f) as fp:
                obj = json.load(fp)
                out.append(obj)
        except Exception:
            pass
    if prefer_n_traj is not None and out:
        matching = [o for o in out if o.get("coverage", {}).get("N_traj_total") == prefer_n_traj]
        if matching:
            return [matching[0]]
    return out


def _find_latest_results(output_dir: str, pattern: str, prefer_stats_n_traj: int | None = None):
    """Find patch_results. If prefer_stats_n_traj set, use results matching stats file with that N_traj."""
    if prefer_stats_n_traj is not None:
        stats_files = sorted(glob.glob(os.path.join(output_dir, f"patch_stats_{pattern}*.json")), key=os.path.getmtime, reverse=True)
        for sf in stats_files:
            try:
                with open(sf) as fp:
                    s = json.load(fp)
                if s.get("coverage", {}).get("N_traj_total") == prefer_stats_n_traj:
                    base = os.path.basename(sf).replace("patch_stats_", "patch_results_")
                    rf = os.path.join(output_dir, base)
                    if os.path.exists(rf):
                        with open(rf) as fp:
                            return json.load(fp)
            except Exception:
                pass
    base = os.path.join(output_dir, f"patch_results_{pattern}*.json")
    files = sorted(glob.glob(base), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f) as fp:
                return json.load(fp)
        except Exception:
            pass
    return []


def plot_patch_k_scaling(output_dir: str, plots_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    """Success/rescue/break vs patch_k (EXP4, 500 trajectories)."""
    stats_by_k = {}
    for k in [1, 2, 5, 10]:
        lst = _find_latest_stats(output_dir, f"EXP_EXP4_k{k}", prefer_n_traj=500)
        if not lst:
            lst = _find_latest_stats(output_dir, f"EXP_EXP4_k{k}")
        if lst:
            stats_by_k[k] = lst[0]

    if not stats_by_k:
        return

    ks = sorted(stats_by_k.keys())
    success = [stats_by_k[k].get("patched_success_rate", 0) for k in ks]
    rescue = [stats_by_k[k].get("rescue_rate", 0) for k in ks]
    break_r = [stats_by_k[k].get("break_rate", 0) for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, success, "o-", color="#2ecc71", linewidth=2, markersize=8, label="Patched success")
    ax.plot(ks, rescue, "s-", color="#3498db", linewidth=2, markersize=8, label="Rescue rate")
    ax.plot(ks, break_r, "^-", color="#e74c3c", linewidth=2, markersize=8, label="Break rate")
    ax.set_xlabel("patch_k")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.set_title("IDT: Success/Rescue/Break vs patch_k (EXP4, N=500)")
    ax.set_ylim(0, max(max(success + rescue + break_r) * 1.15, 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "plot_patch_k_scaling.png"), dpi=100)
    plt.close()


def plot_compute_matched(output_dir: str, plots_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    """Compute-matched baseline vs IDT success (EXP1/EXP2, 500 trajectories)."""
    exp1 = _find_latest_stats(output_dir, "EXP_EXP1_best_of", prefer_n_traj=500)
    if not exp1:
        exp1 = _find_latest_stats(output_dir, "EXP_EXP1_best_of")
    exp2 = _find_latest_stats(output_dir, "EXP_EXP2_compute_matched", prefer_n_traj=500)
    if not exp2:
        exp2 = _find_latest_stats(output_dir, "EXP_EXP2_compute_matched")
    if not exp1 or not exp2:
        return
    s1, s2 = exp1[0], exp2[0]
    methods = ["IDT best-of", "Compute-matched baseline"]
    success_rates = [s1.get("patched_success_rate", 0), s2.get("baseline_success_rate", 0)]
    colors = ["#2ecc71", "#3498db"]
    fig, ax = plt.subplots(figsize=(6, 3))
    y_pos = np.arange(len(methods))
    ax.hlines(y_pos, 0, success_rates, colors=colors, linewidth=8, alpha=0.8)
    ax.scatter(success_rates, y_pos, color=colors, s=120, zorder=5, edgecolors="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Success rate")
    ax.set_title("IDT vs Compute-Matched Baseline (N=500)")
    ax.set_xlim(0, max(success_rates) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "plot_compute_matched.png"), dpi=100)
    plt.close()


def plot_selector_comparison(output_dir: str, plots_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    """Bar chart: success/rescue/break by step selector (EXP5, N=100)."""
    strategies = ["diagnosis", "last_n", "search_steps", "random_steps", "baseline"]
    data = {}
    for s in strategies:
        lst = _find_latest_stats(output_dir, f"EXP_EXP5_{s}", prefer_n_traj=100)
        if not lst:
            lst = _find_latest_stats(output_dir, f"EXP_EXP5_{s}")
        if lst:
            data[s] = lst[0]

    if not data:
        return

    labels = list(data.keys())
    success = [data[s].get("patched_success_rate", 0) for s in labels]
    rescue = [data[s].get("rescue_rate", 0) for s in labels]
    break_r = [data[s].get("break_rate", 0) for s in labels]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y - 0.25, success, 0.25, label="Patched success", color="#2ecc71")
    ax.barh(y, rescue, 0.25, label="Rescue rate", color="#3498db")
    ax.barh(y + 0.25, break_r, 0.25, label="Break rate", color="#e74c3c")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rate")
    ax.legend(loc="lower right")
    ax.set_title("Step selector comparison (EXP5, N=100)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "plot_selector_comparison.png"), dpi=100)
    plt.close()


def plot_delta_histogram(output_dir: str, plots_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    """Histogram of Δ reward for EXP1 best-of (500 trajectories)."""
    records = _find_latest_results(output_dir, "EXP_EXP1_best_of", prefer_stats_n_traj=500)
    if not records:
        records = _find_latest_results(output_dir, "EXP_EXP1_best_of")
    if not records:
        return
    deltas = np.array([r.get("improvement", 0) for r in records if r.get("replay_ok") and r.get("improvement") is not None])
    if len(deltas) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.violinplot([deltas], positions=[0], widths=0.7, showmeans=True, showmedians=True)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xticks([0])
    ax.set_xticklabels(["Δ reward"])
    ax.set_ylabel("Δ reward (patched - baseline)")
    ax.set_title("Distribution of improvement (EXP1 best-of, N=500)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "plot_delta_histogram.png"), dpi=100)
    plt.close()


def plot_experiment_summary(output_dir: str, plots_dir: str) -> None:
    """Overview of all experiments: baseline vs patched success by experiment (EXP1-5)."""
    if not HAS_MATPLOTLIB:
        return
    experiments = [
        ("EXP1 (IDT best-of)", "EXP_EXP1_best_of", 500),
        ("EXP2 (compute-matched)", "EXP_EXP2_compute_matched", 500),
        ("EXP3 (random patch)", "EXP_EXP3_random_patch", 500),
        ("EXP4 k=5", "EXP_EXP4_k5", 500),
    ]
    exp5_strategies = ["diagnosis", "last_n", "search_steps", "random_steps"]
    data = []
    labels = []
    for label, pattern, n in experiments:
        lst = _find_latest_stats(output_dir, pattern, prefer_n_traj=n)
        if not lst:
            lst = _find_latest_stats(output_dir, pattern)
        if lst:
            s = lst[0]
            data.append((s.get("baseline_success_rate", 0), s.get("patched_success_rate", 0)))
            labels.append(label)
    for strat in exp5_strategies:
        lst = _find_latest_stats(output_dir, f"EXP_EXP5_{strat}", prefer_n_traj=100)
        if not lst:
            lst = _find_latest_stats(output_dir, f"EXP_EXP5_{strat}")
        if lst:
            s = lst[0]
            data.append((s.get("baseline_success_rate", 0), s.get("patched_success_rate", 0)))
            labels.append(f"EXP5 {strat}")
    if not data:
        return
    baseline_rates = np.array([d[0] for d in data])
    patched_rates = np.array([d[1] for d in data])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(baseline_rates, patched_rates, s=80, alpha=0.8, c="#2ecc71", edgecolors="black")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (baseline_rates[i], patched_rates[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    lim = max(max(baseline_rates), max(patched_rates)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x (no change)")
    ax.set_xlabel("Baseline success rate")
    ax.set_ylabel("Patched success rate")
    ax.set_title("IDT experiments: Baseline vs Patched")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "plot_experiment_summary.png"), dpi=100)
    plt.close()


def plot_teachable_step_dist(output_dir: str, plots_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    """Distribution of t* (teachable moment step index) from EXP6."""
    path = os.path.join(output_dir, "EXP6_trajectory_discovery_stats.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        stats = json.load(f)
    t_star = stats.get("t_star_distribution", [])
    if not t_star:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    counts, bins, _ = ax.hist(t_star, bins=min(30, max(len(set(t_star)), 5)), alpha=0.5, color="#9b59b6")
    ax.fill_between((bins[:-1] + bins[1:]) / 2, counts, alpha=0.4, color="#9b59b6")
    ax.set_xlabel("Step index t*")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of teachable moment step (EXP6)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "plot_teachable_step_dist.png"), dpi=100)
    plt.close()


def make_all_plots(output_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not installed)")
        return
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_experiment_summary(output_dir, plots_dir)
    plot_patch_k_scaling(output_dir, plots_dir)
    plot_compute_matched(output_dir, plots_dir)
    plot_selector_comparison(output_dir, plots_dir)
    plot_delta_histogram(output_dir, plots_dir)
    plot_teachable_step_dist(output_dir, plots_dir)
    print(f"  Plots saved to {plots_dir}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "outputs")
    make_all_plots(out)
