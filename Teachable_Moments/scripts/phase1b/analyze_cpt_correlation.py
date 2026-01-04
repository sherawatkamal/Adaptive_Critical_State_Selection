#!/usr/bin/env python3
"""Analyze correlation between CPT predictions and actual ELP.

This script computes the correlation between Counterfactual Probability of Teaching (CPT)
and Empirical Learning Potential (ELP) to validate that CPT is a reliable proxy.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from src.utils import setup_logging, save_json


def load_results(path: Path) -> dict[str, Any]:
    """Load micro-training results."""
    with open(path) as f:
        return json.load(f)


def compute_correlation_metrics(
    cpt_values: list[float],
    elp_values: list[float],
) -> dict[str, float]:
    """Compute various correlation metrics."""
    cpt = np.array(cpt_values)
    elp = np.array(elp_values)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(cpt, elp)
    
    # Spearman correlation (rank-based)
    spearman_r, spearman_p = stats.spearmanr(cpt, elp)
    
    # Kendall's tau
    kendall_tau, kendall_p = stats.kendalltau(cpt, elp)
    
    # Mean absolute error (treating CPT as predictor of ELP)
    mae = np.mean(np.abs(cpt - elp))
    
    # Root mean squared error
    rmse = np.sqrt(np.mean((cpt - elp) ** 2))
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((elp - cpt) ** 2)
    ss_tot = np.sum((elp - np.mean(elp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "mae": float(mae),
        "rmse": float(rmse),
        "r_squared": float(r_squared),
        "n_samples": len(cpt_values),
    }


def analyze_by_quadrant(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute correlations separately for each quadrant."""
    by_quadrant = {}
    
    for r in results:
        q = r.get("quadrant", "unknown")
        if q not in by_quadrant:
            by_quadrant[q] = {"cpt": [], "elp": []}
        by_quadrant[q]["cpt"].append(r["cpt_delta"])
        by_quadrant[q]["elp"].append(r["elp"])
    
    quadrant_correlations = {}
    for q, data in by_quadrant.items():
        if len(data["cpt"]) >= 5:  # Need minimum samples
            quadrant_correlations[q] = compute_correlation_metrics(
                data["cpt"], data["elp"]
            )
        else:
            quadrant_correlations[q] = {"n_samples": len(data["cpt"]), "insufficient_data": True}
    
    return quadrant_correlations


def analyze_calibration(
    cpt_values: list[float],
    elp_values: list[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Analyze calibration: do CPT predictions match ELP on average?"""
    cpt = np.array(cpt_values)
    elp = np.array(elp_values)
    
    # Bin by CPT values
    bins = np.linspace(cpt.min(), cpt.max(), n_bins + 1)
    bin_indices = np.digitize(cpt, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            calibration.append({
                "bin_center": float((bins[i] + bins[i+1]) / 2),
                "avg_cpt": float(cpt[mask].mean()),
                "avg_elp": float(elp[mask].mean()),
                "std_elp": float(elp[mask].std()),
                "n_samples": int(mask.sum()),
            })
    
    # Expected calibration error
    ece = 0.0
    total = len(cpt)
    for cal in calibration:
        ece += (cal["n_samples"] / total) * abs(cal["avg_cpt"] - cal["avg_elp"])
    
    return {
        "bins": calibration,
        "expected_calibration_error": float(ece),
    }


def identify_outliers(
    results: list[dict[str, Any]],
    threshold_std: float = 2.0,
) -> list[dict[str, Any]]:
    """Identify snapshots where CPT and ELP diverge significantly."""
    cpt_values = [r["cpt_delta"] for r in results]
    elp_values = [r["elp"] for r in results]
    
    differences = [abs(c - e) for c, e in zip(cpt_values, elp_values)]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    outliers = []
    for r, diff in zip(results, differences):
        if diff > mean_diff + threshold_std * std_diff:
            outliers.append({
                "snapshot_id": r["snapshot_id"],
                "quadrant": r.get("quadrant"),
                "cpt_delta": r["cpt_delta"],
                "elp": r["elp"],
                "difference": diff,
                "z_score": (diff - mean_diff) / std_diff if std_diff > 0 else 0,
            })
    
    return sorted(outliers, key=lambda x: -x["difference"])


def generate_report(
    overall: dict[str, float],
    by_quadrant: dict[str, dict[str, float]],
    calibration: dict[str, Any],
    outliers: list[dict[str, Any]],
) -> str:
    """Generate human-readable analysis report."""
    lines = [
        "=" * 60,
        "CPT vs ELP CORRELATION ANALYSIS",
        "=" * 60,
        "",
        "OVERALL CORRELATION",
        "-" * 40,
        f"  Samples: {overall['n_samples']}",
        f"  Pearson r: {overall['pearson_r']:.4f} (p={overall['pearson_p']:.4e})",
        f"  Spearman r: {overall['spearman_r']:.4f} (p={overall['spearman_p']:.4e})",
        f"  Kendall tau: {overall['kendall_tau']:.4f} (p={overall['kendall_p']:.4e})",
        f"  MAE: {overall['mae']:.4f}",
        f"  RMSE: {overall['rmse']:.4f}",
        f"  R-squared: {overall['r_squared']:.4f}",
        "",
        "INTERPRETATION",
        "-" * 40,
    ]
    
    r = overall['pearson_r']
    if r >= 0.7:
        interp = "Strong positive correlation - CPT is a reliable proxy for ELP"
    elif r >= 0.5:
        interp = "Moderate positive correlation - CPT captures most ELP signal"
    elif r >= 0.3:
        interp = "Weak positive correlation - CPT provides some signal"
    else:
        interp = "Weak or no correlation - CPT may not be reliable"
    lines.append(f"  {interp}")
    
    lines.extend([
        "",
        "CORRELATION BY QUADRANT",
        "-" * 40,
    ])
    
    for q in sorted(by_quadrant.keys()):
        q_data = by_quadrant[q]
        if q_data.get("insufficient_data"):
            lines.append(f"  {q}: insufficient data (n={q_data['n_samples']})")
        else:
            lines.append(
                f"  {q}: r={q_data['pearson_r']:.3f}, "
                f"n={q_data['n_samples']}"
            )
    
    lines.extend([
        "",
        "CALIBRATION",
        "-" * 40,
        f"  Expected Calibration Error: {calibration['expected_calibration_error']:.4f}",
        "",
    ])
    
    if outliers:
        lines.extend([
            "OUTLIERS (CPT-ELP mismatch)",
            "-" * 40,
        ])
        for o in outliers[:5]:  # Top 5
            lines.append(
                f"  {o['snapshot_id']}: CPT={o['cpt_delta']:.3f}, "
                f"ELP={o['elp']:.3f}, diff={o['difference']:.3f}"
            )
    
    lines.extend(["", "=" * 60])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze CPT-ELP correlation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase1b/micro_training_results.json"),
        help="Path to micro-training results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1b"),
        help="Output directory",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=2.0,
        help="Standard deviations for outlier detection",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load results
    logger.info(f"Loading results from {args.input}")
    data = load_results(args.input)
    results = data.get("results", [])
    logger.info(f"Loaded {len(results)} experiment results")
    
    if len(results) < 5:
        logger.error("Insufficient data for correlation analysis")
        return
    
    # Extract CPT and ELP values
    cpt_values = [r["cpt_delta"] for r in results]
    elp_values = [r["elp"] for r in results]
    
    # Compute overall correlation
    logger.info("Computing overall correlation...")
    overall = compute_correlation_metrics(cpt_values, elp_values)
    
    # Analyze by quadrant
    logger.info("Analyzing by quadrant...")
    by_quadrant = analyze_by_quadrant(results)
    
    # Calibration analysis
    logger.info("Analyzing calibration...")
    calibration = analyze_calibration(cpt_values, elp_values)
    
    # Identify outliers
    logger.info("Identifying outliers...")
    outliers = identify_outliers(results, args.outlier_threshold)
    
    # Generate report
    report = generate_report(overall, by_quadrant, calibration, outliers)
    
    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis = {
        "overall_correlation": overall,
        "correlation_by_quadrant": by_quadrant,
        "calibration": calibration,
        "outliers": outliers,
        "validation_passed": overall["pearson_r"] >= 0.5 and overall["pearson_p"] < 0.05,
    }
    
    save_json(analysis, args.output_dir / "cpt_correlation_analysis.json")
    
    report_path = args.output_dir / "cpt_correlation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(report)
    
    # Print validation result
    if analysis["validation_passed"]:
        print("\n[PASS] CPT is a valid proxy for ELP")
    else:
        print("\n[WARN] CPT correlation with ELP is weak - review methodology")


if __name__ == "__main__":
    main()
