#!/usr/bin/env python3
"""Evaluate cross-quadrant transfer learning.

Tests whether models trained on one quadrant generalize to others,
building a 4x4 transfer matrix.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.eval.transfer_matrix import (
    compute_transfer_matrix,
    analyze_transfer_patterns,
    generate_transfer_figure_data,
)


def load_models_config(phase2_dir: Path) -> list[dict[str, Any]]:
    """Load per-quadrant model configurations."""
    models = []
    
    pq_summary = phase2_dir / "models" / "training_summary.json"
    if pq_summary.exists():
        with open(pq_summary) as f:
            data = json.load(f)
        for result in data.get("results", []):
            models.append({
                "name": f"{result['quadrant']}_{result['supervision']}",
                "path": result["checkpoint_path"],
                "quadrant": result["quadrant"],
                "supervision": result["supervision"],
            })
    
    return models


def load_test_snapshots_by_quadrant(
    path: Path,
    n_per_quadrant: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Load test snapshots organized by quadrant."""
    with open(path) as f:
        all_snapshots = json.load(f)
    
    import random
    by_quadrant = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    
    for snap in all_snapshots:
        q = snap.get("labels", {}).get("quadrant")
        if q and q in by_quadrant:
            by_quadrant[q].append(snap)
    
    # Sample
    for q in by_quadrant:
        if len(by_quadrant[q]) > n_per_quadrant:
            by_quadrant[q] = random.sample(by_quadrant[q], n_per_quadrant)
    
    return by_quadrant


def main():
    parser = argparse.ArgumentParser(description="Transfer matrix evaluation")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Directory with trained models",
    )
    parser.add_argument(
        "--test-snapshots",
        type=Path,
        default=Path("results/phase1/labeled_snapshots.json"),
        help="Path to test snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/transfer"),
        help="Output directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--n-per-quadrant",
        type=int,
        default=50,
        help="Test snapshots per quadrant",
    )
    parser.add_argument(
        "--supervision-type",
        type=str,
        default="demo",
        help="Supervision type to evaluate (demo, contrast, hint)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    
    # Load models (filter by supervision type)
    logger.info(f"Loading models from {args.models_dir}")
    all_models = load_models_config(args.models_dir)
    
    # Filter to selected supervision type
    models = [m for m in all_models if m["supervision"] == args.supervision_type]
    logger.info(f"Using {len(models)} models with supervision={args.supervision_type}")
    
    if len(models) < 4:
        logger.warning("Need 4 quadrant models for transfer matrix")
    
    # Load test snapshots by quadrant
    logger.info(f"Loading test snapshots from {args.test_snapshots}")
    test_by_quadrant = load_test_snapshots_by_quadrant(
        args.test_snapshots, args.n_per_quadrant
    )
    
    for q, snaps in test_by_quadrant.items():
        logger.info(f"  {q}: {len(snaps)} test snapshots")
    
    # Compute transfer matrix
    logger.info("Computing transfer matrix...")
    transfer_matrix = compute_transfer_matrix(
        models=models,
        test_snapshots=test_by_quadrant,
        base_model=args.base_model,
    )
    
    # Analyze patterns
    logger.info("Analyzing transfer patterns...")
    analysis = analyze_transfer_patterns(transfer_matrix)
    
    # Generate figure data
    figure_data = generate_transfer_figure_data(transfer_matrix)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "base_model": args.base_model,
            "supervision_type": args.supervision_type,
            "n_per_quadrant": args.n_per_quadrant,
            "seed": args.seed,
        },
        "transfer_matrix": transfer_matrix.to_dict(),
        "analysis": analysis,
        "figure_data": figure_data,
    }
    
    save_json(output, args.output_dir / "transfer_results.json")
    
    # Print transfer matrix
    print("\n" + "=" * 60)
    print(f"TRANSFER MATRIX (supervision={args.supervision_type})")
    print("=" * 60)
    print("\nRows = trained on, Columns = tested on")
    print()
    
    matrix = transfer_matrix.matrix
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    
    # Header
    print(f"{'Train\\Test':<10}", end="")
    for q in quadrants:
        print(f"{q:<10}", end="")
    print()
    print("-" * 50)
    
    # Matrix rows
    for train_q in quadrants:
        print(f"{train_q:<10}", end="")
        for test_q in quadrants:
            val = matrix.get(train_q, {}).get(test_q, 0)
            print(f"{val:.3f}     ", end="")
        print()
    
    print("\n" + "-" * 50)
    print("ANALYSIS")
    print("-" * 50)
    print(f"Diagonal mean (in-quadrant): {analysis.get('diagonal_mean', 0):.3f}")
    print(f"Off-diagonal mean (transfer): {analysis.get('off_diagonal_mean', 0):.3f}")
    print(f"Transfer gap: {analysis.get('transfer_gap', 0):.3f}")
    print(f"Best transfer pair: {analysis.get('best_transfer', 'N/A')}")
    print(f"Worst transfer pair: {analysis.get('worst_transfer', 'N/A')}")
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
