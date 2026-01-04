#!/usr/bin/env python3
"""Evaluate distribution drift using fixed panels.

Tests whether model performance remains stable across evaluation runs
using fixed evaluation panels.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.eval.drift_panel import (
    EvaluationPanel,
    create_evaluation_panel,
    evaluate_on_panel,
    detect_drift,
)


def load_or_create_panel(
    panel_path: Path,
    snapshots_path: Path,
    n_per_quadrant: int = 25,
) -> EvaluationPanel:
    """Load existing panel or create new one."""
    if panel_path.exists():
        with open(panel_path) as f:
            data = json.load(f)
        return EvaluationPanel.from_dict(data)
    
    # Create new panel from labeled snapshots
    with open(snapshots_path) as f:
        snapshots = json.load(f)
    
    panel = create_evaluation_panel(
        snapshots=snapshots,
        n_per_quadrant=n_per_quadrant,
        stratify_by=["quadrant", "depth"],
    )
    
    # Save for future use
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(panel_path, "w") as f:
        json.dump(panel.to_dict(), f, indent=2)
    
    return panel


def load_models_config(phase2_dir: Path) -> list[dict[str, Any]]:
    """Load model configurations."""
    models = []
    
    # Per-quadrant
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
    
    # Baselines
    bl_summary = phase2_dir / "baselines" / "baseline_summary.json"
    if bl_summary.exists():
        with open(bl_summary) as f:
            data = json.load(f)
        for result in data.get("results", []):
            models.append({
                "name": f"baseline_{result['name']}",
                "path": result["checkpoint_path"],
                "type": "baseline",
            })
    
    return models


def main():
    parser = argparse.ArgumentParser(description="Drift evaluation")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Directory with trained models",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("panels/drift_panel.json"),
        help="Path to evaluation panel",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=Path("results/phase1/labeled_snapshots.json"),
        help="Path to labeled snapshots (for panel creation)",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="Path to baseline results for drift comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/drift"),
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
        default=25,
        help="Panel tasks per quadrant",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of evaluation runs per model",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="Threshold for drift detection",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to evaluate",
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
    
    # Load or create evaluation panel
    logger.info("Loading evaluation panel...")
    panel = load_or_create_panel(args.panel, args.snapshots, args.n_per_quadrant)
    logger.info(f"Panel size: {len(panel.tasks)}")
    
    # Load models
    logger.info(f"Loading models from {args.models_dir}")
    all_models = load_models_config(args.models_dir)
    
    if args.models:
        all_models = [m for m in all_models if m["name"] in args.models]
    
    logger.info(f"Evaluating {len(all_models)} models")
    
    # Load baseline results if provided
    baseline_results = None
    if args.baseline_results and args.baseline_results.exists():
        with open(args.baseline_results) as f:
            baseline_results = json.load(f)
    
    # Evaluate each model multiple times
    all_results = []
    
    for model_info in all_models:
        logger.info(f"\nEvaluating {model_info['name']}...")
        
        model_runs = []
        
        for run_idx in range(args.n_runs):
            try:
                result = evaluate_on_panel(
                    model_path=model_info["path"],
                    panel=panel,
                    base_model=args.base_model,
                    seed=args.seed + run_idx,
                )
                
                model_runs.append({
                    "run": run_idx,
                    "accuracy": result.accuracy,
                    "avg_confidence": result.avg_confidence,
                    "by_quadrant": result.by_quadrant,
                })
                
                logger.info(f"  Run {run_idx + 1}: accuracy={result.accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error in run {run_idx}: {e}")
        
        # Detect drift
        if len(model_runs) >= 2:
            drift = detect_drift(
                model_runs,
                threshold=args.drift_threshold,
                baseline=baseline_results.get(model_info["name"]) if baseline_results else None,
            )
        else:
            drift = {"detected": False, "reason": "insufficient_runs"}
        
        all_results.append({
            "model": model_info,
            "runs": model_runs,
            "mean_accuracy": sum(r["accuracy"] for r in model_runs) / len(model_runs) if model_runs else 0,
            "std_accuracy": (
                (sum((r["accuracy"] - sum(r["accuracy"] for r in model_runs) / len(model_runs)) ** 2 
                     for r in model_runs) / len(model_runs)) ** 0.5
                if len(model_runs) > 1 else 0
            ),
            "drift": drift,
        })
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "base_model": args.base_model,
            "n_runs": args.n_runs,
            "drift_threshold": args.drift_threshold,
            "panel_size": len(panel.tasks),
            "seed": args.seed,
        },
        "panel_info": {
            "path": str(args.panel),
            "n_tasks": len(panel.tasks),
        },
        "results": all_results,
    }
    
    save_json(output, args.output_dir / "drift_results.json")
    
    # Save current results as potential baseline for future comparisons
    baseline_output = {
        model_result["model"]["name"]: model_result["mean_accuracy"]
        for model_result in all_results
    }
    save_json(baseline_output, args.output_dir / "baseline_for_comparison.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DRIFT EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Model':<30} {'Mean Acc':<12} {'Std':<12} {'Drift':<12}")
    print("-" * 66)
    
    for r in all_results:
        drift_status = "YES" if r["drift"]["detected"] else "No"
        print(
            f"{r['model']['name']:<30} "
            f"{r['mean_accuracy']:.3f}        "
            f"{r['std_accuracy']:.4f}       "
            f"{drift_status}"
        )
    
    # Summary
    n_drift = sum(1 for r in all_results if r["drift"]["detected"])
    print(f"\nModels with detected drift: {n_drift}/{len(all_results)}")
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
