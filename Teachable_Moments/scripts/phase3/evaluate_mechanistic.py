#!/usr/bin/env python3
"""Mechanistic evaluation of specific behaviors.

Evaluates whether models learned specific skills:
- Action selection accuracy
- State understanding
- Goal tracking
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.eval.mechanistic import (
    MechanisticEvaluator,
    MechanisticConfig,
)


def load_models_config(phase2_dir: Path) -> list[dict[str, Any]]:
    """Load configuration for all trained models."""
    models = []
    
    # Per-quadrant models
    pq_summary = phase2_dir / "models" / "training_summary.json"
    if pq_summary.exists():
        with open(pq_summary) as f:
            data = json.load(f)
        for result in data.get("results", []):
            models.append({
                "name": f"{result['quadrant']}_{result['supervision']}",
                "path": result["checkpoint_path"],
                "type": "per_quadrant",
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


def load_test_snapshots(path: Path, n_per_quadrant: int = 25) -> list[dict[str, Any]]:
    """Load test snapshots for mechanistic evaluation."""
    with open(path) as f:
        all_snapshots = json.load(f)
    
    # Sample stratified by quadrant
    by_quadrant = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for snap in all_snapshots:
        q = snap.get("labels", {}).get("quadrant")
        if q and q in by_quadrant:
            by_quadrant[q].append(snap)
    
    import random
    selected = []
    for q, snaps in by_quadrant.items():
        if len(snaps) <= n_per_quadrant:
            selected.extend(snaps)
        else:
            selected.extend(random.sample(snaps, n_per_quadrant))
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="Mechanistic evaluation")
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
        default=Path("results/phase3/mechanistic"),
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
        help="Test snapshots per quadrant",
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
    
    # Load models
    logger.info(f"Loading models from {args.models_dir}")
    all_models = load_models_config(args.models_dir)
    
    if args.models:
        all_models = [m for m in all_models if m["name"] in args.models]
    
    logger.info(f"Evaluating {len(all_models)} models")
    
    # Load test snapshots
    logger.info(f"Loading test snapshots from {args.test_snapshots}")
    test_snapshots = load_test_snapshots(args.test_snapshots, args.n_per_quadrant)
    logger.info(f"Loaded {len(test_snapshots)} test snapshots")
    
    # Create evaluator
    eval_config = MechanisticConfig(
        compute_action_accuracy=True,
        compute_state_understanding=True,
        compute_goal_tracking=True,
    )
    
    evaluator = MechanisticEvaluator(args.base_model, eval_config)
    
    # Evaluate each model
    results = []
    for model_info in all_models:
        logger.info(f"\nEvaluating {model_info['name']}...")
        
        try:
            result = evaluator.evaluate(model_info["path"], test_snapshots)
            
            results.append({
                "model": model_info,
                "metrics": {
                    "action_accuracy": result.action_accuracy,
                    "state_understanding": result.state_understanding,
                    "goal_tracking": result.goal_tracking,
                    "overall_score": result.overall_score,
                },
                "by_quadrant": result.by_quadrant,
            })
            
            logger.info(
                f"  Action: {result.action_accuracy:.1%}, "
                f"State: {result.state_understanding:.1%}, "
                f"Goal: {result.goal_tracking:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {model_info['name']}: {e}")
            results.append({
                "model": model_info,
                "error": str(e),
            })
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "base_model": args.base_model,
            "n_per_quadrant": args.n_per_quadrant,
            "n_test_snapshots": len(test_snapshots),
            "seed": args.seed,
        },
        "results": results,
    }
    
    save_json(output, args.output_dir / "mechanistic_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MECHANISTIC EVALUATION RESULTS")
    print("=" * 60)
    
    successful = [r for r in results if "metrics" in r]
    successful.sort(key=lambda x: -x["metrics"]["overall_score"])
    
    print(f"{'Model':<30} {'Action':<10} {'State':<10} {'Goal':<10} {'Overall':<10}")
    print("-" * 70)
    
    for r in successful:
        m = r["metrics"]
        print(
            f"{r['model']['name']:<30} "
            f"{m['action_accuracy']:.1%}     "
            f"{m['state_understanding']:.1%}     "
            f"{m['goal_tracking']:.1%}     "
            f"{m['overall_score']:.1%}"
        )
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
