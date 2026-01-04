#!/usr/bin/env python3
"""End-to-end task completion evaluation.

Evaluates trained models on complete WebShop tasks, measuring:
- Task success rate
- Average reward
- Steps to completion
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.eval.end2end import (
    End2EndEvaluator,
    EvaluationConfig,
)


def load_models_config(phase2_dir: Path) -> list[dict[str, Any]]:
    """Load configuration for all trained models."""
    models = []
    
    # Load per-quadrant summary
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
    
    # Load baselines summary
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
    parser = argparse.ArgumentParser(description="End-to-end evaluation")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Directory with trained models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/end2end"),
        help="Output directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=100,
        help="Number of tasks to evaluate on",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum steps per task",
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
    
    # Load models configuration
    logger.info(f"Loading models from {args.models_dir}")
    all_models = load_models_config(args.models_dir)
    
    # Filter if specific models requested
    if args.models:
        all_models = [m for m in all_models if m["name"] in args.models]
    
    logger.info(f"Evaluating {len(all_models)} models")
    
    if not all_models:
        logger.error("No models found to evaluate")
        return
    
    # Create evaluator
    eval_config = EvaluationConfig(
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    evaluator = End2EndEvaluator(args.base_model, eval_config)
    
    # Evaluate each model
    results = []
    for model_info in all_models:
        logger.info(f"\nEvaluating {model_info['name']}...")
        
        try:
            result = evaluator.evaluate(model_info["path"])
            
            results.append({
                "model": model_info,
                "metrics": {
                    "success_rate": result.success_rate,
                    "avg_reward": result.avg_reward,
                    "avg_steps": result.avg_steps,
                    "n_tasks": result.n_tasks,
                },
            })
            
            logger.info(
                f"  Success: {result.success_rate:.1%}, "
                f"Reward: {result.avg_reward:.3f}, "
                f"Steps: {result.avg_steps:.1f}"
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
            "n_tasks": args.n_tasks,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "results": results,
    }
    
    save_json(output, args.output_dir / "end2end_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("END-TO-END EVALUATION RESULTS")
    print("=" * 60)
    
    # Sort by success rate
    successful = [r for r in results if "metrics" in r]
    successful.sort(key=lambda x: -x["metrics"]["success_rate"])
    
    for r in successful:
        m = r["metrics"]
        print(f"  {r['model']['name']}: {m['success_rate']:.1%} success, {m['avg_reward']:.3f} reward")
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
