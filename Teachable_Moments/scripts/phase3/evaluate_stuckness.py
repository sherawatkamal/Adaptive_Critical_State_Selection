#!/usr/bin/env python3
"""Evaluate recovery from stuck states.

Tests whether trained models can recover from common failure patterns:
- Loop detection
- Action repetition
- Progress stalls
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.eval.stuckness import (
    StucknessEvaluator,
    StucknessConfig,
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


def load_stuck_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load or create stuck scenarios for testing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    
    # Default stuck scenarios
    return [
        {
            "type": "action_loop",
            "description": "Agent repeating the same action",
            "history": [
                {"action": "click[search]", "observation": "No results"},
                {"action": "click[search]", "observation": "No results"},
                {"action": "click[search]", "observation": "No results"},
            ],
            "expected_behavior": "Try different action",
        },
        {
            "type": "wrong_page",
            "description": "Agent on irrelevant page",
            "history": [
                {"action": "click[Furniture]", "observation": "Furniture page"},
            ],
            "task": "Buy laptop",
            "expected_behavior": "Navigate to electronics",
        },
        {
            "type": "no_progress",
            "description": "Multiple actions with no task progress",
            "history": [
                {"action": "scroll down", "observation": "More products"},
                {"action": "scroll down", "observation": "More products"},
                {"action": "scroll up", "observation": "More products"},
            ],
            "expected_behavior": "Take decisive action",
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Stuckness evaluation")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Directory with trained models",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("panels/stuck_scenarios.json"),
        help="Path to stuck scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/stuckness"),
        help="Output directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name",
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
    
    # Load stuck scenarios
    logger.info(f"Loading stuck scenarios")
    scenarios = load_stuck_scenarios(args.scenarios)
    logger.info(f"Loaded {len(scenarios)} scenarios")
    
    # Create evaluator
    eval_config = StucknessConfig(
        max_recovery_steps=5,
        loop_detection_window=3,
    )
    
    evaluator = StucknessEvaluator(args.base_model, eval_config)
    
    # Evaluate each model
    results = []
    for model_info in all_models:
        logger.info(f"\nEvaluating {model_info['name']}...")
        
        try:
            result = evaluator.evaluate(model_info["path"], scenarios)
            
            results.append({
                "model": model_info,
                "metrics": {
                    "recovery_rate": result.recovery_rate,
                    "avg_recovery_steps": result.avg_recovery_steps,
                    "loop_escape_rate": result.loop_escape_rate,
                    "novel_action_rate": result.novel_action_rate,
                },
                "by_scenario_type": result.by_scenario_type,
            })
            
            logger.info(
                f"  Recovery: {result.recovery_rate:.1%}, "
                f"Loop escape: {result.loop_escape_rate:.1%}"
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
            "n_scenarios": len(scenarios),
            "seed": args.seed,
        },
        "scenarios": scenarios,
        "results": results,
    }
    
    save_json(output, args.output_dir / "stuckness_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("STUCKNESS EVALUATION RESULTS")
    print("=" * 60)
    
    successful = [r for r in results if "metrics" in r]
    successful.sort(key=lambda x: -x["metrics"]["recovery_rate"])
    
    print(f"{'Model':<30} {'Recovery':<12} {'Loop Escape':<12} {'Novel':<12}")
    print("-" * 66)
    
    for r in successful:
        m = r["metrics"]
        print(
            f"{r['model']['name']:<30} "
            f"{m['recovery_rate']:.1%}        "
            f"{m['loop_escape_rate']:.1%}        "
            f"{m['novel_action_rate']:.1%}"
        )
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
