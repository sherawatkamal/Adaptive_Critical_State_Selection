#!/usr/bin/env python3
"""Evaluate retention over training checkpoints.

Tests for catastrophic forgetting by evaluating models at different
training checkpoints on a fixed evaluation panel.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.eval.retention import (
    compute_retention_curves,
    detect_catastrophic_forgetting,
    compare_retention_across_runs,
)


def find_checkpoints(model_dir: Path) -> list[Path]:
    """Find all checkpoints for a trained model."""
    checkpoints = []
    
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                checkpoints.append(item)
    
    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.name.split("-")[1]))
    
    return checkpoints


def load_evaluation_panel(path: Path) -> list[dict[str, Any]]:
    """Load evaluation panel for retention testing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    
    # Return empty if not found (will need to be created)
    return []


def main():
    parser = argparse.ArgumentParser(description="Retention evaluation")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/phase2/models"),
        help="Directory with trained models",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("panels/retention_panel.json"),
        help="Path to evaluation panel",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/retention"),
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
        "--forgetting-threshold",
        type=float,
        default=0.1,
        help="Threshold for detecting forgetting (relative drop)",
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
    
    # Load evaluation panel
    logger.info(f"Loading evaluation panel from {args.panel}")
    panel = load_evaluation_panel(args.panel)
    
    if not panel:
        logger.warning("No evaluation panel found - using labeled snapshots")
        # Fall back to labeled snapshots
        labeled_path = Path("results/phase1/labeled_snapshots.json")
        if labeled_path.exists():
            with open(labeled_path) as f:
                all_snaps = json.load(f)
            # Sample 100 for panel
            import random
            panel = random.sample(all_snaps, min(100, len(all_snaps)))
    
    logger.info(f"Panel size: {len(panel)}")
    
    # Find model directories
    model_dirs = []
    if args.models_dir.exists():
        for item in args.models_dir.iterdir():
            if item.is_dir():
                if args.models is None or item.name in args.models:
                    model_dirs.append(item)
    
    logger.info(f"Found {len(model_dirs)} model directories")
    
    # Compute retention curves for each model
    all_results = []
    
    for model_dir in model_dirs:
        logger.info(f"\nAnalyzing {model_dir.name}...")
        
        # Find checkpoints
        checkpoints = find_checkpoints(model_dir)
        
        if len(checkpoints) < 2:
            logger.warning(f"Not enough checkpoints for {model_dir.name}")
            continue
        
        logger.info(f"  Found {len(checkpoints)} checkpoints")
        
        try:
            # Compute retention curve
            retention_curve = compute_retention_curves(
                checkpoints=checkpoints,
                panel=panel,
                base_model=args.base_model,
            )
            
            # Detect forgetting
            forgetting = detect_catastrophic_forgetting(
                retention_curve,
                threshold=args.forgetting_threshold,
            )
            
            result = {
                "model": model_dir.name,
                "n_checkpoints": len(checkpoints),
                "retention_curve": [
                    {
                        "checkpoint": str(cp.checkpoint),
                        "step": cp.step,
                        "accuracy": cp.accuracy,
                        "loss": cp.loss,
                    }
                    for cp in retention_curve.checkpoints
                ],
                "forgetting_detected": forgetting.detected,
                "forgetting_magnitude": forgetting.magnitude,
                "forgetting_step": forgetting.step,
            }
            
            all_results.append(result)
            
            status = "FORGETTING" if forgetting.detected else "OK"
            logger.info(f"  [{status}] Final accuracy: {retention_curve.checkpoints[-1].accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error analyzing {model_dir.name}: {e}")
    
    # Compare across runs
    if len(all_results) > 1:
        comparison = compare_retention_across_runs(all_results)
    else:
        comparison = {}
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "base_model": args.base_model,
            "panel_size": len(panel),
            "forgetting_threshold": args.forgetting_threshold,
            "seed": args.seed,
        },
        "results": all_results,
        "comparison": comparison,
    }
    
    save_json(output, args.output_dir / "retention_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RETENTION ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n{'Model':<30} {'Checkpoints':<12} {'Final Acc':<12} {'Forgetting':<12}")
    print("-" * 66)
    
    for r in all_results:
        curve = r["retention_curve"]
        final_acc = curve[-1]["accuracy"] if curve else 0
        forget_status = "YES" if r["forgetting_detected"] else "No"
        
        print(
            f"{r['model']:<30} "
            f"{r['n_checkpoints']:<12} "
            f"{final_acc:.3f}        "
            f"{forget_status}"
        )
    
    if comparison:
        print("\n" + "-" * 60)
        print("COMPARISON")
        print(f"Most stable: {comparison.get('most_stable', 'N/A')}")
        print(f"Most forgetting: {comparison.get('most_forgetting', 'N/A')}")
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
