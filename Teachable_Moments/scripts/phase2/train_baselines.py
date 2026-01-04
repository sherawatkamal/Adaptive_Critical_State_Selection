#!/usr/bin/env python3
"""Train baseline models for comparison.

Baselines:
1. Random sampling: Train on randomly selected snapshots (ignoring quadrant structure)
2. All data: Train on all labeled snapshots with mixed supervision
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from src.utils import setup_logging, save_json, set_seed, get_timestamp, ensure_dir
from src.training.sft_trainer import SFTTrainer, SFTConfig
from src.supervision.sft_dataset import SFTDataset
from src.supervision.format_router import format_supervision
from src.data.trajectory import Snapshot


def load_config(path: Path) -> dict[str, Any]:
    """Load experiment configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_labeled_snapshots(path: Path) -> list[dict[str, Any]]:
    """Load labeled snapshots."""
    with open(path) as f:
        return json.load(f)


def dict_to_snapshot(snap_dict: dict[str, Any]) -> Snapshot:
    """Convert dictionary to Snapshot object."""
    return Snapshot(
        trajectory_id=snap_dict.get("trajectory_id", "unknown"),
        step_index=snap_dict.get("step_index", 0),
        state=snap_dict.get("state", {}),
        action_space=snap_dict.get("action_space", []),
        task_description=snap_dict.get("task_description", ""),
        history=snap_dict.get("history", []),
        env_checkpoint=snap_dict.get("env_checkpoint"),
    )


def create_random_baseline_data(
    snapshots: list[dict[str, Any]],
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create training data by random sampling."""
    random.seed(seed)
    
    # Sample random snapshots
    sampled = random.sample(snapshots, min(n_samples, len(snapshots)))
    
    # Create supervision with random type assignment
    supervision_types = ["demo", "contrast", "hint"]
    training_data = []
    
    for snap_dict in sampled:
        snapshot = dict_to_snapshot(snap_dict)
        labels = snap_dict.get("labels", {})
        
        # Random supervision type
        sup_type = random.choice(supervision_types)
        
        # Get oracle action if available
        oracle_action = labels.get("cpt", {}).get("oracle_action")
        if oracle_action is None:
            oracle_action = "click[Buy Now]"  # Fallback
        
        # Format supervision
        formatted = format_supervision(
            sup_type,
            snapshot,
            oracle_action,
            contrast_negatives=["click[Cancel]", "search[wrong]", "go back"],
            hint_text=labels.get("hint", {}).get("text", "Consider the correct action."),
        )
        
        if formatted:
            training_data.append(formatted)
    
    return training_data


def create_all_data_baseline(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create training data using all snapshots with their assigned supervision."""
    training_data = []
    
    for snap_dict in snapshots:
        snapshot = dict_to_snapshot(snap_dict)
        labels = snap_dict.get("labels", {})
        
        # Use quadrant to determine supervision type
        quadrant = labels.get("quadrant", "Q4")
        
        # Map quadrant to best supervision type based on expected patterns
        sup_type_map = {
            "Q1": "demo",      # High U, High L -> demo most effective
            "Q2": "hint",      # High U, Low L -> hint for guidance
            "Q3": "contrast",  # Low U, High L -> contrast for disambiguation
            "Q4": "demo",      # Low U, Low L -> basic demo
        }
        sup_type = sup_type_map.get(quadrant, "demo")
        
        # Get oracle action
        oracle_action = labels.get("cpt", {}).get("oracle_action")
        if oracle_action is None:
            continue
        
        formatted = format_supervision(
            sup_type,
            snapshot,
            oracle_action,
            contrast_negatives=["click[Cancel]", "search[wrong]", "go back"],
            hint_text=labels.get("hint", {}).get("text", "Consider the correct action."),
        )
        
        if formatted:
            training_data.append(formatted)
    
    return training_data


def train_baseline(
    name: str,
    training_data: list[dict[str, Any]],
    model_name: str,
    output_dir: Path,
    config: SFTConfig,
) -> dict[str, Any]:
    """Train a single baseline model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training baseline: {name}")
    logger.info(f"  Training samples: {len(training_data)}")
    
    # Create dataset
    dataset = SFTDataset(training_data)
    
    # Create trainer
    trainer = SFTTrainer(model_name, config)
    
    # Train
    model_output = output_dir / name
    result = trainer.train(dataset, model_output)
    
    return {
        "name": name,
        "n_samples": len(training_data),
        "final_loss": result.final_loss,
        "training_time": result.training_time,
        "checkpoint_path": str(result.checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase1/labeled_snapshots.json"),
        help="Path to labeled snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/baselines"),
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Experiment configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name",
    )
    parser.add_argument(
        "--random-n",
        type=int,
        default=500,
        help="Number of samples for random baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without training",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    
    # Load configuration
    exp_config = load_config(args.config)
    model_name = args.model or exp_config.get("model", {}).get("name", "meta-llama/Llama-3.2-3B-Instruct")
    
    # Load snapshots
    logger.info(f"Loading snapshots from {args.input}")
    snapshots = load_labeled_snapshots(args.input)
    logger.info(f"Loaded {len(snapshots)} snapshots")
    
    # Create baseline datasets
    logger.info("Creating random baseline data...")
    random_data = create_random_baseline_data(snapshots, args.random_n, args.seed)
    
    logger.info("Creating all-data baseline...")
    all_data = create_all_data_baseline(snapshots)
    
    # Print plan
    print("\n" + "=" * 60)
    print("BASELINE TRAINING PLAN")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"\nBaselines:")
    print(f"  1. random_sampling: {len(random_data)} samples")
    print(f"  2. all_data: {len(all_data)} samples")
    print("=" * 60 + "\n")
    
    if args.dry_run:
        logger.info("Dry run - exiting")
        return
    
    # Training configuration
    training_cfg = exp_config.get("training", {})
    sft_config = SFTConfig(
        epochs=training_cfg.get("epochs", 3),
        batch_size=training_cfg.get("batch_size", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        lora_rank=training_cfg.get("lora_rank", 16),
        lora_alpha=training_cfg.get("lora_alpha", 32),
    )
    
    ensure_dir(args.output_dir)
    
    results = []
    
    # Train random baseline
    result = train_baseline(
        "random_sampling",
        random_data,
        model_name,
        args.output_dir,
        sft_config,
    )
    results.append(result)
    
    # Train all-data baseline
    result = train_baseline(
        "all_data",
        all_data,
        model_name,
        args.output_dir,
        sft_config,
    )
    results.append(result)
    
    # Save summary
    summary = {
        "timestamp": get_timestamp(),
        "config": {
            "model": model_name,
            "random_n": args.random_n,
            "seed": args.seed,
        },
        "results": results,
    }
    
    save_json(summary, args.output_dir / "baseline_summary.json")
    
    # Print results
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
    
    for r in results:
        print(f"  {r['name']}: loss={r['final_loss']:.4f}, n={r['n_samples']}")
    
    print(f"\nModels saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
