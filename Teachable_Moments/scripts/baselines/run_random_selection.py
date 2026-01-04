#!/usr/bin/env python3
"""
Random Selection Baseline.

This baseline randomly selects training examples without any teachability
criteria. This provides a lower bound for comparison.

Comparison with teachable moments framework:
- Teachable moments: Strategic selection based on U(s) and L(s)
- This baseline: Pure random selection

Expected result: Lower improvement than any strategic selection method.

Usage:
    python scripts/baselines/run_random_selection.py \
        --snapshots results/phase1/labeled_snapshots.parquet \
        --n-samples 500 \
        --output results/baselines/random_selection/
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import random

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging, set_seed, ProgressTracker
from src.training.sft_trainer import SFTTrainer, SFTConfig
from src.supervision.formatter import SupervisionFormatter

logger = logging.getLogger(__name__)


@dataclass
class RandomSelectionConfig:
    """Configuration for random selection baseline."""
    
    # Selection
    n_samples: int = 500
    
    # Supervision format
    supervision_format: str = "demo"
    
    # Training
    base_model: str = "meta-llama/Llama-3-8B-Instruct"
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    
    # Output
    output_dir: str = "results/baselines/random_selection"
    seed: int = 42


def run_random_selection_experiment(
    snapshots_path: Path,
    config: RandomSelectionConfig,
) -> dict:
    """
    Run the random selection baseline experiment.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load snapshots
    logger.info(f"Loading snapshots from {snapshots_path}")
    
    if snapshots_path.suffix == ".parquet":
        df = pd.read_parquet(snapshots_path)
    elif snapshots_path.suffix == ".json":
        with open(snapshots_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {snapshots_path.suffix}")
    
    logger.info(f"Loaded {len(df)} snapshots")
    
    # Random sample
    n_samples = min(config.n_samples, len(df))
    df_sampled = df.sample(n=n_samples, random_state=config.seed)
    
    logger.info(f"Randomly selected {len(df_sampled)} snapshots")
    
    # Log distribution statistics for comparison
    stats = {}
    
    if "entropy" in df_sampled.columns:
        stats["entropy"] = {
            "mean": df_sampled["entropy"].mean(),
            "std": df_sampled["entropy"].std(),
        }
        logger.info(f"Entropy: mean={stats['entropy']['mean']:.3f}, std={stats['entropy']['std']:.3f}")
    
    if "leverage" in df_sampled.columns:
        stats["leverage"] = {
            "mean": df_sampled["leverage"].mean(),
            "std": df_sampled["leverage"].std(),
        }
        logger.info(f"Leverage: mean={stats['leverage']['mean']:.3f}, std={stats['leverage']['std']:.3f}")
    
    if "quadrant" in df_sampled.columns:
        stats["quadrant_distribution"] = df_sampled["quadrant"].value_counts().to_dict()
        logger.info(f"Quadrant distribution: {stats['quadrant_distribution']}")
    
    # Prepare training data
    formatter = SupervisionFormatter()
    training_examples = []
    
    for _, row in df_sampled.iterrows():
        snapshot = row.to_dict()
        example = formatter.format_example(
            snapshot=snapshot,
            supervision_type=config.supervision_format,
        )
        training_examples.append(example)
    
    # Save training data
    training_data_path = output_dir / "training_data.json"
    with open(training_data_path, "w") as f:
        json.dump(training_examples, f, indent=2)
    
    # Train model
    sft_config = SFTConfig(
        base_model=config.base_model,
        output_dir=str(output_dir / "checkpoints"),
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
    )
    
    trainer = SFTTrainer(sft_config)
    
    logger.info("Starting training...")
    training_result = trainer.train(training_examples)
    
    # Save results
    results = {
        "config": asdict(config),
        "n_samples": len(df_sampled),
        "selection_stats": stats,
        "training_result": training_result,
    }
    
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment complete. Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run random selection baseline"
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        required=True,
        help="Path to labeled snapshots",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--supervision-format",
        choices=["demo", "contrast", "hint"],
        default="demo",
        help="Supervision format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/random_selection"),
        help="Output directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Base model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    config = RandomSelectionConfig(
        n_samples=args.n_samples,
        supervision_format=args.supervision_format,
        output_dir=str(args.output),
        base_model=args.base_model,
        seed=args.seed,
    )
    
    results = run_random_selection_experiment(args.snapshots, config)
    
    print("\n" + "=" * 50)
    print("Random Selection Baseline Results")
    print("=" * 50)
    print(f"Samples selected: {results['n_samples']}")
    if "quadrant_distribution" in results.get("selection_stats", {}):
        print("\nQuadrant distribution (random):")
        for q, count in results["selection_stats"]["quadrant_distribution"].items():
            print(f"  {q}: {count}")


if __name__ == "__main__":
    main()
