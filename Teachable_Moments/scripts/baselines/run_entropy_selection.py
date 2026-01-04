#!/usr/bin/env python3
"""
Entropy-Only Selection Baseline.

This baseline selects training examples based purely on uncertainty (entropy)
without considering leverage. This tests the hypothesis that high uncertainty
alone is not sufficient for identifying teachable moments.

Comparison with teachable moments framework:
- Teachable moments: Uses both uncertainty AND leverage
- This baseline: Uses only entropy (ignores leverage axis)

Expected result: Lower improvement than quadrant-based selection because
high-entropy states may be unsalvageable (high U, low L = "lost causes").

Usage:
    python scripts/baselines/run_entropy_selection.py \
        --snapshots results/phase1/labeled_snapshots.parquet \
        --n-samples 500 \
        --output results/baselines/entropy_selection/
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
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
class EntropySelectionConfig:
    """Configuration for entropy-based selection."""
    
    # Selection
    n_samples: int = 500
    selection_method: str = "top"  # "top", "threshold", "percentile"
    entropy_threshold: Optional[float] = None  # For threshold method
    entropy_percentile: float = 90  # For percentile method (top 10%)
    
    # Supervision format
    supervision_format: str = "demo"  # demo, contrast, hint
    
    # Training
    base_model: str = "meta-llama/Llama-3-8B-Instruct"
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    
    # Output
    output_dir: str = "results/baselines/entropy_selection"
    seed: int = 42


def load_and_filter_snapshots(
    snapshots_path: Path,
    config: EntropySelectionConfig,
) -> pd.DataFrame:
    """
    Load snapshots and filter by entropy.
    """
    logger.info(f"Loading snapshots from {snapshots_path}")
    
    # Load data
    if snapshots_path.suffix == ".parquet":
        df = pd.read_parquet(snapshots_path)
    elif snapshots_path.suffix == ".json":
        with open(snapshots_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {snapshots_path.suffix}")
    
    logger.info(f"Loaded {len(df)} snapshots")
    
    # Ensure entropy column exists
    if "entropy" not in df.columns and "uncertainty" in df.columns:
        df["entropy"] = df["uncertainty"]
    
    if "entropy" not in df.columns:
        raise ValueError("Snapshots must have 'entropy' or 'uncertainty' column")
    
    # Apply selection method
    if config.selection_method == "top":
        # Select top N by entropy
        df_sorted = df.nlargest(config.n_samples, "entropy")
        
    elif config.selection_method == "threshold":
        # Select all above threshold
        if config.entropy_threshold is None:
            raise ValueError("entropy_threshold required for threshold method")
        df_filtered = df[df["entropy"] >= config.entropy_threshold]
        if len(df_filtered) > config.n_samples:
            df_sorted = df_filtered.sample(n=config.n_samples, random_state=config.seed)
        else:
            df_sorted = df_filtered
            
    elif config.selection_method == "percentile":
        # Select top percentile
        threshold = np.percentile(df["entropy"], config.entropy_percentile)
        df_filtered = df[df["entropy"] >= threshold]
        if len(df_filtered) > config.n_samples:
            df_sorted = df_filtered.sample(n=config.n_samples, random_state=config.seed)
        else:
            df_sorted = df_filtered
    else:
        raise ValueError(f"Unknown selection method: {config.selection_method}")
    
    logger.info(f"Selected {len(df_sorted)} snapshots by entropy")
    logger.info(f"Entropy range: {df_sorted['entropy'].min():.3f} - {df_sorted['entropy'].max():.3f}")
    
    return df_sorted


def prepare_training_data(
    df: pd.DataFrame,
    supervision_format: str,
) -> List[dict]:
    """
    Prepare training examples in specified supervision format.
    """
    formatter = SupervisionFormatter()
    examples = []
    
    for _, row in df.iterrows():
        snapshot = row.to_dict()
        
        example = formatter.format_example(
            snapshot=snapshot,
            supervision_type=supervision_format,
        )
        examples.append(example)
    
    return examples


def run_entropy_selection_experiment(config: EntropySelectionConfig):
    """
    Run the entropy-only selection baseline experiment.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and filter snapshots
    snapshots_path = Path(config.snapshots_path) if hasattr(config, 'snapshots_path') else None
    if snapshots_path is None:
        raise ValueError("snapshots_path required")
    
    df = load_and_filter_snapshots(snapshots_path, config)
    
    # Analyze quadrant distribution (for comparison)
    if "quadrant" in df.columns:
        quadrant_dist = df["quadrant"].value_counts()
        logger.info(f"Quadrant distribution of selected samples:\n{quadrant_dist}")
        
        # Save distribution
        quadrant_dist.to_json(output_dir / "quadrant_distribution.json")
    
    # Analyze leverage distribution
    if "leverage" in df.columns:
        logger.info(f"Leverage stats: mean={df['leverage'].mean():.3f}, "
                   f"std={df['leverage'].std():.3f}")
    
    # Prepare training data
    training_examples = prepare_training_data(df, config.supervision_format)
    
    # Save training data
    training_data_path = output_dir / "training_data.json"
    with open(training_data_path, "w") as f:
        json.dump(training_examples, f, indent=2)
    logger.info(f"Saved {len(training_examples)} training examples")
    
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
        "n_samples": len(df),
        "entropy_stats": {
            "mean": df["entropy"].mean(),
            "std": df["entropy"].std(),
            "min": df["entropy"].min(),
            "max": df["entropy"].max(),
        },
        "training_result": training_result,
    }
    
    if "leverage" in df.columns:
        results["leverage_stats"] = {
            "mean": df["leverage"].mean(),
            "std": df["leverage"].std(),
            "min": df["leverage"].min(),
            "max": df["leverage"].max(),
        }
    
    if "quadrant" in df.columns:
        results["quadrant_distribution"] = df["quadrant"].value_counts().to_dict()
    
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment complete. Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run entropy-only selection baseline"
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
        "--selection-method",
        choices=["top", "threshold", "percentile"],
        default="top",
        help="Selection method",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        help="Entropy threshold (for threshold method)",
    )
    parser.add_argument(
        "--entropy-percentile",
        type=float,
        default=90,
        help="Entropy percentile (for percentile method)",
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
        default=Path("results/baselines/entropy_selection"),
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
    
    config = EntropySelectionConfig(
        n_samples=args.n_samples,
        selection_method=args.selection_method,
        entropy_threshold=args.entropy_threshold,
        entropy_percentile=args.entropy_percentile,
        supervision_format=args.supervision_format,
        output_dir=str(args.output),
        base_model=args.base_model,
        seed=args.seed,
    )
    config.snapshots_path = args.snapshots
    
    results = run_entropy_selection_experiment(config)
    
    print("\n" + "=" * 50)
    print("Entropy Selection Baseline Results")
    print("=" * 50)
    print(f"Samples selected: {results['n_samples']}")
    print(f"Entropy range: {results['entropy_stats']['min']:.3f} - {results['entropy_stats']['max']:.3f}")
    if "quadrant_distribution" in results:
        print("\nQuadrant distribution:")
        for q, count in results["quadrant_distribution"].items():
            print(f"  {q}: {count}")


if __name__ == "__main__":
    main()
