#!/usr/bin/env python3
"""Run micro-training experiments to validate CPT signal.

This script runs small-scale training experiments on individual snapshots
to measure the actual learning potential (ELP) and compare with CPT predictions.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.training.micro_trainer import MicroTrainer, MicroTrainingConfig
from src.data.trajectory import Snapshot


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    """Load labeled snapshots."""
    with open(path) as f:
        return json.load(f)


def select_validation_snapshots(
    snapshots: list[dict[str, Any]],
    n_per_quadrant: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select stratified sample for validation."""
    random.seed(seed)
    
    by_quadrant = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for snap in snapshots:
        q = snap.get("labels", {}).get("quadrant")
        if q and q in by_quadrant:
            by_quadrant[q].append(snap)
    
    selected = []
    for q, q_snaps in by_quadrant.items():
        if len(q_snaps) <= n_per_quadrant:
            selected.extend(q_snaps)
        else:
            selected.extend(random.sample(q_snaps, n_per_quadrant))
    
    return selected


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


def run_micro_experiments(
    snapshots: list[dict[str, Any]],
    model_name: str,
    output_dir: Path,
    config: MicroTrainingConfig,
) -> list[dict[str, Any]]:
    """Run micro-training on each snapshot."""
    logger = logging.getLogger(__name__)
    
    trainer = MicroTrainer(model_name, config)
    results = []
    
    for i, snap_dict in enumerate(snapshots):
        snap_id = snap_dict.get("snapshot_id", f"snap_{i}")
        logger.info(f"Processing snapshot {i+1}/{len(snapshots)}: {snap_id}")
        
        snapshot = dict_to_snapshot(snap_dict)
        labels = snap_dict.get("labels", {})
        
        # Create simple supervision for micro-training
        # Use the oracle action from CPT if available
        cpt_data = labels.get("cpt", {})
        oracle_action = cpt_data.get("oracle_action")
        
        if oracle_action is None:
            # Skip if no oracle action available
            logger.warning(f"No oracle action for {snap_id}, skipping")
            continue
        
        supervision = {
            "type": "demo",
            "snapshot": snapshot,
            "target_action": oracle_action,
        }
        
        # Run micro-training
        try:
            elp_result = trainer.compute_elp(snapshot, supervision)
            
            result = {
                "snapshot_id": snap_id,
                "quadrant": labels.get("quadrant"),
                "cpt_delta": cpt_data.get("delta", 0),
                "cpt_treated": cpt_data.get("prob_treated", 0),
                "cpt_control": cpt_data.get("prob_control", 0),
                "elp": elp_result.elp,
                "prob_before": elp_result.prob_before,
                "prob_after": elp_result.prob_after,
                "converged": elp_result.converged,
                "steps_to_converge": elp_result.steps_to_converge,
            }
            results.append(result)
            
            logger.info(
                f"  CPT delta: {result['cpt_delta']:.4f}, "
                f"ELP: {result['elp']:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error processing {snap_id}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run micro-training experiments")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase1/labeled_snapshots.json"),
        help="Path to labeled snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1b"),
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model to use for micro-training",
    )
    parser.add_argument(
        "--n-per-quadrant",
        type=int,
        default=10,
        help="Number of snapshots per quadrant to validate",
    )
    parser.add_argument(
        "--micro-steps",
        type=int,
        default=50,
        help="Number of micro-training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Path to experiment config",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    
    # Load snapshots
    logger.info(f"Loading snapshots from {args.input}")
    snapshots = load_snapshots(args.input)
    logger.info(f"Loaded {len(snapshots)} snapshots")
    
    # Select validation subset
    logger.info(f"Selecting {args.n_per_quadrant} per quadrant for validation")
    validation_snapshots = select_validation_snapshots(
        snapshots, args.n_per_quadrant, args.seed
    )
    logger.info(f"Selected {len(validation_snapshots)} snapshots")
    
    # Configure micro-training
    micro_config = MicroTrainingConfig(
        n_steps=args.micro_steps,
        learning_rate=1e-4,
        batch_size=1,
        convergence_threshold=0.01,
    )
    
    # Run experiments
    logger.info("Running micro-training experiments...")
    results = run_micro_experiments(
        validation_snapshots,
        args.model,
        args.output_dir,
        micro_config,
    )
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "model": args.model,
            "n_per_quadrant": args.n_per_quadrant,
            "micro_steps": args.micro_steps,
            "seed": args.seed,
        },
        "n_experiments": len(results),
        "results": results,
    }
    
    save_json(output, args.output_dir / "micro_training_results.json")
    logger.info(f"Results saved to {args.output_dir / 'micro_training_results.json'}")
    
    # Print summary
    print(f"\nCompleted {len(results)} micro-training experiments")
    
    by_quadrant = {}
    for r in results:
        q = r.get("quadrant", "unknown")
        if q not in by_quadrant:
            by_quadrant[q] = []
        by_quadrant[q].append(r)
    
    for q in sorted(by_quadrant.keys()):
        q_results = by_quadrant[q]
        avg_cpt = sum(r["cpt_delta"] for r in q_results) / len(q_results)
        avg_elp = sum(r["elp"] for r in q_results) / len(q_results)
        print(f"  {q}: n={len(q_results)}, avg CPT={avg_cpt:.4f}, avg ELP={avg_elp:.4f}")


if __name__ == "__main__":
    main()
