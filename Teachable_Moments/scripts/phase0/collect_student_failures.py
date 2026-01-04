#!/usr/bin/env python3
"""
Collect student failures via simulation.

This is the primary data collection script for teachable moments.
Runs the student model on tasks and collects failure cases.

Usage:
    python scripts/phase0/collect_student_failures.py \
        --model-path checkpoints/student_base \
        --n-tasks 500 \
        --output results/student_failures.json

Output:
    - Failure trajectories with full context
    - Failure classification (stuck, wrong_action, timeout, etc.)
    - Statistics on failure patterns
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simulation.student_rollout import StudentRollout, StudentRolloutConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect student failures via simulation"
    )
    
    # Model settings
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to student model checkpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="student",
        help="Name for this model (for tracking)",
    )
    
    # Rollout settings
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=100,
        help="Number of tasks to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="JSON file with specific task IDs to use",
    )
    
    # Failure detection settings
    parser.add_argument(
        "--loop-threshold",
        type=int,
        default=2,
        help="Repeats to detect as stuck loop",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Below this confidence = confusion",
    )
    
    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default="results/student_failures.json",
        help="Output path for failure data",
    )
    parser.add_argument(
        "--collect-all",
        action="store_true",
        help="Collect successful trajectories too",
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Don't save full trajectories (only failures)",
    )
    
    # Environment
    parser.add_argument(
        "--mock-env",
        action="store_true",
        help="Use mock environment for testing",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Student Failure Collection")
    logger.info("=" * 60)
    
    # Load task IDs if specified
    task_ids = None
    if args.task_ids:
        with open(args.task_ids) as f:
            task_ids = json.load(f)
        logger.info(f"Loaded {len(task_ids)} task IDs from {args.task_ids}")
    
    # Configure rollout
    config = StudentRolloutConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        max_steps=args.max_steps,
        n_tasks=args.n_tasks,
        task_ids=task_ids,
        loop_threshold=args.loop_threshold,
        confidence_threshold=args.confidence_threshold,
        collect_all=args.collect_all,
        mock_env=args.mock_env,
    )
    
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tasks: {args.n_tasks}")
    logger.info(f"Max steps: {args.max_steps}")
    
    # Initialize rollout
    rollout = StudentRollout(config)
    
    # Progress callback
    def progress(completed, total):
        if completed % 10 == 0 or completed == total:
            logger.info(f"Progress: {completed}/{total} tasks")
    
    # Run rollouts
    logger.info("Starting rollouts...")
    results = rollout.rollout_batch(progress_callback=progress)
    
    # Compute statistics
    stats = rollout.get_statistics(results)
    
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total rollouts: {stats['n_rollouts']}")
    logger.info(f"Successes: {stats['n_successes']} ({stats['success_rate']:.1%})")
    logger.info(f"Failures: {stats['n_failures']}")
    logger.info(f"Total failure events: {stats['total_failure_events']}")
    logger.info(f"Average steps: {stats['avg_steps']:.1f}")
    logger.info(f"Average reward: {stats['avg_reward']:.3f}")
    
    if stats.get("failure_type_breakdown"):
        logger.info("\nFailure Type Breakdown:")
        for ftype, count in stats["failure_type_breakdown"].items():
            logger.info(f"  {ftype}: {count}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rollout.save_results(
        results,
        str(output_path),
        include_full_trajectories=not args.no_trajectories,
    )
    
    logger.info(f"\nSaved results to {output_path}")
    
    # Also save failures-only file for convenience
    failures_only_path = output_path.with_stem(output_path.stem + "_failures_only")
    failures_data = {
        "statistics": stats,
        "failures": [
            f.to_dict()
            for r in results
            for f in r.failures
        ],
    }
    with open(failures_only_path, "w") as f:
        json.dump(failures_data, f, indent=2)
    
    logger.info(f"Saved failures-only to {failures_only_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
