#!/usr/bin/env python3
"""
Collect teacher demonstrations via simulation.

Runs the teacher model (e.g., GPT-4o) on tasks to collect:
1. Expert demonstrations for demo supervision
2. Comparison data for teachable gap analysis

Usage:
    python scripts/phase0/collect_teacher_demos.py \
        --model gpt-4o \
        --n-tasks 100 \
        --output results/teacher_demos.json

    # Or compare with student failures:
    python scripts/phase0/collect_teacher_demos.py \
        --model gpt-4o \
        --student-results results/student_failures.json \
        --output results/teacher_demos.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use exported env vars

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simulation.teacher_rollout import TeacherRollout, TeacherRolloutConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect teacher demonstrations via simulation"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Teacher model name (e.g., gpt-4o, claude-3-opus)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable for API key",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0 for deterministic)",
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
    
    # Comparison with student
    parser.add_argument(
        "--student-results",
        type=str,
        default=None,
        help="Path to student results for comparison",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only run on tasks where student failed",
    )
    
    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default="results/teacher_demos.json",
        help="Output path for demonstration data",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        default=True,
        help="Include teacher reasoning in output",
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
    logger.info("Teacher Demonstration Collection")
    logger.info("=" * 60)
    
    # Check API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.mock_env:
        logger.error(f"API key not found in {args.api_key_env}")
        return 1
    
    # Load student results if comparing
    student_results = None
    task_ids = None
    
    if args.student_results:
        logger.info(f"Loading student results from {args.student_results}")
        with open(args.student_results) as f:
            student_data = json.load(f)
        
        student_results = student_data.get("results", [])
        logger.info(f"Loaded {len(student_results)} student trajectories")
        
        # Filter to failures only if requested
        if args.failures_only:
            failed_tasks = [
                r["task_id"] for r in student_results
                if not r.get("success", False)
            ]
            task_ids = failed_tasks
            logger.info(f"Filtering to {len(task_ids)} failed tasks")
        else:
            task_ids = [r["task_id"] for r in student_results]
    
    elif args.task_ids:
        with open(args.task_ids) as f:
            task_ids = json.load(f)
        logger.info(f"Loaded {len(task_ids)} task IDs from {args.task_ids}")
    
    # Configure rollout
    config = TeacherRolloutConfig(
        model_name=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_steps=args.max_steps,
        n_tasks=args.n_tasks,
        task_ids=task_ids,
        mock_env=args.mock_env,
    )
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {len(task_ids) if task_ids else args.n_tasks}")
    logger.info(f"Max steps: {args.max_steps}")
    
    # Initialize rollout
    rollout = TeacherRollout(config)
    
    # Progress callback
    def progress(completed, total):
        if completed % 10 == 0 or completed == total:
            logger.info(f"Progress: {completed}/{total} tasks")
    
    # Run rollouts
    logger.info("Starting teacher rollouts...")
    results = rollout.rollout_batch(
        student_results=student_results,
        progress_callback=progress,
    )
    
    # Compute statistics
    stats = rollout.get_statistics(results)
    
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total rollouts: {stats['n_rollouts']}")
    logger.info(f"Successes: {stats['n_successes']} ({stats['success_rate']:.1%})")
    logger.info(f"Average steps: {stats['avg_steps']:.1f}")
    logger.info(f"Average reward: {stats['avg_reward']:.3f}")
    logger.info(f"Total API calls: {stats['total_api_calls']}")
    
    # Compare with student if available
    comparison = None
    if student_results:
        comparison = rollout.compare_with_student(results, student_results)
        
        logger.info("\nStudent-Teacher Comparison:")
        logger.info(f"  Common tasks: {comparison['n_common_tasks']}")
        logger.info(f"  Teachable gaps: {comparison['teachable_gaps']} ({comparison['teachable_gap_rate']:.1%})")
        logger.info(f"  Both succeed: {comparison['both_succeed']}")
        logger.info(f"  Both fail: {comparison['both_fail']}")
        logger.info(f"  Teacher success rate: {comparison['teacher_success_rate']:.1%}")
        logger.info(f"  Student success rate: {comparison['student_success_rate']:.1%}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rollout.save_results(
        results,
        str(output_path),
        include_reasoning=args.include_reasoning,
    )
    
    logger.info(f"\nSaved results to {output_path}")
    
    # Save comparison if available
    if comparison:
        comparison_path = output_path.with_stem(output_path.stem + "_comparison")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {comparison_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
