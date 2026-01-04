#!/usr/bin/env python3
"""
Analyze teachable gaps between student and teacher.

This script identifies the most valuable teaching opportunities:
where the student fails but the teacher succeeds.

Usage:
    python scripts/phase0/analyze_gaps.py \
        --student-results results/student_failures.json \
        --teacher-results results/teacher_demos.json \
        --output results/teachable_gaps.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simulation.failure_detector import FailureDetector, FailureDetectorConfig
from simulation.student_rollout import RolloutResult, FailureEvent, FailureType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze teachable gaps between student and teacher"
    )
    
    parser.add_argument(
        "--student-results",
        type=str,
        required=True,
        help="Path to student rollout results",
    )
    parser.add_argument(
        "--teacher-results",
        type=str,
        default=None,
        help="Path to teacher rollout results (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/teachable_gaps.json",
        help="Output path for gap analysis",
    )
    
    # Analysis settings
    parser.add_argument(
        "--min-pattern-count",
        type=int,
        default=3,
        help="Minimum occurrences to form a pattern",
    )
    parser.add_argument(
        "--teachability-threshold",
        type=float,
        default=0.5,
        help="Minimum teachability score to include",
    )
    
    # Output options
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top teachable moments to highlight",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "report", "both"],
        default="both",
        help="Output format",
    )
    
    return parser.parse_args()


def load_student_results(path: str) -> list[RolloutResult]:
    """Load student results from JSON."""
    with open(path) as f:
        data = json.load(f)
    
    results = []
    for r in data.get("results", []):
        # Reconstruct RolloutResult
        failures = []
        for f in r.get("failures", []):
            failure = FailureEvent(
                trajectory_id=f["trajectory_id"],
                step_idx=f["step_idx"],
                failure_type=FailureType(f["failure_type"]),
                state=f["state"],
                action_taken=f["action_taken"],
                valid_actions=f.get("valid_actions", []),
                recent_actions=f.get("recent_actions", []),
                recent_states=f.get("recent_states", []),
                cumulative_reward=f.get("cumulative_reward", 0),
                steps_remaining=f.get("steps_remaining", 0),
                action_probs=f.get("action_probs"),
                model_confidence=f.get("model_confidence"),
                teacher_action=f.get("teacher_action"),
                teacher_would_succeed=f.get("teacher_would_succeed"),
            )
            failures.append(failure)
        
        result = RolloutResult(
            trajectory_id=r["trajectory_id"],
            task_id=r["task_id"],
            success=r["success"],
            total_reward=r["total_reward"],
            n_steps=r["n_steps"],
            states=r.get("states", []),
            actions=r.get("actions", []),
            rewards=r.get("rewards", []),
            failures=failures,
            duration_seconds=r.get("duration_seconds", 0),
            model_name=r.get("model_name", ""),
        )
        results.append(result)
    
    return results


def generate_report(analysis: dict, top_k: int) -> str:
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("TEACHABLE MOMENTS ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary
    summary = analysis.get("summary", {})
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total failures analyzed: {summary.get('total_failures', 0)}")
    lines.append(f"Unique patterns found: {summary.get('unique_patterns', 0)}")
    lines.append(f"Teachable gaps identified: {summary.get('teachable_gaps', 0)}")
    lines.append(f"Average failure step: {summary.get('avg_failure_step', 0):.1f}")
    lines.append("")
    
    # Failure type distribution
    type_dist = summary.get("failure_type_distribution", {})
    if type_dist:
        lines.append("FAILURE TYPE DISTRIBUTION")
        lines.append("-" * 40)
        for ftype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
            lines.append(f"  {ftype}: {count}")
        lines.append("")
    
    # Top patterns
    patterns = analysis.get("patterns", [])
    if patterns:
        lines.append(f"TOP FAILURE PATTERNS (showing {min(len(patterns), 10)})")
        lines.append("-" * 40)
        for i, pattern in enumerate(patterns[:10]):
            lines.append(f"\n{i+1}. {pattern['pattern_id']}")
            lines.append(f"   Type: {pattern['failure_type']}")
            lines.append(f"   Count: {pattern['count']}")
            lines.append(f"   Description: {pattern['description']}")
            lines.append(f"   Teachability: {pattern['teachability_score']:.2f}")
            lines.append(f"   Suggested intervention: {pattern['suggested_intervention']}")
        lines.append("")
    
    # Teachable gaps
    gaps = analysis.get("teachable_gaps", [])
    if gaps:
        lines.append(f"TOP TEACHABLE GAPS (showing {min(len(gaps), top_k)})")
        lines.append("-" * 40)
        for i, gap in enumerate(gaps[:top_k]):
            lines.append(f"\n{i+1}. Task: {gap['task_id']}")
            lines.append(f"   Student failure: {gap['student_failure_type']} at step {gap['student_failure_step']}")
            lines.append(f"   Student action: {gap['student_action_at_failure']}")
            lines.append(f"   Teacher action: {gap['teacher_action_at_same_point']}")
            lines.append(f"   Strategy diff: {gap['strategy_difference']}")
            lines.append(f"   Teachability: {gap['teachability_score']:.2f}")
            lines.append(f"   Recommended supervision: {gap['supervision_recommendation']}")
        lines.append("")
    
    # Recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(recommendations[:10]):
            priority = rec.get("priority", "medium")
            lines.append(f"\n{i+1}. [{priority.upper()}] {rec.get('action', 'No action specified')}")
            if "pattern" in rec:
                lines.append(f"   Pattern: {rec['pattern']}")
            if "count" in rec:
                lines.append(f"   Affected instances: {rec['count']}")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Teachable Gap Analysis")
    logger.info("=" * 60)
    
    # Load student results
    logger.info(f"Loading student results from {args.student_results}")
    student_results = load_student_results(args.student_results)
    logger.info(f"Loaded {len(student_results)} student trajectories")
    
    # Count failures
    total_failures = sum(len(r.failures) for r in student_results)
    logger.info(f"Total failure events: {total_failures}")
    
    # Load teacher results if available
    teacher_results = None
    if args.teacher_results:
        logger.info(f"Loading teacher results from {args.teacher_results}")
        with open(args.teacher_results) as f:
            teacher_data = json.load(f)
        teacher_results = teacher_data.get("results", [])
        logger.info(f"Loaded {len(teacher_results)} teacher trajectories")
    
    # Configure detector
    config = FailureDetectorConfig(
        min_pattern_count=args.min_pattern_count,
    )
    
    # Run analysis
    logger.info("Running failure analysis...")
    detector = FailureDetector(config)
    analysis = detector.analyze_failures(student_results, teacher_results)
    
    # Filter by teachability
    if args.teachability_threshold > 0:
        original_gap_count = len(analysis.get("teachable_gaps", []))
        analysis["teachable_gaps"] = [
            g for g in analysis.get("teachable_gaps", [])
            if g.get("teachability_score", 0) >= args.teachability_threshold
        ]
        logger.info(
            f"Filtered gaps by teachability >= {args.teachability_threshold}: "
            f"{original_gap_count} -> {len(analysis['teachable_gaps'])}"
        )
    
    # Sort gaps by teachability
    analysis["teachable_gaps"] = sorted(
        analysis.get("teachable_gaps", []),
        key=lambda g: g.get("teachability_score", 0),
        reverse=True,
    )
    
    # Print summary
    summary = analysis.get("summary", {})
    logger.info("\nAnalysis Summary:")
    logger.info(f"  Total failures: {summary.get('total_failures', 0)}")
    logger.info(f"  Unique patterns: {summary.get('unique_patterns', 0)}")
    logger.info(f"  Teachable gaps: {len(analysis.get('teachable_gaps', []))}")
    
    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.output_format in ["json", "both"]:
        detector.save_analysis(analysis, str(output_path))
        logger.info(f"Saved JSON analysis to {output_path}")
    
    if args.output_format in ["report", "both"]:
        report = generate_report(analysis, args.top_k)
        report_path = output_path.with_suffix(".txt")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")
        
        # Also print to console
        print("\n" + report)
    
    # Save prioritized training data
    training_data_path = output_path.with_stem(output_path.stem + "_training_data")
    training_data = {
        "high_priority": [
            g for g in analysis.get("teachable_gaps", [])
            if g.get("teachability_score", 0) >= 0.7
        ],
        "medium_priority": [
            g for g in analysis.get("teachable_gaps", [])
            if 0.4 <= g.get("teachability_score", 0) < 0.7
        ],
        "patterns_to_address": [
            p for p in analysis.get("patterns", [])
            if p.get("count", 0) >= 5
        ],
    }
    
    with open(training_data_path, "w") as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"Saved prioritized training data to {training_data_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
