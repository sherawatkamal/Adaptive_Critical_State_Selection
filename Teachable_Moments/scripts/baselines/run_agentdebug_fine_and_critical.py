#!/usr/bin/env python3
"""
Run AgentDebug Fine and Critical Baselines.

This script runs the AgentDebug Fine analysis (step-by-step verification)
and Critical analysis (focused on critical decision points).

Usage:
  python scripts/baselines/run_agentdebug_fine_and_critical.py \
    --trajectories results/phase0/student_failures/student_failures.json \
    --output results/baselines/agentdebug_fine.json \
    --mode fine \
    --teacher-model gpt-4o
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging
from src.baselines.agentdebug import AgentDebugBaseline, AgentDebugConfig
from src.teacher.client import create_teacher_client, TeacherConfig
from src.data.snapshot import TeacherHint, ErrorType

logger = logging.getLogger(__name__)


def analyze_step_by_step(client, trajectory: dict, max_steps: int = 5) -> list[dict]:
    """Perform fine-grained step-by-step analysis."""
    steps = trajectory.get("steps", trajectory.get("rollout_states", []))
    if not steps:
        return []
    
    results = []
    
    # Focus on last N steps (where failures typically occur)
    focus_steps = steps[-max_steps:] if len(steps) > max_steps else steps
    
    for i, step in enumerate(focus_steps):
        step_idx = step.get("step_idx", i)
        observation = step.get("observation", "")[:1000]
        action_taken = step.get("action_taken", "")
        valid_actions = step.get("valid_actions", [])[:10]
        
        prompt = f"""Analyze this single step from an agent trajectory.

Step {step_idx}:
Observation: {observation}
Action Taken: {action_taken}
Valid Actions: {', '.join(valid_actions)}

Is this a correct action? If not, what should the agent have done instead?

Respond with JSON:
{{
    "is_correct": true/false,
    "suggested_action": "action if incorrect",
    "rationale": "brief explanation",
    "error_type": "planning_error|execution_error|perception_error|none"
}}"""

        try:
            if hasattr(client, "generate_text"):
                resp = client.generate_text(prompt)
            else:
                resp = client.generate(prompt)
            
            import re
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                data = json.loads(match.group())
                results.append({
                    "step_idx": step_idx,
                    "is_correct": data.get("is_correct", True),
                    "suggested_action": data.get("suggested_action", ""),
                    "rationale": data.get("rationale", ""),
                    "error_type": data.get("error_type", "none"),
                })
        except Exception as e:
            logger.debug(f"Step {step_idx} analysis failed: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run AgentDebug Fine/Critical baseline")
    parser.add_argument("--trajectories", required=True, help="Path to trajectories JSON")
    parser.add_argument("--output", required=True, help="Output path for analysis results")
    parser.add_argument("--mode", choices=["fine", "critical"], default="fine", help="Analysis mode")
    parser.add_argument("--teacher-model", default="gpt-4o", help="Teacher model to use")
    parser.add_argument("--teacher-config", type=Path, default=None, help="Teacher config YAML")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit number of trajectories")
    parser.add_argument("--max-steps-per-traj", type=int, default=5, help="Max steps to analyze per trajectory")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Load trajectories
    with open(args.trajectories) as f:
        trajectories = json.load(f)
    
    if args.max_trajectories:
        trajectories = trajectories[:args.max_trajectories]
    
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Create teacher client
    if args.teacher_config and args.teacher_config.exists():
        teacher_cfg = TeacherConfig.from_yaml(str(args.teacher_config))
    else:
        teacher_cfg = TeacherConfig(model=args.teacher_model)
    teacher_cfg.model = args.teacher_model
    
    client = create_teacher_client(teacher_cfg)
    
    # Analyze trajectories
    results = []
    total_errors_found = 0
    
    for i, traj in enumerate(trajectories):
        traj_id = traj.get("trajectory_id", f"traj_{i}")
        logger.info(f"Analyzing trajectory {i+1}/{len(trajectories)}: {traj_id}")
        
        try:
            step_analyses = analyze_step_by_step(
                client, traj, max_steps=args.max_steps_per_traj
            )
            
            errors_found = sum(1 for s in step_analyses if not s.get("is_correct", True))
            total_errors_found += errors_found
            
            result = {
                "trajectory_id": traj_id,
                "task_id": traj.get("task_id", ""),
                "success": traj.get("success", False),
                "step_analyses": step_analyses,
                "n_steps_analyzed": len(step_analyses),
                "n_errors_found": errors_found,
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze {traj_id}: {e}")
            results.append({
                "trajectory_id": traj_id,
                "error": str(e),
                "step_analyses": [],
                "n_steps_analyzed": 0,
                "n_errors_found": 0,
            })
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "method": f"agentdebug_{args.mode}",
            "n_trajectories": len(trajectories),
            "n_analyzed": len(results),
            "total_errors_found": total_errors_found,
            "results": results,
        }, f, indent=2)
    
    logger.info(f"Wrote results to {output_path}")
    
    # Summary
    n_with_errors = sum(1 for r in results if r.get("n_errors_found", 0) > 0)
    print(f"\nAgentDebug {args.mode.capitalize()} Summary:")
    print(f"  Trajectories analyzed: {len(results)}")
    print(f"  With errors found: {n_with_errors}")
    print(f"  Total errors: {total_errors_found}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

