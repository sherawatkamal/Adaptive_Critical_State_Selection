#!/usr/bin/env python3
"""
Run AgentDebug Global Baseline.

This script runs the AgentDebug Global analysis (full-trajectory analysis)
on failure trajectories and produces hints/corrections using an LLM teacher.

Usage:
  python scripts/baselines/run_agentdebug_global.py \
    --trajectories results/phase0/student_failures/student_failures.json \
    --output results/baselines/agentdebug_global.json \
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

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run AgentDebug Global baseline")
    parser.add_argument("--trajectories", required=True, help="Path to trajectories JSON")
    parser.add_argument("--output", required=True, help="Output path for analysis results")
    parser.add_argument("--teacher-model", default="gpt-4o", help="Teacher model to use")
    parser.add_argument("--teacher-config", type=Path, default=None, help="Teacher config YAML")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit number of trajectories")
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
    
    # Create AgentDebug baseline
    config = AgentDebugConfig(mode="global")
    baseline = AgentDebugBaseline(client, config)
    
    # Analyze trajectories
    results = []
    for i, traj in enumerate(trajectories):
        traj_id = traj.get("trajectory_id", f"traj_{i}")
        logger.info(f"Analyzing trajectory {i+1}/{len(trajectories)}: {traj_id}")
        
        try:
            hints = baseline.analyze_trajectory(traj)
            
            result = {
                "trajectory_id": traj_id,
                "task_id": traj.get("task_id", ""),
                "success": traj.get("success", False),
                "hints": [h.to_dict() if hasattr(h, "to_dict") else {"suggested_action": h.suggested_action, "rationale": h.rationale, "error_type": h.error_type.value if hasattr(h.error_type, "value") else str(h.error_type), "confidence": h.confidence} for h in hints],
                "n_hints": len(hints),
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze {traj_id}: {e}")
            results.append({
                "trajectory_id": traj_id,
                "error": str(e),
                "hints": [],
                "n_hints": 0,
            })
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "method": "agentdebug_global",
            "n_trajectories": len(trajectories),
            "n_analyzed": len(results),
            "results": results,
        }, f, indent=2)
    
    logger.info(f"Wrote results to {output_path}")
    
    # Summary
    n_with_hints = sum(1 for r in results if r.get("n_hints", 0) > 0)
    print(f"\nAgentDebug Global Summary:")
    print(f"  Trajectories analyzed: {len(results)}")
    print(f"  With hints: {n_with_hints}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

