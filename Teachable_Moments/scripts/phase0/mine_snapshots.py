#!/usr/bin/env python3
"""
Mine snapshots from expert trajectories (Setup A).

Converts expert trajectory logs (from collect_expert_trajectories.py)
into canonical Snapshot objects for downstream processing.
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.snapshot import Snapshot
from src.utils.common import setup_logging

logger = logging.getLogger(__name__)


def mine_snapshots(
    trajectory_file: str,
    output_file: str,
    max_snapshots: Optional[int] = None,
) -> int:
    """
    Mine snapshots from trajectory file.
    
    Args:
        trajectory_file: Path to trajectory JSON file
        output_file: Path to output JSON file
        max_snapshots: Maximum number of snapshots to mine
        
    Returns:
        Number of snapshots mined
    """
    with open(trajectory_file) as f:
        trajectories = json.load(f)
    
    snapshots = []
    
    for traj in trajectories:
        if max_snapshots and len(snapshots) >= max_snapshots:
            break
            
        traj_id = traj.get("trajectory_id", "unknown")
        task_id = traj.get("task_id", "unknown")
        success = traj.get("success", False)
        
        # Check each step for env state
        for step in traj.get("steps", []):
            if max_snapshots and len(snapshots) >= max_snapshots:
                break
                
            env_state_b64 = step.get("env_state_b64")
            
            # If we have a state, create a snapshot
            if env_state_b64:
                step_idx = step.get("step_idx", 0)
                
                # Create snapshot
                snapshot = Snapshot.from_dict({
                    "id": f"{traj_id}_step{step_idx}",
                    "task_id": task_id,
                    "trajectory_id": traj_id,
                    "step_idx": step_idx,
                    "observation": step.get("observation", ""),
                    "valid_actions": step.get("valid_actions", []),
                    "last_action": step.get("action_taken") or step.get("last_action"),
                    "env_state_b64": env_state_b64,
                    # Expert trajectories are usually from GPT-4
                    "agent_prefix": traj.get("expert_model", "gpt-4"),
                    "trajectory_outcome": "success" if success else "failure",
                })
                
                snapshots.append(snapshot.to_dict())
                
    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(snapshots, f, indent=2)
        
    logger.info(f"Mined {len(snapshots)} snapshots from {len(trajectories)} trajectories")
    return len(snapshots)


def main():
    parser = argparse.ArgumentParser(description="Mine snapshots from expert trajectories")
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Path to trajectory JSON file (e.g., all_trajectories_*.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for snapshots JSON",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum snapshots to mine",
    )
    
    args = parser.parse_args()
    setup_logging()
    
    mine_snapshots(args.trajectories, args.output, args.max)


if __name__ == "__main__":
    main()
