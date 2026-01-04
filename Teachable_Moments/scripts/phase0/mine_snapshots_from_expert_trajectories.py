#!/usr/bin/env python3
"""
Mine restorable Snapshot objects from expert trajectory logs.

Input: expert_trajectories.json produced by scripts/phase0/collect_expert_trajectories.py
Output: snapshots_expert.json (list of Snapshot.to_dict() dicts)

This provides the "expert-sampled state distribution" experiment:
  - Sample states along expert rollouts (success + failure)
  - Later, label those states with student U and leverage to study teachability
    outside of only failure-centric states.

Typical use:
  python scripts/phase0/mine_snapshots_from_expert_trajectories.py \
    --expert-trajectories results/phase0/expert_trajectories/expert_trajectories.json \
    --output results/phase0/snapshots_expert.json \
    --sample-rate 1.0 \
    --require-env-state
"""

import argparse
import base64
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.snapshot import Snapshot
from src.utils.common import setup_logging


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _safe_b64_to_bytes(b64: str | None) -> bytes | None:
    if not b64:
        return None
    try:
        return base64.b64decode(b64.encode("utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-trajectories", required=True, help="Path to expert_trajectories.json")
    ap.add_argument("--output", required=True, help="Output snapshots JSON path")
    ap.add_argument("--sample-rate", type=float, default=1.0, help="Probability of keeping each step (0-1)")
    ap.add_argument("--max-snapshots", type=int, default=999999)
    ap.add_argument("--require-env-state", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    random.seed(args.seed)

    trajs = _load_json(Path(args.expert_trajectories))
    snapshots_out: list[dict] = []
    kept = 0

    for t in trajs:
        traj_id = t.get("trajectory_id", "unknown_traj")
        task_id = t.get("task_id", "unknown_task")
        task_desc = t.get("task_description", "")
        steps = t.get("steps", [])

        for s in steps:
            if kept >= args.max_snapshots:
                break
            if args.sample_rate < 1.0 and random.random() > args.sample_rate:
                continue

            step_idx = int(s.get("step_idx", 0))
            env_state_bytes = _safe_b64_to_bytes(s.get("env_state_b64"))

            if args.require_env_state and not env_state_bytes:
                continue

            snap = Snapshot(
                id=f"{traj_id}_step{step_idx}",
                task_id=str(task_id),
                step_idx=step_idx,
                observation=s.get("observation", ""),
                valid_actions=s.get("valid_actions", []) or [],
                action_taken=s.get("action_taken", ""),
                last_action=None,
                reward=float(s.get("reward", 0.0)),
                done=bool(s.get("done", False)),
                info={
                    "source": "expert_trajectory",
                    "trajectory_id": traj_id,
                    "task_description": task_desc,
                    "expert_model": t.get("expert_model", ""),
                    "success": t.get("success", False),
                },
                env_state_bytes=env_state_bytes,
                instruction_text=task_desc,
                agent_prefix=None,
            )
            snapshots_out.append(snap.to_dict())
            kept += 1

    _dump_json(snapshots_out, Path(args.output))
    print(f"Wrote {len(snapshots_out)} snapshots -> {args.output}")


if __name__ == "__main__":
    main()
