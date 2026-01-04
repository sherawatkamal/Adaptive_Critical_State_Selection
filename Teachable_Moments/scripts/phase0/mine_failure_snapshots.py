#!/usr/bin/env python3
"""Mine replayable failure snapshots from student rollouts.

Input: rollouts.json produced by src/simulation/student_rollout.StudentRollout.save_results()
Output: JSON list of Snapshot dicts compatible with src/data/snapshot.py

Policy:
- For each rollout, use first failure event step_idx if present; else last step.
- Select snapshot at max(0, failure_step - k).
- Requires rollout_states[*].env_state_b64 to be present (use the patched student_rollout.py).
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Allow running as a script from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.snapshot import Snapshot


def _safe_get_failure_step(result: Dict[str, Any]) -> int:
    failures = result.get("failures") or []
    if failures:
        try:
            return int(failures[0].get("step_idx", 0))
        except Exception:
            return 0
    # fallback: last action step
    rollout_states = result.get("rollout_states") or []
    if rollout_states:
        return int(rollout_states[-1].get("step_idx", len(rollout_states) - 1))
    return max(0, int(result.get("n_steps", 1)) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="Path to rollouts.json")
    ap.add_argument("--k", type=int, default=3, help="Steps before failure to snapshot")
    ap.add_argument("--output", required=True, help="Output JSON for mined snapshots")
    ap.add_argument("--max", type=int, default=None, help="Optional max snapshots")
    args = ap.parse_args()

    with open(args.rollouts, "r") as f:
        data = json.load(f)

    results = data.get("results") or []
    snapshots: List[dict] = []
    ts = time.time()

    for r in results:
        rollout_states = r.get("rollout_states") or []
        if not rollout_states:
            # cannot mine without restorable states
            continue

        failure_step = _safe_get_failure_step(r)
        target_step = max(0, failure_step - args.k)

        # pick the rollout_state with matching step_idx if possible
        rs = None
        for cand in rollout_states:
            if int(cand.get("step_idx", -1)) == target_step:
                rs = cand
                break
        if rs is None:
            # fallback: index
            idx = min(target_step, len(rollout_states) - 1)
            rs = rollout_states[idx]

        env_state_b64 = rs.get("env_state_b64")
        if not env_state_b64:
            continue

        snap = Snapshot.from_dict({
            "id": f"{r.get('task_id','unknown')}_step{int(rs.get('step_idx',0))}",
            "task_id": r.get("task_id", "unknown"),
            "step_idx": int(rs.get("step_idx", 0)),
            "trajectory_id": r.get("trajectory_id", "unknown"),
            "env_state_b64": env_state_b64,
            "observation": rs.get("observation", ""),
            "valid_actions": rs.get("valid_actions", []),
            "agent_prefix": rs.get("instruction_text", ""),
            "last_action": rs.get("action_taken") or rs.get("last_action") or None,
            "trajectory_outcome": "failure" if not r.get("success", False) else "success",
            "timestamp": ts,
        })
        snapshots.append(snap.to_dict())
        if args.max is not None and len(snapshots) >= args.max:
            break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(snapshots, f, indent=2)

    print(f"Wrote {len(snapshots)} snapshots to {args.output}")


if __name__ == "__main__":
    main()
