#!/usr/bin/env python3
"""
Mine restorable Snapshot objects from student rollout logs.

Input: student_failures.json produced by scripts/phase0/collect_student_failures.py
Output: snapshots_student.json (list of Snapshot.to_dict() dicts)

This script is intentionally "mechanical": it does NOT do labeling (U/L/quadrants).
It only extracts *restorable* decision states (env_state_b64 -> env_state_bytes).

Typical use:
  python scripts/phase0/mine_snapshots_from_student_rollouts.py \
    --student-rollouts results/phase0/student_failures/student_failures.json \
    --output results/phase0/snapshots_student.json \
    --strategy failure_steps \
    --include-k-before 2 \
    --max-snapshots-per-trajectory 5 \
    --require-env-state
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path
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


def _safe_b64_to_bytes(b64: Optional[str]) -> Optional[bytes]:
    if not b64:
        return None
    try:
        return base64.b64decode(b64.encode("utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-rollouts", required=True, help="Path to student_failures.json")
    ap.add_argument("--output", required=True, help="Output snapshots JSON path")
    ap.add_argument(
        "--strategy",
        choices=["failure_steps", "all_steps"],
        default="failure_steps",
        help="Which steps to extract as snapshots",
    )
    ap.add_argument(
        "--include-k-before",
        type=int,
        default=0,
        help="Also include k steps BEFORE each failure step (e.g., 2 => failure-2,failure-1,failure)",
    )
    ap.add_argument(
        "--max-snapshots-per-trajectory",
        type=int,
        default=999999,
        help="Cap snapshots per trajectory (after de-dup)",
    )
    ap.add_argument(
        "--require-env-state",
        action="store_true",
        help="Drop snapshots that don't have env_state_b64 (recommended).",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)

    rollouts = _load_json(Path(args.student_rollouts))
    snapshots_out: list[dict] = []

    for r in rollouts:
        traj_id = r.get("trajectory_id", "unknown_traj")
        task_id = r.get("task_id", "unknown_task")
        instruction_text = r.get("instruction_text", "")

        rollout_states = r.get("rollout_states", [])
        # Backwards-compatible: some logs store a flat `steps` list.
        if (not rollout_states) and isinstance(r.get("steps"), list):
            rollout_states = []
            for i, step in enumerate(r.get("steps") or []):
                if not isinstance(step, dict):
                    continue
                rollout_states.append(
                    {
                        "step_idx": i,
                        "observation": step.get("observation", ""),
                        "valid_actions": step.get("valid_actions", []) or [],
                        "env_state_b64": step.get("env_state_b64"),
                        "done": step.get("done"),
                        "reward": step.get("reward"),
                    }
                )

        # Build candidate step indices
        step_indices: set[int] = set()
        if args.strategy == "all_steps":
            step_indices.update(
                [int(s.get("step_idx")) for s in rollout_states if isinstance(s, dict) and s.get("step_idx") is not None]
            )
        else:
            failures = r.get("failures", [])
            # Backwards-compatible: infer failures from terminal steps when explicit failures are absent.
            if (not failures) and rollout_states:
                terminal_indices = [
                    int(s.get("step_idx"))
                    for s in rollout_states
                    if isinstance(s, dict) and s.get("done") is True and s.get("step_idx") is not None
                ]
                if terminal_indices:
                    failures = [{"step_idx": max(terminal_indices)}]
            for f in failures:
                fs = f.get("step_idx")
                if fs is None:
                    continue
                fs = int(fs)
                for k in range(int(args.include_k_before), -1, -1):
                    step_indices.add(max(0, fs - k))

        step_indices = sorted({i for i in step_indices if isinstance(i, int) and i >= 0})

        # Index rollout_states by step_idx
        by_idx = {
            int(s.get("step_idx")): s
            for s in rollout_states
            if isinstance(s, dict) and s.get("step_idx") is not None
        }

        per_traj = 0
        for step_idx in step_indices:
            if per_traj >= args.max_snapshots_per_trajectory:
                break

            s = by_idx.get(step_idx)
            if not s:
                continue

            env_state_bytes = _safe_b64_to_bytes(s.get("env_state_b64"))
            if args.require_env_state and not env_state_bytes:
                continue

            snap = Snapshot(
                id=f"{traj_id}_step{step_idx}",
                task_id=str(task_id),
                step_idx=int(step_idx),
                trajectory_id=str(traj_id),
                observation=s.get("observation", ""),
                valid_actions=s.get("valid_actions", []) or [],
                env_state_bytes=env_state_bytes,
                agent_prefix=None,
            )

            snapshots_out.append(snap.to_dict())
            per_traj += 1

    _dump_json(snapshots_out, Path(args.output))
    print(f"Wrote {len(snapshots_out)} snapshots -> {args.output}")


if __name__ == "__main__":
    main()
