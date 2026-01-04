#!/usr/bin/env python3
"""
Mine a *global* retention task panel from Phase-0 rollouts.

Why
----
CPT and micro-training both benefit from a small curated retention set to detect
forgetting / negative transfer ("does this patch help the failure but hurt general behavior?").

This script is intentionally cheap:
- it simply reuses tasks where the *student/base* already succeeded in Phase 0.

Inputs
------
- rollouts.json produced by `src/simulation/student_rollout.StudentRollout.save_results()`

Output
------
- JSON list of task_ids (or {"tasks":[...]} if --wrap)

Example
-------
python scripts/phase0/mine_retention_tasks.py \
  --rollouts results/phase0/student_rollouts.json \
  --n 50 \
  --output panels/retention_tasks.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine retention tasks from rollouts")
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50, help="Number of unique successful tasks to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wrap", action="store_true", help="Write as {tasks:[...]} instead of a raw list")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.rollouts) as f:
        data = json.load(f)

    results = data.get("results") or []
    successes: List[Any] = []
    seen: Set[Any] = set()

    for r in results:
        if not r.get("success", False):
            continue
        tid = r.get("task_id")
        if tid is None:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        successes.append(tid)

    if not successes:
        raise RuntimeError("No successful tasks found in rollouts. Cannot build retention panel.")

    random.shuffle(successes)
    selected = successes[: args.n]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"tasks": selected} if args.wrap else selected, f, indent=2)

    print(f"Wrote {len(selected)} retention task ids to {args.output}")


if __name__ == "__main__":
    main()
