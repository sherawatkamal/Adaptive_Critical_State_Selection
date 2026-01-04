"""Phase 0 (v8): Collect student rollouts.

This script exists primarily so higher-level orchestration (e.g. `run_all_pilots_v8.py`)
can depend on a stable CLI.

It runs the student policy in the WebShop environment for N tasks and writes a JSON
artifact compatible with downstream snapshot mining.

Example:
  python scripts/phase0/run_student_rollouts_v8.py \
    --model-path sshleifer/tiny-gpt2 \
    --n-tasks 5 \
    --max-steps 15 \
    --output results/phase0/rollouts.json

If you want a fast end-to-end smoke test without WebShop installed:
  python scripts/phase0/run_student_rollouts_v8.py --mock-env ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

# Ensure repo root on path so `src.*` imports work when running as a script
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.student_rollout import StudentRollout, StudentRolloutConfig

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Run student rollouts (v8) and save rollouts.json")
    ap.add_argument("--model-path", type=str, required=True, help="HF model path for the student")
    ap.add_argument("--n-tasks", type=int, default=100, help="# tasks to attempt")
    ap.add_argument("--max-steps", type=int, default=30, help="Max steps per episode")
    ap.add_argument("--output", type=str, required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--collect-all", action="store_true", help="If set, keep successes too")
    ap.add_argument("--mock-env", action="store_true", help="Use MockWebShopEnv (no WebShop required)")
    ap.add_argument("--task-ids", type=str, default=None, help="Optional JSON list of task IDs")

    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = None
    if args.task_ids:
        with open(args.task_ids) as f:
            task_ids = json.load(f)
        if not isinstance(task_ids, list):
            raise SystemExit("--task-ids must point to a JSON list")

    cfg = StudentRolloutConfig(
        model_name="student_base",
        model_path=args.model_path,
        max_steps=args.max_steps,
        n_tasks=args.n_tasks,
        task_ids=task_ids,
        collect_all=bool(args.collect_all),
        mock_env=bool(args.mock_env),
        save_model_outputs=True,
    )

    logger.info("Starting rollouts")
    sr = StudentRollout(cfg)
    results, stats = sr.run_rollouts()

    sr.save_results(str(out_path), results, stats)

    logger.info(f"Wrote rollouts to {out_path} ({len(results)} episodes)")


if __name__ == "__main__":
    main()
