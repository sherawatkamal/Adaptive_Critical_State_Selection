#!/usr/bin/env python3
"""Run a small but comprehensive pilot that exercises the whole pipeline.

This pilot is intended to answer a single question:

  "If we can run this, can we confidently scale up and produce the paper figures?"

It runs:
  (P0) Collect student rollouts
  (P0) Mine snapshots from rollouts
  (P1) Label snapshots (uncertainty + leverage + CPT + teacher hints)
  (P2) Train a small subset of the training matrix
  (P3) Evaluate models (tasks + snapshots + transfer)
  (P4) Train the teachability predictor

Default settings are tiny and should be feasible on a laptop (depending on env/model).

NOTE: The P1 labeling step uses the teacher API by default (OpenAI). Export OPENAI_API_KEY.

Example:
  python scripts/pilots/run_all_pilots_v8.py \
      --out-dir results/pilot_v8 \
      --base-model sshleifer/tiny-gpt2 \
      --teacher-model gpt-4o-mini \
      --mock-env
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use exported env vars

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True, help="Where to write pilot artifacts")
    ap.add_argument("--config", type=str, default="configs/pilot_experiment.yaml", help="Pilot YAML config")
    ap.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2", help="HF model name or local checkpoint")
    ap.add_argument("--teacher-model", type=str, default="gpt-4o-mini", help="Teacher API model")
    ap.add_argument("--n-tasks", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--mock-env", action="store_true", help="Use the mock WebShop env")
    ap.add_argument("--skip-api", action="store_true", help="Use mock teacher (no API call)")
    ap.add_argument(
        "--run-ids",
        type=str,
        nargs="*",
        default=[
            "Q1_highU_highL_demo",
            "Q1_highU_highL_contrast",
            "Q1_highU_highL_hint",
            "B1_uniform",
        ],
        help="Subset of training-matrix run_ids to train in the pilot",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # P0: rollouts
    rollouts_path = out_dir / "phase0_rollouts.json"
    cmd = [
        sys.executable,
        "scripts/phase0/run_student_rollouts_v8.py",
        "--model-path",
        args.base_model,
        "--n-tasks",
        str(args.n_tasks),
        "--max-steps",
        str(args.max_steps),
        "--output",
        str(rollouts_path),
    ]
    if args.mock_env:
        cmd.append("--mock-env")
    _run(cmd)

    # P0: mine snapshots
    snapshots_path = out_dir / "phase0_snapshots.json"
    _run(
        [
            sys.executable,
            "scripts/phase0/mine_failure_snapshots.py",
            "--rollouts",
            str(rollouts_path),
            "--k",
            "3",
            "--max",
            "50",
            "--output",
            str(snapshots_path),
        ]
    )

    # P1: labeling
    phase1_dir = out_dir / "phase1_labeling"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_api and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Either export it, or run with --skip-api to use mock teacher."
        )

    cmd = [
        sys.executable,
        "scripts/phase1/run_labeling.py",
        "--snapshots",
        str(snapshots_path),
        "--output-dir",
        str(phase1_dir),
        "--config",
        args.config,
        "--student-checkpoint",
        args.base_model,
        "--expert-checkpoint",
        args.base_model,
        "--assign-quadrants",
        "--teacher-model",
        args.teacher_model,
    ]
    if args.mock_env:
        cmd.append("--mock-env")
    if args.skip_api:
        cmd.append("--mock-teacher")
    _run(cmd)

    labeled_path = phase1_dir / "labeled_snapshots.jsonl"

    # P2: training
    phase2_dir = out_dir / "phase2_training"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/phase2/run_training_matrix_v8.py",
        "--labeled-snapshots",
        str(labeled_path),
        "--config",
        args.config,
        "--base-model",
        args.base_model,
        "--output-dir",
        str(phase2_dir),
        "--max-train-samples",
        "20",
        "--baseline-n-samples",
        "20",
        "--run-ids",
    ] + list(args.run_ids)
    _run(cmd)

    # P3: evaluation
    phase3_dir = out_dir / "phase3_eval"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/phase3/run_eval_suite_v8.py",
        "--training-summary",
        str(phase2_dir / "training_summary.json"),
        "--labeled-snapshots",
        str(labeled_path),
        "--output-dir",
        str(phase3_dir),
        "--n-tasks",
        str(args.n_tasks),
        "--n-snapshots-per-quadrant",
        "5",
        "--max-steps",
        str(args.max_steps),
    ]
    if args.mock_env:
        cmd.append("--mock-env")
    _run(cmd)

    # P4: predictor
    phase4_dir = out_dir / "phase4_predictor"
    phase4_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/phase4/train_predictor_v8.py",
            "--labeled-snapshots",
            str(labeled_path),
            "--output-dir",
            str(phase4_dir),
        ]
    )

    print("\nPilot complete.")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
