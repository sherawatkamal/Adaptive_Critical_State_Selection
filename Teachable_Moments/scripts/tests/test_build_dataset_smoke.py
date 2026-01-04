#!/usr/bin/env python3
"""Smoke test: build_dataset runs end-to-end on mock env.

Run:
  python scripts/tests/test_build_dataset_smoke.py

What it checks:
- build_dataset.py can:
  (1) read rollouts.json
  (2) mine snapshots
  (3) generate mock teacher hints
  (4) compute UQ features
  (5) (optionally) run leverage + CPT with tiny budgets
  (6) write expected output files
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    from src.simulation.student_rollout import StudentRollout, StudentRolloutConfig

    class DummyModel:
        pass

    out_dir = root / "tmp" / "build_dataset_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.json"

    # Create tiny rollouts
    cfg = StudentRolloutConfig(
        model_name="dummy",
        model_path=None,
        n_tasks=2,
        max_steps=4,
        mock_env=True,
        save_env_state=True,
        save_model_outputs=True,
    )
    sr = StudentRollout(cfg, model=DummyModel())
    results = sr.rollout_batch()
    sr.save_results(results, rollouts_path)

    build_script = root / "scripts" / "phase1" / "build_dataset.py"
    ds_dir = out_dir / "dataset_out"

    # Run with tiny leverage/CPT budgets (keeps smoke test fast)
    subprocess.check_call(
        [
            sys.executable,
            str(build_script),
            "--rollouts",
            str(rollouts_path),
            "--output-dir",
            str(ds_dir),
            "--mock-env",
            "--student-policy",
            "first",
            "--teacher-hints",
            "mock",
            "--max-snapshots",
            "2",
            "--snapshot-offsets",
            "1",
            "--panel-n",
            "2",
            "--env-max-steps",
            "6",
            "--leverage-n-policy",
            "1",
            "--leverage-n-force",
            "1",
            "--leverage-horizon",
            "2",
            "--leverage-parallel",
            "1",
            "--cpt-n-per-condition",
            "1",
            "--cpt-horizon",
            "2",
            "--cpt-max-steps",
            "2",
        ]
    , cwd=str(root))

    # Validate outputs
    expected = [
        ds_dir / "snapshots.json",
        ds_dir / "labeled_snapshots.json",
        ds_dir / "thresholds.json",
        ds_dir / "panel.json",
        ds_dir / "summary.json",
    ]
    for p in expected:
        assert p.exists(), f"Missing output: {p}"

    labeled = json.loads((ds_dir / "labeled_snapshots.json").read_text())
    assert isinstance(labeled, list) and len(labeled) > 0, "No labeled records"

    # Each record should have core keys
    need = {"id", "U", "uncertainty", "teacher_hint", "snapshot", "quadrant"}
    for rec in labeled:
        missing = need - set(rec.keys())
        assert not missing, f"Record missing keys: {missing}"

    print("PASS: build_dataset smoke")


if __name__ == "__main__":
    main()
