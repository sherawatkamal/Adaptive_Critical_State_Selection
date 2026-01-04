import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_train_one_run_end_to_end(tmp_path: Path):
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("Set RUN_SLOW_TESTS=1 to enable HF download/training smoke test")

    # Minimal labeled snapshot with a teacher action, enough to exercise LoRA SFT.
    labeled = [
        {
            "snapshot_id": "s1",
            "quadrant": "Q1_highU_highL",
            "snapshot": {
                "id": "s1",
                "task_id": "t1",
                "observation": "You are on a page.",
                "valid_actions": ["click[foo]"],
                "last_action": "click[foo]",
                "agent_prefix": "",
                "teacher_hint": {"suggested_action": "click[foo]", "rationale": "do it"},
            },
        }
    ]

    labeled_path = tmp_path / "labeled.json"
    labeled_path.write_text(json.dumps(labeled))

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "phase2" / "run_training_matrix_v8.py"),
        "--labeled-snapshots",
        str(labeled_path),
        "--output-dir",
        str(out_dir),
        "--config",
        str(ROOT / "configs" / "pilot_experiment.yaml"),
        "--base-model",
        "sshleifer/tiny-gpt2",
        "--run-ids",
        "Q1_highU_highL_demo",
        "--max-train-samples",
        "1",
    ]

    subprocess.check_call(cmd, cwd=str(ROOT))

    summary_path = out_dir / "training_summary.json"
    assert summary_path.exists(), "Expected training_summary.json"

    summary = json.loads(summary_path.read_text())
    assert "model_paths" in summary and summary["model_paths"], "Expected at least one saved model"
