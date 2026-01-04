import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_run_training_matrix_v8_dry_run(tmp_path: Path):
    labeled = [
        {
            "snapshot_id": "s1",
            "quadrant": "Q1_highU_highL",
            "snapshot": {
                "id": "s1",
                "task_id": "t1",
                "observation": "obs",
                "valid_actions": ["a"],
                "last_action": "a",
                "agent_prefix": "",
                "teacher_hint": {"suggested_action": "a", "rationale": "r"},
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
        "--config",
        str(ROOT / "configs" / "experiment.yaml"),
        "--output-dir",
        str(out_dir),
        "--base-model",
        "gpt2",
        "--dry-run",
    ]

    subprocess.check_call(cmd, cwd=str(ROOT))

    # Dry run should still write training_matrix.json for reproducibility.
    assert (out_dir / "training_matrix.json").exists()
