import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _write_pilot_config(path: Path) -> None:
    # Small budgets so CI-style runs finish quickly.
    yaml_text = """
experiment:
  name: test_labeling_smoke
  seed: 1

environment:
  max_steps: 6

labeling:
  uncertainty:
    n_mc_samples: 1
    temperature: 1.0
  leverage:
    n_rollouts_A: 1
    n_rollouts_B: 1
    max_steps: 6
  cpt:
    n_per_condition: 1
    max_steps: 6
  quadrant:
    u_threshold: 0.5
    l_threshold: 0.3
    adaptive_thresholds: false
"""
    path.write_text(yaml_text)


def test_run_labeling_cli_smoke(tmp_path: Path):
    # Import lazily after creating tmp_path to avoid accidental import path issues.
    sys.path.insert(0, str(ROOT))
    from src.data.webshop_env import create_env, WebShopConfig

    env = create_env(WebShopConfig(max_steps=6), mock=True)
    obs = env.reset("task0")
    env_state_b64 = base64.b64encode(env.get_state()).decode("utf-8")

    snapshots = [
        {
            "id": "snap0",
            "task_id": "task0",
            "trajectory_id": "traj0",
            "step_idx": 0,
            "env_state_b64": env_state_b64,
            "observation": obs["observation"],
            "valid_actions": obs["valid_actions"],
            "last_action": "search[item]",
            "agent_prefix": "",
        }
    ]

    snap_path = tmp_path / "snapshots.json"
    snap_path.write_text(json.dumps(snapshots, indent=2))

    cfg_path = tmp_path / "config.yaml"
    _write_pilot_config(cfg_path)

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "phase1" / "run_labeling.py"),
        "--snapshots",
        str(snap_path),
        "--output-dir",
        str(out_dir),
        "--config",
        str(cfg_path),
        "--mock-env",
        "--mock-policy",
        "--mock-teacher",
        "--assign-quadrants",
    ]

    subprocess.check_call(cmd, cwd=str(ROOT))

    labeled_path = out_dir / "labeled_snapshots.json"
    assert labeled_path.exists()

    labeled = json.loads(labeled_path.read_text())
    assert isinstance(labeled, list) and len(labeled) == 1

    rec = labeled[0]
    assert "snapshot" in rec and "teacher_hint" in rec["snapshot"]
    assert "quadrant" in rec
