#!/usr/bin/env python3
"""Smoke test: student rollout produces replayable states + snapshot mining works.

Run:
  python scripts/tests/test_student_rollout_and_snapshot_mining.py

What it checks:
- StudentRollout saves env_state_b64 for each rollout step (replayable snapshots).
- mine_failure_snapshots produces snapshots with env_state_b64.
- We can restore an env from a mined snapshot and take a step.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    from src.simulation.student_rollout import StudentRollout, StudentRolloutConfig

    # Create tiny rollouts with mock env + dummy model (no transformers needed)
    class DummyModel:
        pass

    out_dir = root / "tmp" / "rollout_and_mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.json"
    snapshots_path = out_dir / "snapshots.json"

    cfg = StudentRolloutConfig(
        model_name="dummy",
        model_path=None,
        n_tasks=2,
        max_steps=4,
        mock_env=True,
        save_env_state=True,
        save_model_outputs=False,
    )

    sr = StudentRollout(cfg, model=DummyModel())
    results = sr.rollout_batch()
    sr.save_results(results, rollouts_path)

    data = json.loads(rollouts_path.read_text())
    assert "results" in data and len(data["results"]) > 0, "No rollouts saved"

    # Check env_state_b64 appears in rollout_states
    any_state = False
    for r in data["results"]:
        for st in r.get("rollout_states", []):
            any_state = True
            assert st.get("env_state_b64"), "Missing env_state_b64 in rollout state"
    assert any_state, "No rollout_states found"

    # Run mining script
    mine_script = root / "scripts" / "phase0" / "mine_failure_snapshots.py"
    subprocess.check_call(
        [
            sys.executable,
            str(mine_script),
            "--rollouts",
            str(rollouts_path),
            "--k",
            "1",
            "--output",
            str(snapshots_path),
        ]
    , cwd=str(root))

    snaps = json.loads(snapshots_path.read_text())
    assert isinstance(snaps, list) and len(snaps) > 0, "No snapshots mined"
    for s in snaps:
        assert s.get("env_state_b64"), "Mined snapshot missing env_state_b64"

    # Try replay
    from src.data.webshop_env import create_env, WebShopConfig

    env = create_env(WebShopConfig(max_steps=5), mock=True)
    b = base64.b64decode(snaps[0]["env_state_b64"].encode("utf-8"))
    obs = env.set_state(b)
    assert "observation" in obs and "valid_actions" in obs, "Bad set_state response"

    # take one step
    if obs["valid_actions"]:
        env.step(obs["valid_actions"][0])

    env.close()
    print("PASS: rollout -> mining -> replay")


if __name__ == "__main__":
    main()
