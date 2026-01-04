import base64
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.snapshot import Snapshot


def test_snapshot_from_dict_accepts_action_taken_and_agent_prefix_none():
    state = {"x": 1}
    env_state_b64 = base64.b64encode(pickle.dumps(state)).decode("utf-8")

    d = {
        "id": "s1",
        "task_id": "t1",
        "trajectory_id": "traj1",
        "step_index": 3,
        "env_state_b64": env_state_b64,
        "observation": "obs",
        "valid_actions": ["a", "b"],
        "action_taken": "click[a]",
        "agent_prefix": None,
    }

    s = Snapshot.from_dict(d)
    assert s.id == "s1"
    assert s.step_idx == 3
    assert s.last_action == "click[a]"
    assert s.agent_prefix == ""
    assert isinstance(s.env_state_bytes, (bytes, type(None)))
    assert s.env_state_bytes is not None
