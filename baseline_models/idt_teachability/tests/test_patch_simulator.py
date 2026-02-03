import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest

from idt_teachability.patch_simulator import PatchSimulator


class MockEnv:
    """
    A tiny deterministic env to test replay + forced patch logic.

    State is an integer. Actions are strings that map to next state.

    Terminal:
      - If state reaches 3 => done True, reward 10.0 (success)
      - Else reward 0.0

    Valid actions depend on state.
    """
    def __init__(self):
        self.state = 0
        self.goal = "reach-3"

    def reset(self, task_id=None):
        self.state = 0
        obs = f"state={self.state}"
        info = {"goal": self.goal, "valid": self._valid_actions()}
        return obs, info

    def _valid_actions(self):
        if self.state == 0:
            return ["a", "b"]
        if self.state == 1:
            return ["c", "d"]
        if self.state == 2:
            return ["e"]
        return []

    def step(self, action: str):
        # Transition logic
        if self.state == 0:
            self.state = 1 if action == "a" else 2
        elif self.state == 1:
            self.state = 3 if action == "c" else 2
        elif self.state == 2:
            self.state = 3 if action == "e" else 2

        done = self.state == 3
        reward = 10.0 if done else 0.0
        obs = f"state={self.state}"
        info = {"goal": self.goal, "valid": self._valid_actions()}
        return obs, reward, done, info


class MockAgent:
    """Always pick the first valid action."""
    def get_action(self, obs, info, method="softmax"):
        valid = info.get("valid", [])
        if not valid:
            return "noop", {"type": "noop"}
        return valid[0], {"type": "agent"}


@pytest.fixture
def sim():
    env = MockEnv()
    agent = MockAgent()
    return PatchSimulator(env, agent, max_steps=10, debug=False)


def test_replay_prefix_ok(sim):
    traj = {
        "task_id": "t0",
        "goal": "reach-3",
        "steps": [
            {"action_taken": "a"},  # state 0->1
            {"action_taken": "d"},  # state 1->2
        ],
    }
    rep = sim.replay_prefix(traj, target_step=2)
    assert rep.ok is True
    assert rep.obs == "state=2"
    assert rep.info["valid"] == ["e"]


def test_replay_prefix_ends_early(sim):
    # Replay reaches terminal before target_step
    traj = {
        "task_id": "t0",
        "goal": "reach-3",
        "steps": [
            {"action_taken": "b"},  # 0->2
            {"action_taken": "e"},  # 2->3 (done)
        ],
    }
    rep = sim.replay_prefix(traj, target_step=3)
    assert rep.ok is False
    assert rep.done is True
    assert rep.reward == 100.0  # 10.0 * 10.0 scaling


def test_forced_action_changes_outcome(sim):
    traj = {
        "task_id": "t0",
        "goal": "reach-3",
        "steps": [
            {"action_taken": "a"},  # 0->1
        ],
    }
    # Start from step 1 (state=1). Baseline agent picks "c" and succeeds.
    base = sim.rollout_from_state(traj, target_step=1, forced_first_action=None)
    assert base.success is True
    assert base.reward == 100.0

    # Force a worse action "d" at state=1 leads to state=2, but agent can still pick "e" and succeed.
    patched = sim.rollout_from_state(traj, target_step=1, forced_first_action="d")
    assert patched.success is True
    assert patched.reward == 100.0

    # Force invalid action should fall back to agent ("c") and succeed
    patched2 = sim.rollout_from_state(traj, target_step=1, forced_first_action="INVALID")
    assert patched2.success is True
    assert patched2.reward == 100.0
