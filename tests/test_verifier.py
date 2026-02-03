"""Verifier returns success_rate=1.0 for known good patch, 0.0 for bad (toy env)."""

import pytest

from idt.env_adapter import ToyEnvAdapter
from idt.agent_adapter import ToyAgentAdapter
from idt.patches import ReplaceActionPatch
from idt.types import Trajectory
from idt.verify import estimate_recovery_probability
from idt.toy_env.toy_env import TOY_FAILURE_ACTIONS, TOY_PATCH_STEP, TOY_PATCH_REPLACEMENT
from idt.agent_adapter import ActionCandidate


def _make_toy_traj():
    return Trajectory(
        traj_id="0",
        task_id=0,
        instruction="Reach state 3.",
        observations=[f"state={j} goal=3" for j in range(4)],
        actions=TOY_FAILURE_ACTIONS.copy(),
        rewards=[0.0] * 3,
        done=True,
        info={},
    )


def test_verifier_good_patch_succeeds():
    env = ToyEnvAdapter(seed=0)
    agent = ToyAgentAdapter()
    traj = _make_toy_traj()
    patch = ReplaceActionPatch(TOY_PATCH_STEP, TOY_PATCH_REPLACEMENT)
    result = estimate_recovery_probability(
        env, agent, traj, TOY_PATCH_STEP, patch,
        attempt_budget=3,
        max_rollout_steps=10,
        stochastic_policy=False,
        base_seed=0,
    )
    assert result.attempts == 3
    assert result.success_rate == 1.0
    assert result.successes == 3


def test_verifier_bad_patch_fails():
    """Use an agent that always stays; then patch replace(0, stay) never reaches goal."""
    env = ToyEnvAdapter(seed=0)

    class StayOnlyAgent:
        def act(self, observation, history, info=None, stochastic=True, seed=0):
            return "stay"
        def propose_actions(self, observation, history, info=None, top_n=5, stochastic=True, seed=0):
            return [ActionCandidate("stay", 1.0)]

    agent = StayOnlyAgent()
    traj = _make_toy_traj()
    patch = ReplaceActionPatch(0, "stay")
    result = estimate_recovery_probability(
        env, agent, traj, 0, patch,
        attempt_budget=2,
        max_rollout_steps=10,
        stochastic_policy=False,
        base_seed=0,
    )
    assert result.success_rate == 0.0
    assert result.successes == 0
