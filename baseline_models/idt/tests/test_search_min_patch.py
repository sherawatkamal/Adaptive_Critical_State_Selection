"""Minimal patch search finds the correct patch and step in toy env."""

import pytest

from idt.env_adapter import ToyEnvAdapter
from idt.agent_adapter import ToyAgentAdapter
from idt.propose import HeuristicPatchProposer
from idt.search import SearchConfig, search_minimal_patch
from idt.types import Trajectory
from idt.toy_env.toy_env import TOY_FAILURE_ACTIONS, TOY_PATCH_STEP, TOY_PATCH_REPLACEMENT


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


def test_search_finds_patch():
    env = ToyEnvAdapter(seed=0)
    agent = ToyAgentAdapter()
    proposer = HeuristicPatchProposer(agent=agent)
    traj = _make_toy_traj()
    config = SearchConfig(
        attempt_budget_schedule=[1, 2],
        threshold=0.6,
        max_rollout_steps=10,
        max_candidates_per_type=5,
        base_seed=0,
    )
    result = search_minimal_patch(env, agent, proposer, traj, config=config)
    assert result.found_patch is True
    assert result.best_step == TOY_PATCH_STEP
    assert result.patch_type == "replace"
    assert result.teachable_label is True
    assert result.R1 >= 0.6
    assert result.best_patch is not None
    assert result.best_patch.get("payload") == TOY_PATCH_REPLACEMENT
