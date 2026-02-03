"""Replay reaches same state as original rollout (toy env)."""

import pytest

from idt.env_adapter import ToyEnvAdapter
from idt.toy_env.toy_env import TOY_FAILURE_ACTIONS, ToyEnv


def test_replay_same_state_as_rollout():
    env = ToyEnv(seed=0)
    env.reset(0)
    observations_rollout = [env._observation()]
    for a in TOY_FAILURE_ACTIONS:
        obs, _, done, _ = env.step(a)
        observations_rollout.append(obs)
        if done:
            break
    final_rollout = observations_rollout[-1]

    adapter = ToyEnvAdapter(seed=0)
    obs_replay = adapter.replay(0, TOY_FAILURE_ACTIONS)
    assert "state=" in obs_replay
    assert obs_replay == final_rollout


def test_replay_deterministic():
    adapter = ToyEnvAdapter(seed=42)
    o1 = adapter.replay(0, ["step", "stay"])
    o2 = adapter.replay(0, ["step", "stay"])
    assert o1 == o2
