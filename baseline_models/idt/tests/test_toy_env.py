"""Unit tests for toy env: reset/step correctness."""

import pytest

from idt.toy_env.toy_env import ToyEnv, TOY_FAILURE_ACTIONS, TOY_SUCCESS_ACTIONS


def test_toy_env_reset():
    env = ToyEnv(seed=0)
    obs = env.reset(0)
    assert "state=0" in obs
    assert env.state == 0


def test_toy_env_step_advance():
    env = ToyEnv(seed=0)
    env.reset(0)
    obs, reward, done, info = env.step("step")
    assert env.state == 1
    assert reward == 0.0
    assert not done
    obs, reward, done, info = env.step("step")
    assert env.state == 2
    obs, reward, done, info = env.step("step")
    assert env.state == 3
    assert done
    assert reward == 1.0


def test_toy_env_stay():
    env = ToyEnv(seed=0)
    env.reset(0)
    obs, reward, done, info = env.step("stay")
    assert env.state == 0
    assert reward == 0.0


def test_toy_failure_trajectory_does_not_reach_goal():
    env = ToyEnv(seed=0)
    env.reset(0)
    for a in TOY_FAILURE_ACTIONS:
        obs, reward, done, info = env.step(a)
        if done:
            break
    assert env.state != ToyEnv.GOAL_STATE or reward != 1.0


def test_toy_success_trajectory_reaches_goal():
    env = ToyEnv(seed=0)
    env.reset(0)
    for a in TOY_SUCCESS_ACTIONS:
        obs, reward, done, info = env.step(a)
        if done:
            break
    assert env.state == ToyEnv.GOAL_STATE
    assert reward == 1.0
