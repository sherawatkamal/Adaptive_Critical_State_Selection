"""
Patch verifier: estimate_recovery_probability(env, agent, traj, step_t, patch, ...).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from idt.patches import Patch
from idt.types import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class VerifierResult:
    """Result of verification: success_rate, counts, compute."""

    success_rate: float
    successes: int
    attempts: int
    env_steps: int
    model_calls: int
    per_attempt_outcomes: Optional[List[bool]] = None


def estimate_recovery_probability(
    env: Any,
    agent: Any,
    traj: Trajectory,
    step_t: int,
    patch: Patch,
    attempt_budget: int,
    max_rollout_steps: int,
    stochastic_policy: bool = True,
    base_seed: int = 0,
) -> VerifierResult:
    """
    Reconstruct state at step_t by replaying actions[:step_t].
    Apply patch to get patched action(s) at/around t, then let agent continue.
    Run attempt_budget rollouts with seeds base_seed + k.
    Return VerifierResult (success_rate, successes, attempts, env_steps, model_calls).
    """
    successes = 0
    env_steps_total = 0
    model_calls_total = 0
    outcomes: List[bool] = []

    prefix_actions = traj.actions[:step_t]
    # Patched full action list then take suffix from step_t (patch applies to full list).
    patched_full = patch.apply(traj.actions)
    patched_rest = patched_full[step_t:] if step_t < len(patched_full) else []
    # So after prefix we execute patched_rest[0], patched_rest[1], ... (patch might replace one or insert).
    # For "patch then continue with policy": after prefix, we execute the patched action(s) at t,
    # then agent acts for remaining steps.
    # Simplification: execute all patched_rest then agent until done (or we interpret patch as only fixing step t).
    # Spec: "Execute prefix, execute patched action(s) at/around t, then let agent continue."
    # So: replay prefix -> obs_t. Then we have one (or more) patched actions. Execute them, then agent.
    # If patch is ReplaceActionPatch(step_t, new_action): we execute new_action once, then agent.
    # If InsertActionPatch(step_t, action): we execute action, then original action at t, then agent.
    # So patched_rest can have length len(rest_actions) or len(rest_actions)+1. We execute patched_rest one by one
    # until we've applied the "fixed" part, then switch to agent. Easiest: execute all of patched_rest (they're
    # the fixed actions for steps t, t+1, ...), then agent. But patched_rest might be longer than rollout.
    # Simpler: execute prefix, then first action from patched_rest (the patched step), then agent for rest.
    first_patched_action = patched_rest[0] if patched_rest else None

    for k in range(attempt_budget):
        seed = base_seed + k
        obs = env.replay(traj.task_id, prefix_actions)
        env_steps = 0
        model_calls = 0

        # Execute patched action at step t
        if first_patched_action:
            obs, reward, done, info = env.step(first_patched_action)
            env_steps += 1
            if done:
                success = env.is_success(reward, done, info)
                successes += 1 if success else 0
                outcomes.append(success)
                env_steps_total += env_steps
                model_calls_total += model_calls
                continue

        # Agent rollout
        history = list(prefix_actions)
        if first_patched_action:
            history.append(first_patched_action)
        for _ in range(max_rollout_steps - 1):
            action = agent.act(obs, history, info=info, stochastic=stochastic_policy, seed=seed)
            model_calls += 1
            obs, reward, done, info = env.step(action)
            env_steps += 1
            history.append(action)
            if done:
                break
        success = env.is_success(reward, done, info)
        successes += 1 if success else 0
        outcomes.append(success)
        env_steps_total += env_steps
        model_calls_total += model_calls

    attempts = attempt_budget
    success_rate = successes / attempts if attempts else 0.0
    return VerifierResult(
        success_rate=success_rate,
        successes=successes,
        attempts=attempts,
        env_steps=env_steps_total,
        model_calls=model_calls_total,
        per_attempt_outcomes=outcomes,
    )
