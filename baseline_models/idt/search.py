"""
Minimal patch search: find minimal patch that achieves recovery above threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from idt.patches import Patch, is_search_action
from idt.types import PatchSearchResult, StepContext, Trajectory
from idt.verify import estimate_recovery_probability, VerifierResult

logger = logging.getLogger(__name__)


def _default_step_order(traj: Trajectory, max_steps_to_consider: Optional[int] = None) -> List[int]:
    """Prioritize later steps and search steps. Returns list of step indices to try."""
    n = traj.length
    if max_steps_to_consider is not None:
        n = min(n, max_steps_to_consider)
    indices = list(range(n))
    # Later steps first
    indices.sort(key=lambda t: (-t, t))
    # Search steps earlier in the sorted list (higher priority)
    def key(t: int) -> tuple:
        action = traj.action_at(t)
        is_search = 1 if is_search_action(action) else 0
        return (-is_search, -t, t)
    indices.sort(key=key)
    return indices


@dataclass
class SearchConfig:
    attempt_budget_schedule: List[int] = field(default_factory=lambda: [1, 3, 5])
    threshold: float = 0.6
    max_steps_to_consider: Optional[int] = None
    max_candidates_per_type: int = 5
    max_rollout_steps: int = 50
    stochastic_policy: bool = True
    base_seed: int = 0


def search_minimal_patch(
    env: Any,
    agent: Any,
    patch_proposer: Any,
    traj: Trajectory,
    config: Optional[SearchConfig] = None,
    step_order_fn: Optional[Callable[[Trajectory], List[int]]] = None,
) -> PatchSearchResult:
    """
    Iterate steps (later and search steps first). For each step, propose patches,
    evaluate at K=1 first; if success, escalate to K=3, K=5. Select best by minimal cost then highest R.
    """
    cfg = config or SearchConfig()
    step_order = step_order_fn(traj) if step_order_fn else _default_step_order(
        traj, cfg.max_steps_to_consider
    )

    total_env_steps = 0
    total_model_calls = 0
    best_patch: Optional[Patch] = None
    best_step: Optional[int] = None
    best_type: Optional[str] = None
    best_cost = float("inf")
    R1, R3, R5 = 0.0, 0.0, 0.0
    teachable = False

    for step_t in step_order:
        context = StepContext.from_trajectory(traj, step_t)
        candidates = patch_proposer.propose(
            traj, step_t, context,
            max_candidates_per_type=cfg.max_candidates_per_type,
            seed=cfg.base_seed + step_t,
        )
        for patch in candidates:
            # K=1 first
            v1 = estimate_recovery_probability(
                env, agent, traj, step_t, patch,
                attempt_budget=cfg.attempt_budget_schedule[0],
                max_rollout_steps=cfg.max_rollout_steps,
                stochastic_policy=cfg.stochastic_policy,
                base_seed=cfg.base_seed,
            )
            total_env_steps += v1.env_steps
            total_model_calls += v1.model_calls
            if v1.success_rate < cfg.threshold:
                continue
            r1 = v1.success_rate
            r3, r5 = r1, r1
            if len(cfg.attempt_budget_schedule) > 1:
                v3 = estimate_recovery_probability(
                    env, agent, traj, step_t, patch,
                    attempt_budget=cfg.attempt_budget_schedule[1],
                    max_rollout_steps=cfg.max_rollout_steps,
                    stochastic_policy=cfg.stochastic_policy,
                    base_seed=cfg.base_seed,
                )
                total_env_steps += v3.env_steps
                total_model_calls += v3.model_calls
                r3 = v3.success_rate
            if len(cfg.attempt_budget_schedule) > 2:
                v5 = estimate_recovery_probability(
                    env, agent, traj, step_t, patch,
                    attempt_budget=cfg.attempt_budget_schedule[2],
                    max_rollout_steps=cfg.max_rollout_steps,
                    stochastic_policy=cfg.stochastic_policy,
                    base_seed=cfg.base_seed,
                )
                total_env_steps += v5.env_steps
                total_model_calls += v5.model_calls
                r5 = v5.success_rate
            cost = patch.cost()
            # Prefer lower cost; then higher r5; then earlier step (minimal intervention).
            if cost < best_cost or (cost == best_cost and r5 > R5) or (
                cost == best_cost and r5 == R5 and (best_step is None or step_t < best_step)
            ):
                best_cost = cost
                best_patch = patch
                best_step = step_t
                best_type = patch.patch_type
                R1, R3, R5 = r1, r3, r5
                teachable = r5 >= cfg.threshold

    return PatchSearchResult(
        traj_id=traj.traj_id,
        task_id=traj.task_id,
        found_patch=best_patch is not None,
        best_patch=best_patch.to_dict() if best_patch else None,
        best_step=best_step,
        patch_type=best_type,
        R1=R1,
        R3=R3,
        R5=R5,
        teachable_label=teachable,
        total_env_steps=total_env_steps,
        total_model_calls=total_model_calls,
        compute_counters={"env_steps": total_env_steps, "model_calls": total_model_calls},
    )
