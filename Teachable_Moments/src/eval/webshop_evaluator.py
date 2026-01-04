"""WebShop evaluator for end-to-end success, transfer, and retention.

Designed to work with:
- src/data/webshop_env.WebShopEnvWrapper (dict obs with 'observation','valid_actions','instruction_text')
- a policy exposing get_action(observation, valid_actions, instruction_text=None)

This is intentionally minimal and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import time

from src.data.webshop_env import WebShopConfig, create_env


@dataclass
class EvalConfig:
    max_steps: int = 30
    n_rollouts: int = 1
    seed: int = 0


def evaluate_policy(
    policy: Any,
    task_ids: List[str],
    config: Optional[EvalConfig] = None,
    mock_env: bool = False,
) -> Dict[str, Any]:
    config = config or EvalConfig()
    rng = random.Random(config.seed)

    env = create_env(WebShopConfig(max_steps=config.max_steps), mock=mock_env)

    successes = 0
    rewards: List[float] = []
    per_task: List[Dict[str, Any]] = []

    start = time.time()
    for tid in task_ids:
        task_successes = 0
        task_rewards: List[float] = []
        for r in range(config.n_rollouts):
            obs = env.reset(tid)
            done = False
            total_reward = 0.0
            steps = 0
            while not done and steps < config.max_steps:
                action = policy.get_action(
                    obs.get("observation", ""),
                    obs.get("valid_actions", []),
                    instruction_text=obs.get("instruction_text", None),
                )
                obs, reward, done, _info = env.step(action)
                total_reward += float(reward)
                steps += 1
            is_succ = env.is_success(total_reward)
            task_successes += int(is_succ)
            task_rewards.append(total_reward)
        # aggregate
        if task_successes > 0:
            successes += 1
        rewards.extend(task_rewards)
        per_task.append({
            "task_id": tid,
            "success_any": task_successes > 0,
            "success_rate": task_successes / config.n_rollouts,
            "mean_reward": sum(task_rewards) / len(task_rewards),
        })

    return {
        "n_tasks": len(task_ids),
        "success_rate_any": successes / max(len(task_ids), 1),
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "per_task": per_task,
        "duration_seconds": time.time() - start,
    }
