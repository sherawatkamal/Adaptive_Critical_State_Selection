"""
Environment adapter for IDT: WebShop and toy env with reset, step, replay.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from idt.types import Trajectory

logger = logging.getLogger(__name__)


class EnvAdapterBase:
    """Base interface: reset(task_id), step(action), replay(task_id, actions_prefix)."""

    def reset(self, task_id: Any) -> str:
        """Reset to task; return initial observation."""
        raise NotImplementedError

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action; return (observation, reward, done, info)."""
        raise NotImplementedError

    def replay(self, task_id: Any, actions_prefix: List[str]) -> str:
        """Replay actions from reset; return observation at end of prefix."""
        obs = self.reset(task_id)
        for a in actions_prefix:
            obs, reward, done, _ = self.step(a)
            if done:
                break
        return obs

    def is_success(self, reward: float, done: bool, info: Dict[str, Any]) -> bool:
        """Determine if episode is successful (repository convention)."""
        return bool(done and reward >= 1.0)


class ToyEnvAdapter(EnvAdapterBase):
    """Wraps idt.toy_env.ToyEnv for IDT."""

    def __init__(self, seed: int = 0) -> None:
        from idt.toy_env.toy_env import ToyEnv
        self._env = ToyEnv(seed=seed)

    def reset(self, task_id: Any) -> str:
        return self._env.reset(task_id)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        return self._env.step(action)

    def is_success(self, reward: float, done: bool, info: Dict[str, Any]) -> bool:
        return bool(done and reward >= 1.0)


class WebShopEnvAdapter(EnvAdapterBase):
    """
    Wraps WebShop text env: reset(task_id), step(action), replay(actions_prefix).
    Uses deterministic replay from reset. Caches observation after each prefix (keyed by traj_id, step).
    """

    def __init__(
        self,
        observation_mode: str = "text",
        num_products: Optional[int] = None,
        human_goals: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._observation_mode = observation_mode
        self._num_products = num_products
        self._human_goals = human_goals
        self._seed = seed
        self._env = None
        self._cache: Dict[tuple, str] = {}

    def _get_env(self):  # type: ignore
        if self._env is not None:
            return self._env
        try:
            from web_agent_site.envs import WebAgentTextEnv
            self._env = WebAgentTextEnv(
                observation_mode=self._observation_mode,
                filter_goals=None,
                limit_goals=-1,
                num_products=self._num_products,
                human_goals=self._human_goals,
            )
            return self._env
        except Exception as e:
            logger.warning("WebShop env not available: %s. Use ToyEnvAdapter for tests.", e)
            raise

    def reset(self, task_id: Any) -> str:
        env = self._get_env()
        obs, _ = env.reset(session=task_id)
        return obs

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        env = self._get_env()
        obs, reward, done, info = env.step(action)
        if info is None:
            info = {}
        return obs, float(reward), bool(done), info

    def is_success(self, reward: float, done: bool, info: Dict[str, Any]) -> bool:
        """WebShop: success when done and reward >= 1.0 (or score == 10 in some code)."""
        return bool(done and reward >= 1.0)

    def replay(self, task_id: Any, actions_prefix: List[str]) -> str:
        obs = self.reset(task_id)
        for a in actions_prefix:
            obs, _, done, _ = self.step(a)
            if done:
                break
        return obs

    def replay_cached(self, traj_id: str, task_id: Any, step_t: int, traj: Trajectory) -> str:
        """Replay prefix actions[:step_t] with cache key (traj_id, step_t)."""
        key = (traj_id, step_t)
        if key in self._cache:
            return self._cache[key]
        prefix = traj.actions[:step_t]
        obs = self.replay(task_id, prefix)
        self._cache[key] = obs
        return obs


class WebShopWebEnvAdapter(EnvAdapterBase):
    """
    Wraps baseline_models WebEnv (from env.py). Use when you need info['valid'] for the agent.
    Reward from WebEnv is scaled by 10; success = done and reward >= 10.
    """

    def __init__(self, web_env: Any) -> None:
        self._env = web_env
        self._last_info: Dict[str, Any] = {}

    def reset(self, task_id: Any) -> str:
        ob, info = self._env.reset(task_id)
        self._last_info = info if info else {}
        return ob

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        ob, reward, done, info = self._env.step(action)
        self._last_info = info if info else {}
        return ob, float(reward), bool(done), self._last_info

    def is_success(self, reward: float, done: bool, info: Dict[str, Any]) -> bool:
        """WebEnv scales reward by 10; EEF uses success = reward == 10."""
        return bool(done and reward >= 10.0)

    def get_info(self) -> Dict[str, Any]:
        """Return last info (e.g. valid actions) after reset or step."""
        return dict(self._last_info)
