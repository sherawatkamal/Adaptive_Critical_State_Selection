"""
Agent adapter for IDT: act(), propose_actions(top_n) with optional fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionCandidate:
    """Single candidate action with score."""

    action: str
    score: float
    extra: Optional[Dict[str, Any]] = None


class AgentAdapterBase:
    """Interface: act(obs, history, info?, stochastic, seed), propose_actions(obs, history, info?, top_n, ...)."""

    def act(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        stochastic: bool = True,
        seed: int = 0,
    ) -> str:
        """Return one action string. info may contain 'valid' (valid actions) for WebShop-style agents."""
        raise NotImplementedError

    def propose_actions(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        stochastic: bool = True,
        seed: int = 0,
    ) -> List[ActionCandidate]:
        """Return top_n candidate actions with scores. Fallback: sample N with different seeds."""
        raise NotImplementedError


class ToyAgentAdapter(AgentAdapterBase):
    """Deterministic agent for toy env: prefers 'step' when state < 3, else 'stay'."""

    def act(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        stochastic: bool = True,
        seed: int = 0,
    ) -> str:
        if "state=3" in observation:
            return "stay"
        return "step"

    def propose_actions(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        stochastic: bool = True,
        seed: int = 0,
    ) -> List[ActionCandidate]:
        candidates = [ActionCandidate("step", 1.0), ActionCandidate("stay", 0.5)]
        return candidates[:top_n]


class FallbackAgentAdapter(AgentAdapterBase):
    """
    Wrapper that adds propose_actions via sampling when wrapped agent only has act().
    Sample N actions with seeds base_seed, base_seed+1, ...; deduplicate; use frequency as score.
    """

    def __init__(self, act_only_agent: Any, base_seed: int = 0) -> None:
        self._agent = act_only_agent
        self._base_seed = base_seed

    def act(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        stochastic: bool = True,
        seed: int = 0,
    ) -> str:
        return self._agent.act(observation, history, info=info, stochastic=stochastic, seed=seed)

    def propose_actions(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        stochastic: bool = True,
        seed: int = 0,
    ) -> List[ActionCandidate]:
        from collections import Counter
        actions: List[str] = []
        for k in range(max(top_n * 2, 10)):
            s = seed + k
            a = self._agent.act(observation, history, info=info, stochastic=True, seed=s)
            actions.append(a)
        counts = Counter(actions)
        ordered = counts.most_common(top_n)
        return [ActionCandidate(a, float(c) / len(actions)) for a, c in ordered]


class WebShopAgentAdapter(AgentAdapterBase):
    """Wraps baseline_models EEF Agent: get_action(obs, info, method) -> (action, action_info)."""

    def __init__(self, eef_agent: Any) -> None:
        self._agent = eef_agent

    def act(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        stochastic: bool = True,
        seed: int = 0,
    ) -> str:
        if info is None:
            info = {}
        method = "softmax" if stochastic else "greedy"
        action, _ = self._agent.get_action(observation, info, method=method)
        return action

    def propose_actions(
        self,
        observation: str,
        history: List[str],
        info: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        stochastic: bool = True,
        seed: int = 0,
    ) -> List[ActionCandidate]:
        """Fallback: sample N actions with different seeds; use frequency as score."""
        from collections import Counter
        if info is None:
            info = {}
        actions_list: List[str] = []
        for k in range(max(top_n * 2, 10)):
            method = "softmax"
            action, _ = self._agent.get_action(observation, info, method=method)
            actions_list.append(action)
        counts = Counter(actions_list)
        ordered = counts.most_common(top_n)
        return [ActionCandidate(a, float(c) / len(actions_list)) for a, c in ordered]
