"""A tiny dependency-free policy for smoke tests.

This is **not** meant for research results. It exists so that you can run
pipeline smoke tests (rollout -> snapshot mining -> labeling) without
needing to download a large model.

API matches the rest of the codebase:
- get_action(observation, valid_actions, instruction_text=None) -> str
- get_action_distribution(...) -> Dict[action, probability]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class RandomPolicy:
    """Random policy over valid actions (uniform)."""

    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def get_action(
        self,
        observation: str,
        valid_actions: List[str],
        instruction_text: Optional[str] = None,
    ) -> str:
        if not valid_actions:
            return ""
        return self._rng.choice(valid_actions)

    def get_action_distribution(
        self,
        observation: str,
        valid_actions: List[str],
        instruction_text: Optional[str] = None,
    ) -> Dict[str, float]:
        if not valid_actions:
            return {}
        p = 1.0 / float(len(valid_actions))
        return {a: p for a in valid_actions}
