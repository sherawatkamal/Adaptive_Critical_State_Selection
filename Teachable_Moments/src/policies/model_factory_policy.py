"""Policy wrappers used throughout labeling/training/evaluation.

This file adds a single canonical interface around utils.model_factory.ModelFactory:

- get_action(observation, valid_actions, instruction_text=None) -> str
- get_action_distribution(...) -> Dict[action, prob]

Using this wrapper prevents prompt drift and keeps UQ/leverage/CPT consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelFactoryPolicy:
    """A policy backed by ModelFactory.decode_action() and masked scoring."""

    model_factory: Any
    model: Any = None
    tokenizer: Any = None

    def _ensure_loaded(self) -> Tuple[Any, Any]:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = self.model_factory.load()
            # common: set pad token
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model, self.tokenizer

    def get_action(
        self,
        observation: str,
        valid_actions: List[str],
        instruction_text: Optional[str] = None,
    ) -> str:
        self._ensure_loaded()
        action, _probs, _raw = self.model_factory.decode_action(
            observation=observation,
            valid_actions=valid_actions,
            task_description=instruction_text,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        return action

    def get_action_distribution(
        self,
        observation: str,
        valid_actions: List[str],
        instruction_text: Optional[str] = None,
    ) -> Dict[str, float]:
        self._ensure_loaded()
        _action, probs, _raw = self.model_factory.decode_action(
            observation=observation,
            valid_actions=valid_actions,
            task_description=instruction_text,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        # Already normalized by ModelFactory scoring; return as-is
        return probs
