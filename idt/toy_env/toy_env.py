"""
Deterministic toy environment for IDT unit tests.

State machine: S0 -> S1 -> S2 -> S3 (goal). Actions: "step" (advance) or "stay".
A known failure trajectory is ["stay", "stay", "step"] (ends at S2).
Replacing the first "stay" with "step" yields ["step", "stay", "step"] -> success.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class ToyEnv:
    """Deterministic mini env: states 0,1,2,3. Action 'step' advances, 'stay' keeps state."""

    NUM_STATES = 4
    GOAL_STATE = 3
    VALID_ACTIONS = ("step", "stay")

    def __init__(self, seed: int = 0) -> None:
        self._state = 0
        self._step_count = 0
        self._max_steps = 10
        self._seed = seed
        self._instruction = "Reach state 3."

    def reset(self, task_id: Any = 0) -> str:
        """Reset to initial state. task_id ignored; env is single-task."""
        self._state = 0
        self._step_count = 0
        return self._observation()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action. Returns (observation, reward, done, info)."""
        action = action.strip().lower()
        if action not in self.VALID_ACTIONS:
            action = "stay"
        if action == "step" and self._state < self.GOAL_STATE:
            self._state += 1
        self._step_count += 1
        done = self._state == self.GOAL_STATE or self._step_count >= self._max_steps
        reward = 1.0 if (done and self._state == self.GOAL_STATE) else 0.0
        obs = self._observation()
        info: Dict[str, Any] = {"valid": list(self.VALID_ACTIONS), "state": self._state}
        return obs, reward, done, info

    def _observation(self) -> str:
        return f"state={self._state} goal=3 instruction={self._instruction}"

    def replay(self, task_id: Any, actions_prefix: List[str]) -> str:
        """Replay actions from reset and return observation at end. Deterministic."""
        self.reset(task_id)
        for a in actions_prefix:
            obs, _, done, _ = self.step(a)
            if done:
                return obs
        return self._observation()

    @property
    def state(self) -> int:
        return self._state


# Canonical failure trajectory: ends at S2 (not goal).
TOY_FAILURE_ACTIONS = ["stay", "stay", "step"]
# Patch: replace action at index 0 with "step" -> success.
TOY_PATCH_STEP = 0
TOY_PATCH_REPLACEMENT = "step"
# Success trajectory: three "step" actions reach state 3.
TOY_SUCCESS_ACTIONS = ["step", "step", "step"]
