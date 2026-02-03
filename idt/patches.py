"""
Patch types: ReplaceActionPatch, InsertActionPatch, EditQueryPatch.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Patch(ABC):
    """Base patch: apply(actions) -> new actions, cost(), to_dict/from_dict."""

    patch_type: str
    step_t: int
    payload: Any

    @abstractmethod
    def apply(self, actions: List[str]) -> List[str]:
        """Return new action list with patch applied."""
        pass

    @abstractmethod
    def cost(self) -> float:
        """Patch cost (lower = smaller intervention)."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {"patch_type": self.patch_type, "step_t": self.step_t, "payload": self.payload}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Patch":
        t = d.get("patch_type", "")
        if t == "replace":
            return ReplaceActionPatch.from_dict(d)
        if t == "insert":
            return InsertActionPatch.from_dict(d)
        if t == "edit_query":
            return EditQueryPatch.from_dict(d)
        raise ValueError(f"Unknown patch_type: {t}")


class ReplaceActionPatch(Patch):
    """Replace action at step_t with new_action."""

    def __init__(self, step_t: int, new_action: str) -> None:
        super().__init__(patch_type="replace", step_t=step_t, payload=new_action)

    def apply(self, actions: List[str]) -> List[str]:
        out = list(actions)
        if 0 <= self.step_t < len(out):
            out[self.step_t] = self.payload
        return out

    def cost(self) -> float:
        return 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReplaceActionPatch":
        return cls(step_t=d["step_t"], new_action=d["payload"])


class InsertActionPatch(Patch):
    """Insert action_to_insert at step_t (before the action that was at step_t)."""

    def __init__(self, step_t: int, action_to_insert: str) -> None:
        super().__init__(patch_type="insert", step_t=step_t, payload=action_to_insert)

    def apply(self, actions: List[str]) -> List[str]:
        out = list(actions)
        out.insert(self.step_t, self.payload)
        return out

    def cost(self) -> float:
        return 2.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InsertActionPatch":
        return cls(step_t=d["step_t"], action_to_insert=d["payload"])


def _parse_search_action(action: str) -> tuple:
    """Return (prefix, query) for search[query] or (None, None)."""
    if not action.strip().startswith("search["):
        return None, None
    m = re.match(r"search\[(.*)\]", action.strip(), re.DOTALL)
    if m:
        return "search[", m.group(1).strip()
    return None, None


class EditQueryPatch(Patch):
    """Replace query in search[query] at step_t with new_query. Valid only if action at t is search."""

    def __init__(self, step_t: int, new_query: str) -> None:
        super().__init__(patch_type="edit_query", step_t=step_t, payload=new_query)

    def apply(self, actions: List[str]) -> List[str]:
        out = list(actions)
        if 0 <= self.step_t < len(out):
            prefix, _ = _parse_search_action(out[self.step_t])
            if prefix is not None:
                out[self.step_t] = f"search[{self.payload}]"
        return out

    def cost(self) -> float:
        return 1.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditQueryPatch":
        return cls(step_t=d["step_t"], new_query=d["payload"])


def is_search_action(action: str) -> bool:
    """True if action is search[query]."""
    return action.strip().lower().startswith("search[") and "]" in action
