"""
Patch proposer: interface and HeuristicPatchProposer baseline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from idt.patches import (
    EditQueryPatch,
    InsertActionPatch,
    Patch,
    ReplaceActionPatch,
    is_search_action,
)
from idt.types import StepContext, Trajectory


class PatchProposer(ABC):
    """Interface: propose(traj, step_t, context) -> List[Patch]."""

    @abstractmethod
    def propose(
        self,
        traj: Trajectory,
        step_t: int,
        context: StepContext,
        max_candidates_per_type: int = 10,
        seed: int = 0,
    ) -> List[Patch]:
        pass


# Small library of recovery actions (WebShop-style).
RECOVERY_ACTIONS = [
    "click[back to search]",
    "click[description]",
    "click[features]",
    "click[reviews]",
    "click[< prev]",
    "click[next >]",
]


def _extract_constraints_from_instruction(instruction: str) -> List[str]:
    """Cheap string extraction: words after 'with', 'and', commas."""
    tokens: List[str] = []
    for part in re.split(r"[,;]|\band\b|\bwith\b", instruction, flags=re.I):
        part = part.strip()
        if part and len(part) > 2:
            tokens.append(part.lower())
    return tokens[:10]


class HeuristicPatchProposer(PatchProposer):
    """
    Baseline proposer:
    - ReplaceActionPatch: use agent.propose_actions(top_n).
    - InsertActionPatch: small library of recovery actions.
    - EditQueryPatch: if action at t is search[query], propose rewritten queries from instruction.
    Deterministic given seed (agent sampling uses seed).
    """

    def __init__(self, agent: Optional[any] = None) -> None:
        self._agent = agent

    def propose(
        self,
        traj: Trajectory,
        step_t: int,
        context: StepContext,
        max_candidates_per_type: int = 10,
        seed: int = 0,
    ) -> List[Patch]:
        patches: List[Patch] = []
        action_t = context.action_at_t

        # ReplaceActionPatch: alternatives at this step
        info = {"valid": context.valid_actions} if context.valid_actions else None
        if self._agent is not None and step_t < traj.length:
            try:
                candidates = self._agent.propose_actions(
                    context.observation,
                    context.action_history,
                    info=info,
                    top_n=max_candidates_per_type,
                    stochastic=True,
                    seed=seed,
                )
                for c in candidates[:max_candidates_per_type]:
                    if c.action != action_t:
                        patches.append(ReplaceActionPatch(step_t, c.action))
            except Exception:
                pass
        elif step_t < traj.length:
            for rec in RECOVERY_ACTIONS[:max_candidates_per_type]:
                if rec != action_t:
                    patches.append(ReplaceActionPatch(step_t, rec))

        # InsertActionPatch: recovery actions before step_t
        for rec in RECOVERY_ACTIONS[:5]:
            patches.append(InsertActionPatch(step_t, rec))

        # EditQueryPatch: only if action at t is search
        if is_search_action(action_t):
            constraints = _extract_constraints_from_instruction(context.instruction)
            m = re.match(r"search\[(.*)\]", action_t.strip(), re.DOTALL | re.I)
            base_query = m.group(1).strip() if m else ""
            variants = [base_query]
            for c in constraints[:5]:
                if c and c not in base_query.lower():
                    variants.append(f"{base_query} {c}".strip())
            variants = list(dict.fromkeys(variants))[:max_candidates_per_type]
            for v in variants:
                if v != base_query or len(variants) == 1:
                    patches.append(EditQueryPatch(step_t, v))

        return patches
