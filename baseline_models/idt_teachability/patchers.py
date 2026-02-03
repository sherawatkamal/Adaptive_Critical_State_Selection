#!/usr/bin/env python3
"""
Patchers: propose a "patch action" to try at a candidate mistake step.

In your current EEF simulator, you patch by *restarting from a state* and letting the agent
explore (softmax). This file adds "action patching": force an alternative first action
at the chosen step, then continue with the same agent policy.

This is meant to support the "teachable moments" research question:
- Where should we intervene? (step selection)
- What should we change? (patch action selection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any
import random
import re


@dataclass
class PatchProposal:
    action: str
    score: float = 0.0
    meta: Optional[Dict[str, Any]] = None


class BasePatcher:
    """Abstract patcher interface."""
    name: str = "base"

    def propose(
        self,
        obs: str,
        goal: str,
        valid_actions: Sequence[str],
        *,
        original_action: Optional[str] = None,
        agent=None,
        k: int = 5,
    ) -> List[PatchProposal]:
        raise NotImplementedError


class RandomPatcher(BasePatcher):
    """Randomly propose k actions (excluding the original action if present)."""
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def propose(
        self,
        obs: str,
        goal: str,
        valid_actions: Sequence[str],
        *,
        original_action: Optional[str] = None,
        agent=None,
        k: int = 5,
    ) -> List[PatchProposal]:
        cands = list(valid_actions)
        if original_action in cands:
            cands.remove(original_action)
        self._rng.shuffle(cands)
        cands = cands[: max(0, k)]
        return [PatchProposal(action=a, score=0.0, meta={"ranker": "random"}) for a in cands]


class AgentTopKPatcher(BasePatcher):
    """
    Use the student's policy distribution to propose top-k alternative actions.

    Motivation:
      - If the corrective action is "surprising but plausible", it should be high-ish rank
        under the student but not necessarily top-1 (RSR-style intuition).
      - Practically: cheap, no extra models.
    """
    name = "agent_topk"

    def propose(
        self,
        obs: str,
        goal: str,
        valid_actions: Sequence[str],
        *,
        original_action: Optional[str] = None,
        agent=None,
        k: int = 5,
    ) -> List[PatchProposal]:
        if agent is None:
            raise ValueError("AgentTopKPatcher requires `agent`.")
        if not valid_actions:
            return []

        # For search states (generative), agent.get_action_probs returns None in your pipeline.
        probs = agent.get_action_probs(obs, list(valid_actions))
        if probs is None:
            # Fall back: just return the first k valid actions (excluding original)
            out: List[PatchProposal] = []
            for a in valid_actions:
                if a == original_action:
                    continue
                out.append(PatchProposal(action=a, score=0.0, meta={"ranker": "fallback_no_probs"}))
                if len(out) >= k:
                    break
            return out

        # Sort by probability descending
        prob_list = probs.detach().cpu().tolist()
        idxs = sorted(range(len(valid_actions)), key=lambda i: prob_list[i], reverse=True)

        out: List[PatchProposal] = []
        for i in idxs:
            a = valid_actions[i]
            if a == original_action:
                continue
            out.append(PatchProposal(action=a, score=float(prob_list[i]), meta={"ranker": "agent_topk", "p": float(prob_list[i]), "idx": int(i)}))
            if len(out) >= k:
                break
        return out


class DiagnosisTextPatcher(BasePatcher):
    """
    Parse a diagnosis model's natural-language response to extract an action string.

    If your diagnosis model is prompted to "optionally explain which action should have been taken instead",
    we can try to grab `click[...]` or `search[...]` from its response.

    Note: This patcher only proposes *one* action (k ignored), because the diagnosis response is usually single-action.
    """
    name = "diagnosis_text"

    _ACTION_RE = re.compile(r'(click\[[^\]]+\]|search\[[^\]]+\])', re.IGNORECASE)

    def __init__(self, require_in_valid: bool = True):
        self.require_in_valid = require_in_valid

    def propose(
        self,
        obs: str,
        goal: str,
        valid_actions: Sequence[str],
        *,
        original_action: Optional[str] = None,
        agent=None,
        k: int = 5,
        diagnosis_response: Optional[str] = None,
    ) -> List[PatchProposal]:
        if not diagnosis_response:
            return []

        m = self._ACTION_RE.search(diagnosis_response)
        if not m:
            return []
        action = m.group(1)

        # Normalize capitalization to match env actions (often lowercase 'click[')
        action = action.strip()
        action = action[0:4].lower() + action[4:] if action.lower().startswith("click") else action
        action = action[0:6].lower() + action[6:] if action.lower().startswith("search") else action

        if self.require_in_valid and (action not in valid_actions):
            return []

        if action == original_action:
            return []

        return [PatchProposal(action=action, score=1.0, meta={"ranker": "diagnosis_text"})]


def make_patcher(name: str, *, seed: Optional[int] = None) -> BasePatcher:
    name = name.lower().strip()
    if name == "random":
        return RandomPatcher(seed=seed)
    if name in ("agent_topk", "topk"):
        return AgentTopKPatcher()
    if name in ("diagnosis_text", "diag_text"):
        return DiagnosisTextPatcher()
    raise ValueError(f"Unknown patcher: {name}")
