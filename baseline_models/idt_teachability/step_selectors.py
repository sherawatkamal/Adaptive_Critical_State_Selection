#!/usr/bin/env python3
"""
Step selectors for IDT experiments.

Select which steps in a failed trajectory to evaluate as candidate "mistake" steps.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional, Any


def select_steps_last_n(trajectory: Dict[str, Any], M: int = 8) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Select last M steps (before terminal)."""
    steps = trajectory.get("steps", [])
    T = len(steps)
    if T <= 1:
        return [], []
    max_valid = T - 2  # exclude last (terminal)
    indices = list(range(max(0, max_valid - M + 1), max_valid + 1))
    indices = indices[-M:] if len(indices) > M else indices
    info = [{"state_idx": i, "method": "last_n"} for i in indices]
    return indices, info


def select_steps_search_only(trajectory: Dict[str, Any], M: int = 10) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Select only steps where valid_actions include search[...]."""
    steps = trajectory.get("steps", [])
    T = len(steps)
    if T <= 1:
        return [], []
    indices = []
    for i in range(T - 1):
        va = steps[i].get("valid_actions", [])
        if va and isinstance(va[0], str) and va[0].startswith("search["):
            indices.append(i)
    indices = indices[:M]
    info = [{"state_idx": i, "method": "search_steps"} for i in indices]
    return indices, info


def select_steps_random(
    trajectory: Dict[str, Any],
    M: int = 3,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Randomly select M steps (same count as diagnosis typically)."""
    steps = trajectory.get("steps", [])
    T = len(steps)
    if T <= 1:
        return [], []
    max_valid = T - 2
    candidates = list(range(max_valid + 1))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    indices = sorted(candidates[:M])
    info = [{"state_idx": i, "method": "random"} for i in indices]
    return indices, info
