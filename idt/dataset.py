"""
Build and load IDT dataset (JSONL): traj_id, step_t, patch_dict, R1/R3/R5, teachable_label, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from idt.types import PatchSearchResult, Trajectory

logger = logging.getLogger(__name__)


def record_from_search_result(result: PatchSearchResult) -> Dict[str, Any]:
    """Build a single JSONL record from PatchSearchResult."""
    return {
        "traj_id": result.traj_id,
        "task_id": result.task_id,
        "step_t": result.best_step,
        "patch_dict": result.best_patch,
        "patch_cost": None,  # can add from patch.cost() when building
        "patch_type": result.patch_type,
        "R1": result.R1,
        "R3": result.R3,
        "R5": result.R5,
        "teachable_label": result.teachable_label,
        "found_patch": result.found_patch,
        "compute_counters": result.compute_counters,
    }


def save_dataset(path: str | Path, records: List[Dict[str, Any]]) -> None:
    """Append or write JSONL records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load IDT dataset from JSONL."""
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
