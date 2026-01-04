"""Shared I/O helpers for Phase-1 labeling scripts.

All Phase-1 scripts should read/write the **current** Snapshot/LabeledSnapshot
schema from `src.data.snapshot`.

We support both JSON and JSONL inputs/outputs:
  - JSON:  a list of objects
  - JSONL: one JSON object per line
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, Optional

# Ensure repo root on path so `import src.*` works when these helpers are used
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.snapshot import Snapshot, LabeledSnapshot


def iter_json_objects(path: Path) -> Iterator[dict]:
    """Yield dicts from JSON or JSONL."""
    if path.suffix.lower() == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
        return

    if isinstance(data, dict):
        # common wrappers
        for key in ("labeled_snapshots", "snapshots", "items", "data"):
            if key in data and isinstance(data[key], list):
                for obj in data[key]:
                    if isinstance(obj, dict):
                        yield obj
                return

    raise ValueError(f"Unrecognized JSON format at {path}")


def load_labeled_snapshots(path: Path, max_items: Optional[int] = None) -> list[LabeledSnapshot]:
    """Load either LabeledSnapshot dicts or raw Snapshot dicts.

    If the file contains Snapshot dicts, we wrap them into LabeledSnapshot with
    placeholder values.
    """
    out: list[LabeledSnapshot] = []
    for i, obj in enumerate(iter_json_objects(path)):
        if max_items is not None and i >= max_items:
            break

        if "snapshot" in obj:
            out.append(LabeledSnapshot.from_dict(obj))
        else:
            snap = Snapshot.from_dict(obj)
            out.append(
                LabeledSnapshot(
                    snapshot=snap,
                    U=0.0,
                    uncertainty_features={},
                    leverage=None,
                    cpt=None,
                    depth=None,
                    quadrant="UNASSIGNED",
                    held_out=False,
                    split="train",
                )
            )
    return out


def save_jsonl(items: list[LabeledSnapshot], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


def save_json(items: list[LabeledSnapshot], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([it.to_dict() for it in items], f, indent=2)


def save_both(items: list[LabeledSnapshot], output_dir: Path, stem: str = "labeled_snapshots") -> tuple[Path, Path]:
    """Write both JSONL and JSON list outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{stem}.jsonl"
    json_path = output_dir / f"{stem}.json"
    save_jsonl(items, jsonl_path)
    save_json(items, json_path)
    return jsonl_path, json_path
