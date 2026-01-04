#!/usr/bin/env python3
"""Generate / attach teacher hints for snapshots (current schema).

Reads Snapshot or LabeledSnapshot files and writes updated LabeledSnapshot
objects with `snapshot.teacher_hint` filled.

Teacher hints are required for:
  - Leverage (single-step forcing)
  - CPT patch construction

This script uses the TeacherClient cache (disk) to avoid repeat API calls.

If you want an end-to-end run (hints + uncertainty + leverage + CPT), use:
  python scripts/phase1/run_labeling.py ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.teacher.client import TeacherConfig, create_teacher_client
from src.teacher.structured_hint import generate_teacher_hint
from src.utils.common import setup_logging, set_seed

from scripts.phase1._labeling_io import load_labeled_snapshots, save_both


logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _teacher_cfg_from_yaml(cfg: dict) -> TeacherConfig:
    teacher = cfg.get("teacher", {}) or {}
    return TeacherConfig(
        model=str(teacher.get("model", "gpt-4o")),
        temperature=float(teacher.get("temperature", 0.7)),
        max_tokens=int(teacher.get("max_tokens", 512)),
        cache_dir=str(teacher.get("cache_dir", ".teacher_cache")),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate teacher hints")
    p.add_argument("--input", type=Path, required=True, help="Snapshots or labeled snapshots (.json/.jsonl)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))

    p.add_argument("--mock-teacher", action="store_true")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    cfg_dict: dict = {}
    if args.config and args.config.exists():
        cfg_dict = _load_yaml(args.config) or {}
    teacher_cfg = _teacher_cfg_from_yaml(cfg_dict)

    items = load_labeled_snapshots(args.input, max_items=args.max_items)
    logger.info("Loaded %d items", len(items))

    teacher_client = create_teacher_client(teacher_cfg, mock=args.mock_teacher)
    logger.info("Teacher enabled: mock=%s, model=%s", bool(args.mock_teacher), teacher_cfg.model)

    n_new = 0
    for ls in items:
        snap = ls.snapshot
        if snap.teacher_hint is not None:
            continue
        try:
            snap.teacher_hint = generate_teacher_hint(
                teacher_client=teacher_client,
                instruction_text="",  # best-effort: include the goal in agent_prefix if you have it
                observation=snap.observation,
                valid_actions=snap.valid_actions,
                student_action=snap.last_action,
            )
            n_new += 1
        except Exception as e:
            logger.warning("Teacher hint failed for %s: %s", snap.id, e)

    logger.info("Generated %d new hints (others already had hints)", n_new)
    jsonl_path, json_path = save_both(items, args.output_dir, stem="labeled_snapshots")
    logger.info("Wrote %s and %s", jsonl_path, json_path)

    # helpful metadata
    meta_path = args.output_dir / "teacher_hint_metadata.json"
    meta_path.write_text(
        __import__("json").dumps(
            {
                "teacher": asdict(teacher_cfg),
                "mock_teacher": bool(args.mock_teacher),
                "n_items": len(items),
                "n_new_hints": n_new,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
