#!/usr/bin/env python3
"""Compute recovery depth labels (d_expert, d_force) for labeled snapshots.

Depth labels are computed per-trajectory and attached to each snapshot.
This script expects that leverage labels already exist (L_local/L_upper),
since depth is derived from leverage patterns over time.

Input/Output use the current `LabeledSnapshot.to_dict()` schema.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label.depth import group_snapshots_by_trajectory, compute_depth_for_all_trajectories
from src.utils.common import setup_logging

from scripts.phase1._labeling_io import load_labeled_snapshots, save_both


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute depth labels")
    p.add_argument("--input", type=Path, required=True, help="Labeled snapshots (.json/.jsonl)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--recovery-threshold",
        type=float,
        default=0.5,
        help="Treat a state as 'recoverable' if p_force/p_expert >= this",
    )
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)

    items = load_labeled_snapshots(args.input, max_items=args.max_items)
    logger.info("Loaded %d items", len(items))

    grouped = group_snapshots_by_trajectory(items)
    depths = compute_depth_for_all_trajectories(grouped, recovery_threshold=float(args.recovery_threshold))

    n_with_depth = 0
    for ls in items:
        ls.depth = depths.get(ls.snapshot.id)
        if ls.depth is not None:
            n_with_depth += 1

    logger.info(
        "Computed depth for %d trajectories; depth attached to %d/%d snapshots",
        len(grouped),
        n_with_depth,
        len(items),
    )

    jsonl_path, json_path = save_both(items, args.output_dir, stem="labeled_snapshots")
    logger.info("Wrote %s and %s", jsonl_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
