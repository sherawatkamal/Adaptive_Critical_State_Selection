#!/usr/bin/env python3
"""Assign quadrant labels to (labeled) snapshots.

This script operates on the **current** `LabeledSnapshot.to_dict()` schema.

It expects that each item has:
  - `U` (uncertainty scalar)
  - `leverage.L_local` (actionability)

Threshold selection
------------------
For paper-ready results we recommend:
  1) run Phase1 labeling to compute U and L_local
  2) plot distributions with `scripts/analysis/plot_quadrant_thresholds.py`
  3) choose thresholds manually and pass them here

If you omit thresholds, we use a heuristic (`median` or `percentile_75`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label.quadrant import assign_quadrant, compute_thresholds
from src.utils.common import setup_logging

from scripts.phase1._labeling_io import load_labeled_snapshots, save_both


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign quadrant labels")
    p.add_argument("--input", type=Path, required=True, help="Labeled snapshots (.json/.jsonl)")
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--U-threshold", type=float, default=None)
    p.add_argument("--L-threshold", type=float, default=None)
    p.add_argument(
        "--method",
        type=str,
        default="median",
        choices=["median", "percentile_75"],
        help="Threshold heuristic if manual thresholds not provided",
    )

    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)

    items = load_labeled_snapshots(args.input, max_items=args.max_items)
    logger.info("Loaded %d items", len(items))

    # Only items with leverage can be assigned
    usable = [ls for ls in items if ls.leverage is not None]
    if not usable:
        logger.warning("No leverage labels found; cannot assign quadrants")
        save_both(items, args.output_dir, stem="labeled_snapshots")
        return 0

    # Compute thresholds from usable subset
    as_dicts = [ls.to_dict() for ls in usable]
    U_thr, L_thr = compute_thresholds(as_dicts, method=args.method)
    if args.U_threshold is not None:
        U_thr = float(args.U_threshold)
    if args.L_threshold is not None:
        L_thr = float(args.L_threshold)

    logger.info("Using thresholds: U=%.6f, L_local=%.6f (method=%s)", U_thr, L_thr, args.method)

    counts: dict[str, int] = {}
    for ls in items:
        if ls.leverage is None:
            ls.quadrant = "UNASSIGNED"
            continue
        U_val = float(ls.U)
        L_local = float(ls.leverage.L_local)
        q = assign_quadrant(U_val, L_local, U_thr, L_thr)
        ls.quadrant = q
        counts[q] = counts.get(q, 0) + 1

    jsonl_path, json_path = save_both(items, args.output_dir, stem="labeled_snapshots")
    logger.info("Wrote %s and %s", jsonl_path, json_path)

    thresholds_path = args.output_dir / "quadrant_thresholds.json"
    thresholds_path.write_text(json.dumps({"U_threshold": U_thr, "L_threshold": L_thr, "method": args.method, "counts": counts}, indent=2))
    logger.info("Wrote %s", thresholds_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
