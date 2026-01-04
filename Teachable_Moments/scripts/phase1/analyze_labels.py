#!/usr/bin/env python3
"""Analyze Phase-1 labels and write a compact summary JSON.

This replaces the legacy `analyze_labels.py` which expected an older schema.

Input:  labeled snapshots in current `LabeledSnapshot.to_dict()` format.
Output: <output_dir>/label_analysis.json with:
  - counts per quadrant
  - summary stats for U and leverage
  - optional CPT stats if present
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import setup_logging

from scripts.phase1._labeling_io import load_labeled_snapshots


logger = logging.getLogger(__name__)


def _stats(x: list[float]) -> dict:
    if not x:
        return {}
    arr = np.array(x, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze labeled snapshots")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)

    items = load_labeled_snapshots(args.input, max_items=args.max_items)
    logger.info("Loaded %d items", len(items))

    quadrant_counts: dict[str, int] = {}
    U_vals: list[float] = []
    L_local: list[float] = []
    L_upper: list[float] = []
    cpt_deltas: dict[str, list[float]] = {"placebo": [], "demo": [], "contrast": [], "hint": []}

    for ls in items:
        quadrant_counts[ls.quadrant] = quadrant_counts.get(ls.quadrant, 0) + 1
        U_vals.append(float(ls.U))
        if ls.leverage is not None:
            L_local.append(float(ls.leverage.L_local))
            L_upper.append(float(ls.leverage.L_upper))
        if ls.cpt is not None:
            # We store per-condition deltas in CPTLabels
            for k in cpt_deltas.keys():
                v = getattr(ls.cpt, f"delta_{k}", None)
                if v is not None:
                    cpt_deltas[k].append(float(v))

    out = {
        "n_items": len(items),
        "quadrant_counts": quadrant_counts,
        "U": _stats(U_vals),
        "leverage": {
            "L_local": _stats(L_local),
            "L_upper": _stats(L_upper),
        },
        "cpt": {k: _stats(v) for k, v in cpt_deltas.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "label_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
