#!/usr/bin/env python3
"""
Assign quadrants to labeled snapshots using MANUAL thresholds.

Input: labeled snapshots JSON (list of LabeledSnapshot.to_dict dicts) with U and leverage.L_local filled.
Output: labeled snapshots JSON with quadrant field assigned.

Example:
  python scripts/phase1/assign_quadrants_manual.py \
    --labeled-in results/phase1/labeled_snapshots_prequad.json \
    --labeled-out results/phase1/labeled_snapshots.json \
    --U-threshold 1.25 \
    --L-threshold 0.10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.label.quadrant import assign_quadrant
from src.utils.common import setup_logging


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-in", required=True)
    ap.add_argument("--labeled-out", required=True)
    ap.add_argument("--U-threshold", type=float, required=True)
    ap.add_argument("--L-threshold", type=float, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)

    labeled = _load_json(Path(args.labeled_in))
    out = []
    assigned = 0

    for d in labeled:
        if not isinstance(d, dict):
            out.append(d)
            continue
        U = d.get("U", None)
        lev = d.get("leverage", {}) or {}
        L = lev.get("L_local", None)

        if U is None or L is None:
            d["quadrant"] = None
            out.append(d)
            continue

        d["quadrant"] = assign_quadrant(float(U), float(L), float(args.U_threshold), float(args.L_threshold))
        assigned += 1
        out.append(d)

    _dump_json(out, Path(args.labeled_out))
    print(f"Assigned quadrants for {assigned} snapshots -> {args.labeled_out}")


if __name__ == "__main__":
    main()
