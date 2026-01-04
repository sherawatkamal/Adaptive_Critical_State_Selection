#!/usr/bin/env python3
"""Plot U and leverage distributions to choose quadrant thresholds.

This script is intentionally simple and "human-in-the-loop":
  1) it visualizes the distributions of U and L_local
  2) it creates a scatter plot (U vs L_local)
  3) it writes a JSON file with a few suggested threshold candidates

You can then pick thresholds manually and run:
  python scripts/phase1/assign_quadrants.py --U-threshold ... --L-threshold ...

Input: labeled snapshots in current `LabeledSnapshot.to_dict()` schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase1._labeling_io import iter_json_objects


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot distributions for quadrant thresholds")
    p.add_argument("--input", type=Path, required=True, help="labeled_snapshots.json or .jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--show", action="store_true", help="Show figures interactively")
    return p.parse_args()


def _load_UL(path: Path, max_items: int | None) -> tuple[list[float], list[float]]:
    U: list[float] = []
    L: list[float] = []
    for i, obj in enumerate(iter_json_objects(path)):
        if max_items is not None and i >= max_items:
            break
        # obj may be LabeledSnapshot dict or Snapshot dict; we handle labeled only here
        if "snapshot" not in obj:
            continue
        u = obj.get("U", None)
        lev = obj.get("leverage") or {}
        l = lev.get("L_local", None)
        if u is None or l is None:
            continue
        try:
            U.append(float(u))
            L.append(float(l))
        except Exception:
            continue
    return U, L


def _suggest_thresholds(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    U, L = _load_UL(args.input, args.max_items)
    if not U or not L:
        raise SystemExit("No (U, L_local) pairs found. Did you run leverage labeling?")

    U_s = _suggest_thresholds(U)
    L_s = _suggest_thresholds(L)

    # Histogram: U
    fig = plt.figure()
    plt.hist(U, bins=50)
    plt.xlabel("Uncertainty U (entropy)")
    plt.ylabel("Count")
    plt.title("Distribution of U")
    fig_path = args.output_dir / "U_hist.pdf"
    plt.savefig(fig_path)
    if args.show:
        plt.show()
    plt.close(fig)

    # Histogram: L_local
    fig = plt.figure()
    plt.hist(L, bins=50)
    plt.xlabel("Leverage L_local")
    plt.ylabel("Count")
    plt.title("Distribution of L_local")
    fig_path2 = args.output_dir / "L_local_hist.pdf"
    plt.savefig(fig_path2)
    if args.show:
        plt.show()
    plt.close(fig)

    # Scatter
    fig = plt.figure()
    plt.scatter(U, L, s=6, alpha=0.5)
    plt.xlabel("U (entropy)")
    plt.ylabel("L_local")
    plt.title("U vs L_local")
    fig_path3 = args.output_dir / "U_vs_L_local_scatter.pdf"
    plt.savefig(fig_path3)
    if args.show:
        plt.show()
    plt.close(fig)

    # Save suggestions
    out_json = args.output_dir / "threshold_suggestions.json"
    out_json.write_text(
        json.dumps(
            {
                "n_points": len(U),
                "U": U_s,
                "L_local": L_s,
                "note": "Pick thresholds manually, then run scripts/phase1/assign_quadrants.py",
            },
            indent=2,
        )
    )

    print(f"Wrote {fig_path}")
    print(f"Wrote {fig_path2}")
    print(f"Wrote {fig_path3}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
