#!/usr/bin/env python3
"""
Visualize the U (uncertainty) and L (leverage) distributions to choose quadrant thresholds.

Input: labeled snapshots JSON (list of LabeledSnapshot.to_dict dicts)
       produced by scripts/phase1/run_labeling.py or label_snapshots_v2.py

Output: PNGs in --outdir and a JSON summary of quantiles (optional).

Example:
  python scripts/phase1/plot_ul_distribution.py \
    --labeled results/phase1/labeled_snapshots.json \
    --outdir results/phase1/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="Input labeled snapshots JSON")
    ap.add_argument("--outdir", required=True, help="Output directory for plots")
    ap.add_argument("--write-quantiles", action="store_true", help="Write quantiles.json next to plots")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)

    labeled = _load_json(Path(args.labeled))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    U_vals = []
    L_vals = []
    for d in labeled:
        if not isinstance(d, dict):
            continue
        U = d.get("U", None)
        lev = d.get("leverage", {}) or {}
        L = lev.get("L_local", None)
        if U is None or L is None:
            continue
        U_vals.append(float(U))
        L_vals.append(float(L))

    if not U_vals:
        raise SystemExit("No (U,L) pairs found in labeled file.")

    U = np.array(U_vals)
    L = np.array(L_vals)

    def q(arr):
        return {
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    qU = q(U)
    qL = q(L)

    print("U quantiles:", qU)
    print("L quantiles:", qL)

    # Histogram U
    plt.figure()
    plt.hist(U, bins=40)
    plt.xlabel("Uncertainty U (entropy)")
    plt.ylabel("Count")
    plt.title("Distribution of Uncertainty U")
    plt.tight_layout()
    plt.savefig(outdir / "hist_U.png", dpi=200)
    plt.close()

    # Histogram L
    plt.figure()
    plt.hist(L, bins=40)
    plt.xlabel("Leverage L_local")
    plt.ylabel("Count")
    plt.title("Distribution of Leverage L_local")
    plt.tight_layout()
    plt.savefig(outdir / "hist_L.png", dpi=200)
    plt.close()

    # Scatter U vs L
    plt.figure()
    plt.scatter(U, L, s=6, alpha=0.5)
    plt.xlabel("Uncertainty U (entropy)")
    plt.ylabel("Leverage L_local")
    plt.title("U vs L_local scatter")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_U_L.png", dpi=200)
    plt.close()

    if args.write_quantiles:
        out = {
            "U": qU,
            "L_local": qL,
            "suggested_thresholds": {
                "median": {"U": qU["median"], "L_local": qL["median"]},
                "p75": {"U": qU["p75"], "L_local": qL["p75"]},
            },
        }
        with open(outdir / "quantiles.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote quantiles -> {outdir / 'quantiles.json'}")

    print(f"Wrote plots -> {outdir}")


if __name__ == "__main__":
    main()
