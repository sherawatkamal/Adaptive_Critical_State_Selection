"""Generate all v8 paper figures.

This script is intentionally *thin*: it just calls individual figure scripts.
User preference: keep each figure in its own script for inspectability.

Usage:
  python scripts/analysis/generate_figures_v8.py --results-dir results --output-dir figures/v8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FIG_SCRIPTS = [
    "scripts/analysis/fig_h1_teachability_landscape.py",
    "scripts/analysis/fig_h3_in_quadrant_heatmap.py",
    "scripts/analysis/fig_h4_transfer_matrix.py",
    "scripts/analysis/fig_h5_frontier_reliability_vs_retention.py",
    "scripts/analysis/fig_h7_selection_frontier.py",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("figures/v8"))
    ap.add_argument("--supervision", type=str, default="demo", choices=["demo", "contrast", "hint", "best"], help="For transfer-matrix figure")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for script in FIG_SCRIPTS:
        cmd = [sys.executable, script, "--results-dir", str(args.results_dir), "--output-dir", str(args.output_dir)]
        if script.endswith("fig_h4_transfer_matrix.py"):
            cmd += ["--supervision", args.supervision]
        print("\n[FIG]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
