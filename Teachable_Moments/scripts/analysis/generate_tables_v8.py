"""Generate all v8 paper tables.

This script is intentionally *thin*: it calls individual table scripts.

Usage:
  python scripts/analysis/generate_tables_v8.py --results-dir results --output-dir tables/v8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TABLE_SCRIPTS = [
    "scripts/analysis/table_h1_quadrant_summary.py",
    "scripts/analysis/table_h3_best_supervision.py",
    "scripts/analysis/table_h5_main_results.py",
    "scripts/analysis/table_h6_predictor_metrics.py",
    "scripts/analysis/table_h7_selection_summary.py",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("tables/v8"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for script in TABLE_SCRIPTS:
        cmd = [sys.executable, script, "--results-dir", str(args.results_dir), "--output-dir", str(args.output_dir)]
        print("\n[TABLE]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
