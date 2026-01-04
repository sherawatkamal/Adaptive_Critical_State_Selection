"""H3 Table: Best supervision format per quadrant (by in-quadrant Δ success).

Reads:
  results/phase3/v8/per_quadrant_results.csv
Writes:
  <output-dir>/table_h3_best_supervision.csv

For each quadrant, selects the format {demo, contrast, hint} that yields the
largest Δ success vs BASE when evaluated on snapshots from the same quadrant.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_run(run_id: str) -> Tuple[str, str]:
    if run_id.startswith("Q"):
        parts = run_id.split("_")
        supervision = parts[-1]
        train_quadrant = "_".join(parts[:-1])
        return train_quadrant, supervision
    return "UNKNOWN", "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--eval-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("tables/v8"))

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = args.eval_dir or (args.results_dir / "phase3" / "v8")
    per_q_path = eval_dir / "per_quadrant_results.csv"
    if not per_q_path.exists():
        raise SystemExit(f"Missing: {per_q_path}")

    rows = load_csv(per_q_path)

    base_by_q: Dict[str, float] = {}
    for r in rows:
        if r.get("model") == "BASE":
            base_by_q[r.get("quadrant")] = float(r.get("success_rate"))

    quadrants = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
    formats = ["demo", "contrast", "hint"]

    best_rows = []
    for q in quadrants:
        best_fmt = None
        best_delta = -1e9
        best_abs = np.nan
        for r in rows:
            model = r.get("model")
            if not model:
                continue
            train_q, fmt = parse_run(model)
            if train_q != q or fmt not in formats:
                continue
            if r.get("quadrant") != q:
                continue
            base = base_by_q.get(q)
            if base is None:
                continue
            abs_sr = float(r.get("success_rate"))
            delta = abs_sr - base
            if delta > best_delta:
                best_delta = delta
                best_fmt = fmt
                best_abs = abs_sr

        best_rows.append(
            {
                "quadrant": q,
                "best_format": best_fmt or "",
                "success_rate": best_abs,
                "delta_vs_base": best_delta if best_fmt is not None else np.nan,
            }
        )

    out_csv = args.output_dir / "table_h3_best_supervision.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quadrant", "best_format", "success_rate", "delta_vs_base"])
        w.writeheader()
        for r in best_rows:
            w.writerow(r)

    logger.info(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
