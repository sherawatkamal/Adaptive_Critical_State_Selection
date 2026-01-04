"""H6 Table: Teachability predictor performance metrics.

Reads:
  results/phase4/**/training_result.json (or --training-result)

Writes:
  <output-dir>/table_h6_predictor_metrics.csv

We surface the key metrics needed for the paper story:
- ELP correlation / RMSE
- Route accuracy
- Quadrant accuracy
- Ranking metrics for ELP (precision@K, NDCG@K) if present
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_training_result(results_dir: Path) -> Path | None:
    candidates = [
        results_dir / "phase4" / "predictor" / "training_result.json",
        results_dir / "phase4" / "training_result.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # recursive search (bounded)
    for p in results_dir.glob("phase4/**/training_result.json"):
        return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--training-result", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("tables/v8"))

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tr_path = args.training_result or find_training_result(args.results_dir)
    if tr_path is None or not tr_path.exists():
        raise SystemExit("Could not find training_result.json (use --training-result)")

    data: Dict[str, Any] = json.loads(tr_path.read_text())
    metrics: Dict[str, Any] = data.get("task_metrics", {}) or {}

    flat = {
        "elp_corr": metrics.get("elp_corr"),
        "elp_rmse": metrics.get("elp_rmse"),
        "route_acc": metrics.get("route_acc"),
        "quadrant_acc": metrics.get("quadrant_acc"),
        "elp_spearman": metrics.get("elp_spearman"),
        "elp_precision@10": metrics.get("elp_precision_at_10"),
        "elp_precision@50": metrics.get("elp_precision_at_50"),
        "elp_ndcg@10": metrics.get("elp_ndcg_at_10"),
        "elp_ndcg@50": metrics.get("elp_ndcg_at_50"),
    }

    out_csv = args.output_dir / "table_h6_predictor_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)

    logger.info(f"Read: {tr_path}")
    logger.info(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
