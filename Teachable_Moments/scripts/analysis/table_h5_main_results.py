"""H5 Table: Main end-to-end results (overall, failure-panel, retention, stuckness).

Reads (Phase-3 eval output dir):
  overall_task_results.csv
  per_quadrant_results.csv
  retention_results.csv
  stuckness_results.csv

Writes:
  <output-dir>/table_h5_main_results.csv

The default behavior includes all models in the eval run.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--eval-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("tables/v8"))

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = args.eval_dir or (args.results_dir / "phase3" / "v8")

    p_overall = eval_dir / "overall_task_results.csv"
    p_panel = eval_dir / "per_quadrant_results.csv"
    p_ret = eval_dir / "retention_results.csv"
    p_stuck = eval_dir / "stuckness_results.csv"
    for p in [p_overall, p_panel, p_ret, p_stuck]:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    overall = load_csv(p_overall)
    panel = load_csv(p_panel)
    ret = load_csv(p_ret)
    stuck = load_csv(p_stuck)

    overall_by_model = {r["model"]: r for r in overall if r.get("model")}

    # Panel mean
    panel_sums: Dict[str, float] = {}
    panel_counts: Dict[str, int] = {}
    for r in panel:
        m = r.get("model")
        if not m:
            continue
        sr = float(r.get("success_rate"))
        panel_sums[m] = panel_sums.get(m, 0.0) + sr
        panel_counts[m] = panel_counts.get(m, 0) + 1
    panel_mean = {m: panel_sums[m] / max(1, panel_counts[m]) for m in panel_sums}
    base_panel = panel_mean.get("BASE", np.nan)

    ret_by_model = {r["model"]: r for r in ret if r.get("model")}

    # Stuckness (snapshots only)
    stuck_by_model = {r["model"]: r for r in stuck if r.get("model") and r.get("dataset_type") == "snapshots"}

    models = sorted(set(panel_mean.keys()) | set(overall_by_model.keys()))

    out_rows = []
    for m in models:
        o = overall_by_model.get(m, {})
        r = ret_by_model.get(m, {})
        s = stuck_by_model.get(m, {})
        pm = panel_mean.get(m, np.nan)
        out_rows.append(
            {
                "model": m,
                "overall_task_success": float(o.get("success_rate", np.nan)) if o else np.nan,
                "overall_task_mean_steps": float(o.get("mean_steps", np.nan)) if o else np.nan,
                "failure_panel_mean_success": float(pm),
                "failure_panel_gain_vs_base": float(pm - base_panel) if np.isfinite(base_panel) else np.nan,
                "retention_success": float(r.get("retention_success_rate", np.nan)) if r else np.nan,
                "retention_drop": float(r.get("retention_drop", np.nan)) if r else np.nan,
                "repeat_action_rate": float(s.get("repeat_action_rate", np.nan)) if s else np.nan,
                "loop_detected_rate": float(s.get("loop_detected_rate", np.nan)) if s else np.nan,
            }
        )

    out_csv = args.output_dir / "table_h5_main_results.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    logger.info(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
