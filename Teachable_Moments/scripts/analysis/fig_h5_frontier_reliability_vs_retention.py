"""H5 Figure: Reliability–Retention frontier.

We plot, for each trained run:
  x = retention_drop  (negative is good; positive = forgetting)
  y = failure_panel_gain = mean_success(failure-panel) - mean_success_BASE(failure-panel)

Reads (from Phase-3 eval output dir):
  overall_task_results.csv
  per_quadrant_results.csv
  retention_results.csv

Writes:
  <output-dir>/fig_h5_frontier_reliability_vs_retention.png
  <output-dir>/fig_h5_frontier_reliability_vs_retention.pdf
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

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
    ap.add_argument("--output-dir", type=Path, default=Path("figures/v8"))

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = args.eval_dir or (args.results_dir / "phase3" / "v8")

    p_overall = eval_dir / "overall_task_results.csv"
    p_panel = eval_dir / "per_quadrant_results.csv"
    p_ret = eval_dir / "retention_results.csv"

    for p in [p_overall, p_panel, p_ret]:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    overall = load_csv(p_overall)
    panel = load_csv(p_panel)
    ret = load_csv(p_ret)

    # Failure-panel mean success by model
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
    base_panel = panel_mean.get("BASE")
    if base_panel is None:
        raise SystemExit("BASE row missing from per_quadrant_results.csv")

    ret_drop = {r.get("model"): float(r.get("retention_drop")) for r in ret if r.get("model")}

    xs = []
    ys = []
    labels = []
    for m, pm in panel_mean.items():
        if m == "BASE":
            continue
        if m not in ret_drop:
            continue
        xs.append(ret_drop[m])
        ys.append(pm - base_panel)
        labels.append(m)

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.scatter(xs, ys, s=50, alpha=0.8)

    # label lightly (top few by gain)
    if labels:
        order = np.argsort(ys)[::-1]
        for idx in order[: min(12, len(order))]:
            plt.annotate(labels[idx], (xs[idx], ys[idx]), fontsize=8, alpha=0.8)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("Retention drop (fine-tuned − base) on base-success tasks")
    plt.ylabel("Failure-panel gain (Δ success vs BASE)")
    plt.title("Reliability vs retention frontier")

    out_png = args.output_dir / "fig_h5_frontier_reliability_vs_retention.png"
    out_pdf = args.output_dir / "fig_h5_frontier_reliability_vs_retention.pdf"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)

    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
