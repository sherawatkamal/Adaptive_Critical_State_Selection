"""H7 Figure: selection frontier (predictor vs baselines).

Plots a 2D frontier:
  x-axis: retention drop (fine-tuned − base) on base-success tasks
  y-axis: failure-panel gain (Δ success vs BASE)

This mirrors H5 but filters to selection-run IDs (SEL_*).

Inputs (from eval suite output dir):
  - per_quadrant_results.csv
  - retention_results.csv

Output:
  - <output-dir>/fig_h7_selection_frontier.png
  - <output-dir>/fig_h7_selection_frontier.pdf
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure H7: selection frontier")
    ap.add_argument("--eval-dir", type=Path, default=Path("results/phase5/v8"))
    ap.add_argument("--output-dir", type=Path, default=Path("figures/v8"))
    ap.add_argument(
        "--prefix",
        type=str,
        default="SEL_",
        help="Run-id prefix to include (default SEL_)",
    )
    args = ap.parse_args()
    setup_logging()

    p_panel = args.eval_dir / "per_quadrant_results.csv"
    p_ret = args.eval_dir / "retention_results.csv"
    if not p_panel.exists() or not p_ret.exists():
        raise SystemExit(f"Missing eval outputs in {args.eval_dir}. Expected per_quadrant_results.csv and retention_results.csv")

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

    # Filter models
    models = [m for m in panel_mean.keys() if m != "BASE" and m.startswith(args.prefix) and m in ret_drop]
    if not models:
        raise SystemExit(f"No models found with prefix {args.prefix} in {args.eval_dir}")

    xs = [ret_drop[m] for m in models]
    ys = [panel_mean[m] - base_panel for m in models]

    # Friendly labels
    pretty = {
        "SEL_random": "random",
        "SEL_entropy": "uncertainty",
        "SEL_predictor": "predictor",
        "SEL_oracle": "oracle",
    }
    labels = [pretty.get(m, m) for m in models]

    fig = plt.figure(figsize=(7.0, 4.5))
    plt.scatter(xs, ys, s=70, alpha=0.85)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), fontsize=9, alpha=0.9)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Retention drop (fine-tuned − base) on base-success tasks")
    plt.ylabel("Failure-panel gain (Δ success vs BASE)")
    plt.title("Selection frontier: reliability gain vs retention")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / "fig_h7_selection_frontier.png"
    out_pdf = args.output_dir / "fig_h7_selection_frontier.pdf"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)

    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
