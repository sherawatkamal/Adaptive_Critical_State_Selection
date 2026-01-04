"""H3 Figure: In-quadrant improvement heatmap.

Uses Phase-3 eval output (v8):
  results/phase3/v8/per_quadrant_results.csv

Produces a 4×3 heatmap (quadrant × supervision format) of Δ success rate vs BASE
when evaluating on *failure-panel snapshots from the same quadrant*.

Writes:
  <output-dir>/fig_h3_in_quadrant_heatmap.png
  <output-dir>/fig_h3_in_quadrant_heatmap.pdf
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_run(run_id: str) -> Tuple[str, str]:
    """Return (train_quadrant, supervision) for v8 run_ids."""
    if run_id.startswith("Q"):
        parts = run_id.split("_")
        # supervision is last token
        supervision = parts[-1]
        train_quadrant = "_".join(parts[:-1])
        return train_quadrant, supervision
    if run_id.startswith("B1"):
        return "ALL", "baseline_uniform"
    if run_id.startswith("B2"):
        return "ALL", "baseline_all"
    return "UNKNOWN", "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--eval-dir", type=Path, default=None, help="Directory containing per_quadrant_results.csv")
    ap.add_argument("--output-dir", type=Path, default=Path("figures/v8"))

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = args.eval_dir or (args.results_dir / "phase3" / "v8")
    per_q_path = eval_dir / "per_quadrant_results.csv"
    if not per_q_path.exists():
        raise SystemExit(f"Missing: {per_q_path}")

    rows = load_csv(per_q_path)

    # base success per eval quadrant
    base_by_q = {}
    for r in rows:
        if r.get("model") == "BASE":
            base_by_q[r.get("quadrant")] = float(r.get("success_rate"))

    quadrants = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
    formats = ["demo", "contrast", "hint"]

    grid = np.full((len(quadrants), len(formats)), np.nan)

    for r in rows:
        model = r.get("model")
        q_eval = r.get("quadrant")
        if not model or not q_eval:
            continue
        train_q, supervision = parse_run(model)
        if train_q not in quadrants:
            continue
        if q_eval != train_q:
            continue
        if supervision not in formats:
            continue
        base = base_by_q.get(train_q)
        if base is None:
            continue
        delta = float(r.get("success_rate")) - float(base)
        i = quadrants.index(train_q)
        j = formats.index(supervision)
        grid[i, j] = delta

    fig = plt.figure(figsize=(6.0, 3.0))
    im = plt.imshow(grid, aspect="auto")
    plt.colorbar(im, label="Δ success vs BASE")

    plt.xticks(range(len(formats)), formats)
    plt.yticks(range(len(quadrants)), [q.split("_")[0] for q in quadrants])
    plt.title("In-quadrant improvement (failure-panel snapshots)")

    out_png = args.output_dir / "fig_h3_in_quadrant_heatmap.png"
    out_pdf = args.output_dir / "fig_h3_in_quadrant_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)

    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
