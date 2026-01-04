"""H4 Figure: Transfer matrix (train quadrant → eval quadrant).

Reads:
  results/phase3/v8/transfer_matrix.csv

By default, plots supervision='demo' only.

Writes:
  <output-dir>/fig_h4_transfer_matrix_<supervision>.png
  <output-dir>/fig_h4_transfer_matrix_<supervision>.pdf

Matrix entries are Δ success vs BASE on the failure-panel snapshots of the eval quadrant.
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
    ap.add_argument("--supervision", type=str, default="demo", choices=["demo", "contrast", "hint", "best"])

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = args.eval_dir or (args.results_dir / "phase3" / "v8")
    path = eval_dir / "transfer_matrix.csv"
    if not path.exists():
        raise SystemExit(f"Missing: {path}")

    rows = load_csv(path)
    quadrants = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]

    # Build matrix: train x eval
    mat = np.full((len(quadrants), len(quadrants)), np.nan)

    for i, tq in enumerate(quadrants):
        for j, eq in enumerate(quadrants):
            candidates = []
            for r in rows:
                if r.get("train_quadrant") != tq or r.get("eval_quadrant") != eq:
                    continue
                sup = r.get("supervision")
                if args.supervision != "best" and sup != args.supervision:
                    continue
                try:
                    candidates.append(float(r.get("delta_success")))
                except Exception:
                    pass
            if not candidates:
                continue
            mat[i, j] = max(candidates) if args.supervision == "best" else float(np.mean(candidates))

    fig = plt.figure(figsize=(5.5, 4.0))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Δ success vs BASE")

    plt.xticks(range(len(quadrants)), [q.split("_")[0] for q in quadrants])
    plt.yticks(range(len(quadrants)), [q.split("_")[0] for q in quadrants])
    plt.xlabel("Eval quadrant")
    plt.ylabel("Train quadrant")
    plt.title(f"Transfer matrix (supervision={args.supervision})")

    out_png = args.output_dir / f"fig_h4_transfer_matrix_{args.supervision}.png"
    out_pdf = args.output_dir / f"fig_h4_transfer_matrix_{args.supervision}.pdf"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)

    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
