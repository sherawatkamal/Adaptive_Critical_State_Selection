"""H1 Figure: Teachability landscape (U vs L_local, colored by ELP_net).

Reads:
  results/phase1/labeled_snapshots.jsonl (or --labeled)
Writes:
  <output-dir>/fig_h1_teachability_landscape.png
  <output-dir>/fig_h1_teachability_landscape.pdf

This figure is a cornerstone for the narrative:
(1) define teachable moments as high-uncertainty & high-leverage
(2) show ELP_net structure across the plane
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("figures/v8"))
    ap.add_argument("--labeled", type=Path, default=None, help="Path to labeled_snapshots.jsonl")
    ap.add_argument("--max-points", type=int, default=5000, help="Downsample for plotting")

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = args.labeled or (args.results_dir / "phase1" / "labeled_snapshots.jsonl")
    if not labeled_path.exists():
        raise SystemExit(f"Missing labeled snapshots: {labeled_path}")

    recs = load_jsonl(labeled_path)
    logger.info(f"Loaded {len(recs)} labeled snapshots")

    rows = []
    for r in recs:
        try:
            U = float(r.get("U"))
            L = float((r.get("leverage") or {}).get("L_local"))
            E = float((r.get("cpt") or {}).get("ELP_net"))
            q = str(r.get("quadrant") or "")
            rows.append((U, L, E, q))
        except Exception:
            continue

    if not rows:
        raise SystemExit("No valid (U, L_local, ELP_net) triples found")

    # Downsample deterministically (head) to avoid heavy plots
    if len(rows) > args.max_points:
        rows = rows[: args.max_points]

    U_vals = [x[0] for x in rows]
    L_vals = [x[1] for x in rows]
    E_vals = [x[2] for x in rows]

    # median thresholds for quadrant boundaries
    import numpy as np

    U_th = float(np.median(U_vals))
    L_th = float(np.median(L_vals))

    fig = plt.figure(figsize=(6.5, 5.0))
    sc = plt.scatter(U_vals, L_vals, c=E_vals, s=10, alpha=0.7)
    plt.axvline(U_th, linestyle="--", linewidth=1)
    plt.axhline(L_th, linestyle="--", linewidth=1)

    plt.xlabel("Uncertainty U")
    plt.ylabel("Leverage L_local")
    plt.title("Teachability landscape (colored by ELP_net)")
    cb = plt.colorbar(sc)
    cb.set_label("ELP_net")

    out_png = args.output_dir / "fig_h1_teachability_landscape.png"
    out_pdf = args.output_dir / "fig_h1_teachability_landscape.pdf"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)

    logger.info(f"Saved: {out_png}")
    logger.info(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
