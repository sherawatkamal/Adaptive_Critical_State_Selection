"""H1 Table: Quadrant summary statistics.

Reads:
  results/phase1/labeled_snapshots.jsonl (or --labeled)
Writes:
  <output-dir>/table_h1_quadrant_summary.csv

Includes:
- count per quadrant
- mean/median U, L_local
- mean/median ELP_net
- fraction of positive ELP_net (teachable)
- distribution of CPT route_net (succeed/salvage/need_hint/dead_end)
"""  

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("tables/v8"))
    ap.add_argument("--labeled", type=Path, default=None)

    args = ap.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = args.labeled or (args.results_dir / "phase1" / "labeled_snapshots.jsonl")
    if not labeled_path.exists():
        raise SystemExit(f"Missing labeled snapshots: {labeled_path}")

    recs = load_jsonl(labeled_path)
    logger.info(f"Loaded {len(recs)} labeled snapshots")

    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        q = r.get("quadrant")
        if not q:
            continue
        by_q[str(q)].append(r)

    rows: List[Dict[str, Any]] = []
    for q in [
        "Q1_highU_highL",
        "Q2_highU_lowL",
        "Q3_lowU_lowL",
        "Q4_lowU_highL",
    ]:
        rs = by_q.get(q, [])
        if not rs:
            continue

        U = np.array([safe_float(r.get("U")) for r in rs], dtype=float)
        L = np.array([safe_float((r.get("leverage") or {}).get("L_local")) for r in rs], dtype=float)
        E = np.array([safe_float((r.get("cpt") or {}).get("ELP_net")) for r in rs], dtype=float)

        # Route distribution
        routes = Counter()
        for r in rs:
            routes[str((r.get("cpt") or {}).get("route_net") or "") or "unknown"] += 1

        def frac(key: str) -> float:
            return routes.get(key, 0) / max(1, len(rs))

        rows.append(
            {
                "quadrant": q,
                "n": len(rs),
                "U_mean": float(np.nanmean(U)),
                "U_median": float(np.nanmedian(U)),
                "L_local_mean": float(np.nanmean(L)),
                "L_local_median": float(np.nanmedian(L)),
                "ELP_net_mean": float(np.nanmean(E)),
                "ELP_net_median": float(np.nanmedian(E)),
                "ELP_pos_rate": float(np.mean(E > 0)),
                "route_succeed_frac": frac("succeed"),
                "route_salvage_frac": frac("salvage"),
                "route_need_hint_frac": frac("need_hint"),
                "route_dead_end_frac": frac("dead_end"),
            }
        )

    out_csv = args.output_dir / "table_h1_quadrant_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    logger.info(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
