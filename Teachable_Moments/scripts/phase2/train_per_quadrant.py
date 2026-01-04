#!/usr/bin/env python3
"""
Phase 2: Train per-quadrant models (v8 schema).

This script runs the main experiment matrix:
  4 quadrants × 3 supervision formats + 2 baselines = 14 runs

It is intentionally thin: it delegates the heavy lifting to
`src/training/per_quadrant.py`.

Inputs
------
- Phase-1 labeled records JSON produced by `scripts/phase1/build_dataset.py`
  Each element is a record with keys: snapshot, teacher_hint, quadrant, ...

Outputs
-------
- A directory with one subdir per training run, containing the LoRA adapter checkpoints.
- training_matrix.json describing the runs

Example
-------
python scripts/phase2/train_per_quadrant.py \
  --input results/phase1/labeled_snapshots.json \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --output-dir results/phase2/models \
  --n-parallel 1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
import os
# Allow running as a script from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import setup_logging, set_seed
from src.training.per_quadrant import run_all_training, QUADRANTS, SUPERVISION_TYPES


logger = logging.getLogger(__name__)



def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def partition_by_quadrant(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {q: [] for q in QUADRANTS}
    for r in records:
        q = r.get("quadrant")
        if q in out:
            out[q].append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-quadrant model matrix (v8)")
    ap.add_argument("--input", type=Path, required=True, help="Phase-1 labeled records JSON")
    ap.add_argument("--base-model", type=str, required=True, help="Base model name/path (HF)")
    ap.add_argument("--output-dir", type=Path, default=Path("results/phase2/models"))
    ap.add_argument("--n-parallel", type=int, default=1)

    # Optional filters
    ap.add_argument("--quadrants", nargs="+", default=None, help=f"Subset of quadrants (default all): {QUADRANTS}")
    ap.add_argument("--supervision", nargs="+", default=None, help=f"Subset of supervision types (default all): {SUPERVISION_TYPES}")

    # Optional SFT config overrides (kept minimal)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--target-modules", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    records = load_records(args.input)
    logger.info(f"Loaded {len(records)} records from {args.input}")

    parts = partition_by_quadrant(records)
    for q in QUADRANTS:
        logger.info(f"  {q}: {len(parts[q])}")

    # Run filter for training matrix
    quadrant_filter = set(args.quadrants) if args.quadrants else None
    supervision_filter = set(args.supervision) if args.supervision else None

    def _run_filter(run_cfg: Dict[str, Any]) -> bool:
        q_ok = True
        s_ok = True
        if quadrant_filter and run_cfg.get("quadrant") != "all":
            q_ok = run_cfg.get("quadrant") in quadrant_filter
        if supervision_filter:
            s_ok = run_cfg.get("supervision") in supervision_filter
        return q_ok and s_ok

    sft_cfg = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    
    if args.target_modules:
        sft_cfg["target_modules"] = args.target_modules

    run_all_training(
        quadrant_partitions=parts,
        base_model_path=args.base_model,
        output_dir=str(args.output_dir),
        n_parallel=args.n_parallel,
        sft_config=sft_cfg,
        run_filter=_run_filter if (quadrant_filter or supervision_filter) else None,
    )


if __name__ == "__main__":
    main()
