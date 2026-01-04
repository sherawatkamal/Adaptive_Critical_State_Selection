"""Phase 2 (v8): Train the per-quadrant training matrix.

Why this script exists
----------------------
The repo contains older, pre-v8 training scripts that assume a different data schema.
`src/training/per_quadrant.py` is the v8 training engine (14 runs: 4 quadrants × 3 formats + 2 baselines),
but many orchestration scripts (pilots, experimental plan) want a single CLI entrypoint.

This script provides that stable entrypoint.

Inputs
------
- labeled snapshots (`.jsonl` produced by `scripts/phase1/run_labeling.py`)

Outputs (in --output-dir)
-------------------------
- training_matrix.json
- training_summary.json  (mapping run_id -> model checkpoint dir)
- training_meta.json     (metadata needed by eval scripts)
- <run_id>/...           (trained LoRA adapters + training logs)

Example
-------
python scripts/phase2/run_training_matrix_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --config configs/experiment.yaml \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --output-dir results/phase2/v8 \
  --run-ids Q1_highU_highL_demo B1_uniform \
  --max-train-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_labeled_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    # JSON list
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def _partition_by_quadrant(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    parts: Dict[str, List[Dict[str, Any]]] = {
        "Q1_highU_highL": [],
        "Q2_highU_lowL": [],
        "Q3_lowU_lowL": [],
        "Q4_lowU_highL": [],
    }
    for r in records:
        q = r.get("quadrant")
        if q in parts:
            parts[q].append(r)
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-quadrant matrix (v8)")
    ap.add_argument("--labeled-snapshots", type=Path, required=True)
    ap.add_argument("--config", type=str, required=True, help="YAML config (uses per_quadrant_training)")
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/phase2/v8"))
    ap.add_argument("--dry-run", action="store_true", help="Write matrix/metadata only (no training)")

    # Deadline-friendly knobs
    ap.add_argument("--max-train-samples", type=int, default=None, help="Downsample each run to at most N snapshots")
    ap.add_argument("--baseline-n-samples", type=int, default=None, help="# snapshots for B1_uniform baseline")

    ap.add_argument("--run-ids", nargs="+", default=None, help="Optional subset of run_ids to train")

    args = ap.parse_args()
    setup_logging()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_labeled_records(args.labeled_snapshots)
    logger.info(f"Loaded {len(records)} labeled records")

    parts = _partition_by_quadrant(records)
    for q, items in parts.items():
        logger.info(f"{q}: {len(items)} records")

    from src.training.per_quadrant import TrainingMatrix

    training_matrix = TrainingMatrix.create_default()
    available = {r["run_id"] for r in training_matrix.to_dict()}

    run_filter = None
    run_filter_fn = None
    if args.run_ids:
        run_filter = list(args.run_ids)
        missing = [rid for rid in run_filter if rid not in available]
        if missing:
            raise SystemExit(f"Unknown run_ids: {missing}. Available: {sorted(available)}")

        wanted = set(run_filter)

        def _filter_run(run_dict: Dict[str, Any]) -> bool:
            return run_dict.get("run_id") in wanted

        run_filter_fn = _filter_run

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        training_matrix.save(str(args.output_dir / "training_matrix.json"))

        with open(args.output_dir / "training_summary.json", "w") as f:
            json.dump({"n_runs": 0, "n_completed": 0, "model_paths": {}}, f, indent=2)

        meta = {
            "base_model": args.base_model,
            "labeled_snapshots": str(args.labeled_snapshots),
            "config_yaml": args.config,
            "max_train_samples": args.max_train_samples,
            "baseline_n_samples": args.baseline_n_samples,
            "run_filter": run_filter,
            "dry_run": True,
        }
        with open(args.output_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Dry run complete. Wrote training_matrix.json to %s", args.output_dir)
        return

    from src.training.per_quadrant import run_all_training
    from src.training.sft_trainer import SFTConfig

    # Build training config
    sft_cfg = SFTConfig.from_yaml(args.config).to_dict()
    sft_cfg["base_model"] = args.base_model
    if args.max_train_samples is not None:
        sft_cfg["max_train_samples"] = int(args.max_train_samples)
    if args.baseline_n_samples is not None:
        sft_cfg["baseline_n_samples"] = int(args.baseline_n_samples)

    summary = run_all_training(
        quadrant_partitions=parts,
        base_model_path=args.base_model,
        sft_config=sft_cfg,
        output_dir=str(args.output_dir),
        run_filter=run_filter_fn,
    )

    # Extra metadata file for eval/analysis
    meta = {
        "base_model": args.base_model,
        "labeled_snapshots": str(args.labeled_snapshots),
        "config_yaml": args.config,
        "max_train_samples": args.max_train_samples,
        "baseline_n_samples": args.baseline_n_samples,
        "run_filter": run_filter,
    }
    with open(args.output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Training complete. Wrote summary with {len(summary)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
