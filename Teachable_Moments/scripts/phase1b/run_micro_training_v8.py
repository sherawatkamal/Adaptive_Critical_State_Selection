#!/usr/bin/env python3
"""
Phase 1b: Micro-training validation for CPT (v8 record schema).

This script validates that CPT's cheap scores (ELP_net / route_net) correlate with
actual fine-tuning improvement measured via **1-2 LoRA gradient steps**.

Inputs
------
- labeled_snapshots.json from `scripts/phase1/build_dataset.py`
  (a list of *records* with keys: snapshot, teacher_hint, cpt, quadrant, ...)
- optional panel.json (stratified panel), with a list under "panel" containing ids

Outputs
-------
Writes into --output-dir:
- micro_training_results.json
- cpt_correlation.json   (format used by scripts/analysis/phase1b_figure_cpt_correlation.py)

Notes on rollouts / noise
-------------------------
- Evaluation from the failure snapshot uses cfg.n_validation_rollouts rollouts and reports a success *rate*.
- For deterministic policies (temperature=0, do_sample=False), multiple rollouts will be identical,
  but we keep the interface because WebShop / policies may be stochastic in practice.

Typical usage
-------------
python scripts/phase1b/run_micro_training_v8.py \
  --labeled results/phase1/labeled_snapshots.json \
  --panel results/phase1/panel.json \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --output-dir results/phase1b \
  --n-steps 2 \
  --n-validation-rollouts 3 \
  --rollout-max-steps 15

Optional retention:
  --retention-tasks panels/retention_tasks.json
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running as a script from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.training.micro_trainer import MicroTrainingConfig, evaluate_micro_training
from src.supervision.format_router import generate_supervision_single
from src.data.webshop_env import create_env, WebShopConfig
from src.utils.model_factory import ModelFactory, ModelConfig


logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _extract_panel_ids(panel_obj: Any) -> List[str]:
    """
    Accept a few common formats:
    - {"panel":[{"id":...}, ...]}
    - {"panel":["id1","id2",...]}
    - ["id1","id2",...]
    """
    if panel_obj is None:
        return []
    if isinstance(panel_obj, list):
        # either list[str] or list[dict]
        out = []
        for x in panel_obj:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) and "id" in x:
                out.append(str(x["id"]))
        return out
    if isinstance(panel_obj, dict):
        p = panel_obj.get("panel")
        return _extract_panel_ids(p)
    return []


def _stratified_sample_by_quadrant(
    records: List[Dict[str, Any]],
    n_per_quadrant: int,
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        q = r.get("quadrant", "unknown")
        by_q.setdefault(q, []).append(r)

    sampled: List[Dict[str, Any]] = []
    for q, items in sorted(by_q.items()):
        if len(items) <= n_per_quadrant:
            sampled.extend(items)
        else:
            sampled.extend(random.sample(items, n_per_quadrant))
    return sampled


def _make_micro_sample(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Phase-1 record into the minimal dict expected by micro_trainer.evaluate_micro_training:
      {prompt, completion, env_state_b64, snapshot_id, quadrant, ...}
    """
    snap = record.get("snapshot", {}) or {}
    env_state_b64 = snap.get("env_state_b64")

    # Route-matched supervision based on CPT
    cpt = record.get("cpt", {}) or {}
    route = cpt.get("route_net") or "demo"

    ex = generate_supervision_single(record, format=route)

    return {
        "snapshot_id": snap.get("id", record.get("id", "")),
        "quadrant": record.get("quadrant", ""),
        "task_id": record.get("task_id", snap.get("task_id", "")),
        "route_net": route,
        "cpt_elp_net": float(cpt.get("ELP_net", 0.0)),
        "prompt": ex.input_text,
        "completion": ex.output_text,
        "env_state_b64": env_state_b64,
    }


def _pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Return (r, p) using scipy if available, else numpy-only fallback p=None."""
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        # Fallback: compute r only
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        if len(x_arr) < 2:
            return 0.0, 1.0
        r = np.corrcoef(x_arr, y_arr)[0, 1]
        return float(r), 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1b micro-training validation (v8 schema)")
    ap.add_argument("--labeled", type=Path, required=True, help="Phase-1 labeled records JSON")
    ap.add_argument("--panel", type=Path, default=None, help="Optional panel.json for validation subset")
    ap.add_argument("--output-dir", type=Path, default=Path("results/phase1b"), help="Output directory")
    ap.add_argument("--base-model", type=str, required=True, help="Base model name or path")

    # Selection
    ap.add_argument("--n-per-quadrant", type=int, default=50, help="If no panel provided, stratified samples per quadrant")

    # Micro-training knobs
    ap.add_argument("--n-steps", type=int, default=2, help="LoRA gradient steps per sample")
    ap.add_argument("--learning-rate", type=float, default=1e-4, help="LoRA LR for micro-training")
    ap.add_argument("--max-seq-length", type=int, default=1024, help="Max tokens for micro-training batch")
    ap.add_argument("--lora-target-modules", nargs="+", default=None, help="Target modules for LoRA")

    # Rollout evaluation knobs
    ap.add_argument("--n-validation-rollouts", type=int, default=3, help="Rollouts per (base/tuned) eval from snapshot")
    ap.add_argument("--rollout-max-steps", type=int, default=15, help="Max env steps per eval rollout")
    ap.add_argument("--mock-env", action="store_true", help="Use mock environment (for smoke tests only)")

    # Optional retention evaluation
    ap.add_argument("--retention-tasks", type=Path, default=None, help="JSON list of task_ids for retention eval")
    ap.add_argument("--n-retention", type=int, default=10, help="Number of retention tasks to use (subsample)")

    # Repro
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    records = _load_json(args.labeled)
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {args.labeled}, got {type(records)}")
    logger.info(f"Loaded {len(records)} labeled records from {args.labeled}")

    # Select subset
    selected_records: List[Dict[str, Any]] = []
    if args.panel is not None and args.panel.exists():
        panel_obj = _load_json(args.panel)
        panel_ids = set(_extract_panel_ids(panel_obj))
        if not panel_ids:
            logger.warning(f"Panel file had no ids; falling back to stratified sampling: {args.panel}")
        else:
            selected_records = [r for r in records if (r.get('snapshot', {}).get('id') in panel_ids) or (r.get('id') in panel_ids)]
            logger.info(f"Selected {len(selected_records)} records via panel ids from {args.panel}")

    if not selected_records:
        selected_records = _stratified_sample_by_quadrant(records, args.n_per_quadrant, args.seed)
        logger.info(f"Selected {len(selected_records)} records via stratified sampling (n_per_quadrant={args.n_per_quadrant})")

    # Prepare retention tasks
    retention_task_ids: Optional[List[Any]] = None
    if args.retention_tasks and args.retention_tasks.exists():
        raw = _load_json(args.retention_tasks)
        if isinstance(raw, dict):
            raw = raw.get("tasks", raw.get("task_ids", []))
        if not isinstance(raw, list):
            raise ValueError("retention_tasks must be a JSON list or {tasks:[...]}")
        if len(raw) > args.n_retention:
            random.shuffle(raw)
            raw = raw[: args.n_retention]
        retention_task_ids = raw
        logger.info(f"Using {len(retention_task_ids)} retention tasks from {args.retention_tasks}")

    # Build env factory
    env_cfg = WebShopConfig(max_steps=args.rollout_max_steps)
    def _env_factory():
        return create_env(env_cfg, mock=args.mock_env)

    # Create a single env instance for snapshot evaluations (set_state reuse)
    env = _env_factory()

    # Build base model + LoRA shell
    model_cfg = ModelConfig(
        model_path=args.base_model,
        temperature=0.0,
        do_sample=False,
        merge_lora=False,
    )
    factory = ModelFactory(model_cfg)
    base_model, tokenizer = factory.load()

    # Create fresh LoRA adapter on top of base model
    from peft import get_peft_model, LoraConfig, TaskType
    
    target_modules = args.lora_target_modules
    if not target_modules and "gpt2" in args.base_model:
        target_modules = ["c_attn"]
    elif not target_modules:
        target_modules = ["q_proj", "v_proj"]
        
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()

    # IMPORTANT: ensure the ModelFactory uses the PEFT model for decoding
    factory._model = peft_model  # pylint: disable=protected-access

    micro_cfg = MicroTrainingConfig(
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        n_validation_rollouts=args.n_validation_rollouts,
        rollout_max_steps=args.rollout_max_steps,
        n_validation_tasks=args.n_retention,
    )

    results: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected_records):
        snap_id = (rec.get("snapshot", {}) or {}).get("id", rec.get("id", f"row{idx}"))
        logger.info(f"[{idx+1}/{len(selected_records)}] Micro-training on {snap_id}")

        sample = _make_micro_sample(rec)

        out = evaluate_micro_training(
            env=env,
            peft_model=peft_model,
            tokenizer=tokenizer,
            sample=sample,
            model_factory=factory,
            cfg=micro_cfg,
            retention_task_ids=retention_task_ids,
            env_factory=_env_factory if retention_task_ids else None,
        )

        # Stitch record-level info + outputs
        row = {
            "snapshot_id": sample["snapshot_id"],
            "quadrant": sample["quadrant"],
            "task_id": sample["task_id"],
            "route_net": sample["route_net"],
            "cpt_elp_net": sample["cpt_elp_net"],
            **out,
        }
        results.append(row)

    # Save primary results
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "labeled": str(args.labeled),
            "panel": str(args.panel) if args.panel else None,
            "base_model": args.base_model,
            "micro": micro_cfg.__dict__,
            "seed": args.seed,
            "n_selected": len(selected_records),
            "retention_tasks": str(args.retention_tasks) if args.retention_tasks else None,
        },
        "results": results,
    }
    save_json(output, args.output_dir / "micro_training_results.json")
    logger.info(f"Wrote micro-training results to {args.output_dir / 'micro_training_results.json'}")

    # Build CPT correlation artifact compatible with existing plotting script
    cpt_vals = [float(r.get("cpt_elp_net", 0.0)) for r in results]
    delta_vals = [float(r.get("delta_success_rate", 0.0)) for r in results]

    r, p = _pearson(cpt_vals, delta_vals)

    corr = {
        "pearson_r": r,
        "pearson_p": p,
        "n": len(results),
        "data_points": [
            {
                "snapshot_id": r0.get("snapshot_id"),
                "quadrant": r0.get("quadrant"),
                "cpt": float(r0.get("cpt_elp_net", 0.0)),
                "elp": float(r0.get("delta_success_rate", 0.0)),
                "route_net": r0.get("route_net", ""),
            }
            for r0 in results
        ],
    }
    save_json(corr, args.output_dir / "cpt_correlation.json")
    logger.info(f"Wrote CPT correlation artifact to {args.output_dir / 'cpt_correlation.json'}")

    # Pretty print summary
    print("\n" + "=" * 72)
    print("MICRO-TRAINING SUMMARY")
    print("=" * 72)
    print(f"Samples: {len(results)}")
    print(f"Pearson r(CPT ELP_net, delta_success_rate): {r:.3f} (p={p:.3g})")
    print("Per-quadrant mean delta_success_rate:")
    by_q: Dict[str, List[float]] = {}
    for rr in results:
        by_q.setdefault(rr.get("quadrant", "unknown"), []).append(float(rr.get("delta_success_rate", 0.0)))
    for q, vals in sorted(by_q.items()):
        print(f"  {q}: {np.mean(vals):+.3f}  (n={len(vals)})")
    print("=" * 72)


if __name__ == "__main__":
    main()
