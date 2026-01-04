#!/usr/bin/env python3
"""
Phase 4: Train teachability predictor (v8 record schema).

Goal
----
Train an end-to-end predictor that approximates teachability / ELP from cheap observables.

This implementation is deliberately pragmatic for deadlines:
- Uses **Tier-1 structural features** already computed in Phase 1
- Appends **cheap UQ metrics** (entropy, margin, top_action_prob, n_actions)
- Adds a **Tier-2 text embedding** (default: hashing embedder for speed / no extra deps)
- Trains the multitask predictor in `src/predictor/*` with ELP + leverage + quadrant heads

Inputs
------
- Phase-1 labeled records JSON from `scripts/phase1/build_dataset.py`

Outputs
-------
- output_dir/predictor/   (saved predictor weights + config)
- output_dir/training_result.json
- output_dir/feature_schema.json   (documents feature layout)

Example
-------
python scripts/phase4/train_predictor.py \
  --input results/phase1/labeled_snapshots.json \
  --output-dir results/phase4 \
  --embedder hashing --embedding-dim 256 \
  --epochs 50
"""

import sys
import os
# Allow running as a script from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils import setup_logging, save_json, set_seed
from src.features.tier1_structural import StructuralFeatures
from src.features.tier2_embeddings import create_embedder
from src.predictor.multitask_model import TeachabilityPredictor, PredictorConfig
from src.predictor.training import TrainingConfig, prepare_labels, train_predictor


logger = logging.getLogger(__name__)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _uq_vec(rec: Dict[str, Any]) -> np.ndarray:
    uq = rec.get("uncertainty") or rec.get("uncertainty_features") or {}
    # Prefer flat U if present
    entropy = float(rec.get("U", uq.get("entropy", 0.0)))
    margin = float(uq.get("margin", 0.0))
    top_p = float(uq.get("top_action_prob", 0.0))
    n_actions = float(uq.get("n_actions", 0.0))
    return np.array([entropy, margin, top_p, n_actions], dtype=np.float32)


def _struct_vec(rec: Dict[str, Any], include_uq: bool = True) -> np.ndarray:
    sf = rec.get("structural_features")
    if isinstance(sf, dict):
        # Rehydrate and vectorize
        s = StructuralFeatures(
            step_index=int(sf.get("step_index", 0)),
            trajectory_length=int(sf.get("trajectory_length", 1)),
            relative_position=float(sf.get("relative_position", 0.0)),
            n_available_actions=int(sf.get("n_available_actions", 0)),
            action_space_entropy=float(sf.get("action_space_entropy", 0.0)),
            state_length=int(sf.get("state_length", 0)),
            n_numeric_tokens=int(sf.get("n_numeric_tokens", 0)),
            n_product_mentions=int(sf.get("n_product_mentions", 0)),
            task_complexity=float(sf.get("task_complexity", 0.0)),
            n_constraints=int(sf.get("n_constraints", 0)),
            n_prior_failures=int(sf.get("n_prior_failures", 0)),
            steps_since_last_success=int(sf.get("steps_since_last_success", 0)),
            uncertainty_bin=sf.get("uncertainty_bin"),
            leverage_bin=sf.get("leverage_bin"),
            quadrant=sf.get("quadrant"),
        )
        base = s.to_vector().astype(np.float32)
    else:
        # Fallback: zeros if missing
        base = np.zeros(12, dtype=np.float32)

    if include_uq:
        return np.concatenate([base, _uq_vec(rec)], axis=0)
    return base


def _embed_text(rec: Dict[str, Any]) -> str:
    snap = rec.get("snapshot", {}) or {}
    instr = (rec.get("instruction_text") or "").strip()
    if not instr:
        # Common v8 mining stores the task text in agent_prefix
        instr = (snap.get("instruction_text") or snap.get("agent_prefix") or "").strip()
    obs = (snap.get("observation") or "")
    # Keep embeddings cheap: truncate
    obs = obs[:1500]
    if not instr and isinstance(obs, str) and obs.startswith("[WebShop] Task:"):
        # Best-effort parse from observation header
        first_line = obs.splitlines()[0]
        instr = first_line.replace("[WebShop] Task:", "").strip()
    text = f"Task: {instr}\n\n{obs}".strip()
    return text

def main() -> None:
    ap = argparse.ArgumentParser(description="Train teachability predictor (v8)")
    ap.add_argument("--input", type=Path, required=True, help="Phase-1 labeled records JSON")
    ap.add_argument("--output-dir", type=Path, default=Path("results/phase4"))
    ap.add_argument("--seed", type=int, default=42)

    # Embeddings
    ap.add_argument("--embedder", type=str, default="hashing", choices=["hashing", "sentence_transformer", "mock"])
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--embedding-model", type=str, default=None, help="Model name for sentence_transformer embedder")

    # Predictor config
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training config
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)

    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.input)
    logger.info(f"Loaded {len(records)} records from {args.input}")

    # Build embeddings
    embedder = create_embedder(
        embedder_type=args.embedder,
        model_name=args.embedding_model,
        dim=args.embedding_dim,
    )
    texts = [_embed_text(r) for r in records]
    emb = embedder.embed_batch(texts).astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {emb.shape}")
    if args.embedder in {"hashing", "mock"} and emb.shape[1] != int(args.embedding_dim):
        raise ValueError(
            f"Embedder returned dim={emb.shape[1]} but --embedding-dim={args.embedding_dim}."
        )
    logger.info(f"Embeddings shape: {emb.shape}")

    # Build structural(+UQ) features
    struct = np.stack([_struct_vec(r, include_uq=True) for r in records], axis=0).astype(np.float32)
    logger.info(f"Structural(+UQ) shape: {struct.shape}")

    # Concatenate
    features = np.concatenate([struct, emb], axis=1)

    # Predictor configuration (struct_dim includes UQ)
    predictor_cfg = PredictorConfig(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        predict_uncertainty=False,  # uncertainty is an input feature here
        predict_leverage=True,
        predict_quadrant=True,
        predict_elp=True,
        use_structural=True,
        use_embeddings=True,
        structural_dim=int(struct.shape[1]),
        embedding_dim=int(emb.shape[1]),
    )
    predictor = TeachabilityPredictor(config=predictor_cfg)

    # Labels (use v8 quadrant names)
    quadrant_labels = TeachabilityPredictor.QUADRANT_LABELS
    labels = prepare_labels(records, quadrant_labels=quadrant_labels)

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    result = train_predictor(predictor, features, labels, config=train_cfg)

    # Save artifacts
    predictor_dir = args.output_dir / "predictor"
    predictor.save(str(predictor_dir))
    result.save(str(args.output_dir / "training_result.json"))

    feature_schema = {
        "structural_dim": int(struct.shape[1]),
        "embedding_dim": int(emb.shape[1]),
        "total_dim": int(features.shape[1]),
        "structural_layout": [
            "step_index",
            "trajectory_length",
            "relative_position",
            "n_available_actions",
            "action_space_entropy",
            "state_length",
            "n_numeric_tokens",
            "n_product_mentions",
            "task_complexity",
            "n_constraints",
            "n_prior_failures",
            "steps_since_last_success",
            "U_entropy",
            "margin",
            "top_action_prob",
            "n_actions",
        ],
        "embedder": {
            "type": args.embedder,
            "dim": int(emb.shape[1]),
            "model": args.embedding_model,
        },
        "labels": ["leverage(L_local)", "elp(ELP_net)", "quadrant"],
    }
    save_json(feature_schema, args.output_dir / "feature_schema.json")

    logger.info(f"Saved predictor to {predictor_dir}")
    logger.info(f"Saved training result to {args.output_dir / 'training_result.json'}")


if __name__ == "__main__":
    main()
