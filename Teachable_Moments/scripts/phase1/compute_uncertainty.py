#!/usr/bin/env python3
"""Compute uncertainty labels for snapshots in the **current** schema.

This script reads either:
  - raw Snapshot.to_dict() objects, or
  - existing LabeledSnapshot.to_dict() objects

and outputs updated LabeledSnapshot objects with:
  - `uncertainty_features` filled
  - `U` set to the primary metric (entropy)

It is mainly useful if you want to run Phase-1 labeling step-by-step.
For an end-to-end pipeline, use `scripts/phase1/run_labeling.py`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label.uncertainty import compute_all_uncertainty
from src.utils.common import setup_logging, set_seed

from scripts.phase1._labeling_io import load_labeled_snapshots, save_both


logger = logging.getLogger(__name__)


class _MockPolicy:
    def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
        if not valid_actions:
            return {}
        p = 1.0 / len(valid_actions)
        return {a: p for a in valid_actions}


class _ModelFactoryPolicy:
    def __init__(self, model_factory: Any, model: Any, tokenizer: Any):
        self._factory = model_factory
        self._model = model
        self._tokenizer = tokenizer

    def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
        return self._factory.get_action_distribution(
            observation=observation,
            valid_actions=valid_actions,
            model=self._model,
            tokenizer=self._tokenizer,
        )


def _load_policy(checkpoint: str, device: str) -> _ModelFactoryPolicy:
    from src.utils.model_factory import ModelFactory, ModelConfig

    cfg = ModelConfig.from_checkpoint(checkpoint)
    cfg.device = device
    factory = ModelFactory(cfg)
    model, tokenizer = factory.load()
    return _ModelFactoryPolicy(factory, model, tokenizer)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute uncertainty labels")
    p.add_argument("--input", type=Path, required=True, help="Snapshots or labeled snapshots (.json/.jsonl)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--student-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mock-policy", action="store_true")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    items = load_labeled_snapshots(args.input, max_items=args.max_items)
    logger.info("Loaded %d items", len(items))

    if args.mock_policy:
        policy: Any = _MockPolicy()
        logger.info("Using mock policy")
    else:
        if not args.student_checkpoint:
            raise SystemExit("--student-checkpoint is required unless --mock-policy is set")
        policy = _load_policy(args.student_checkpoint, device=args.device)

    for ls in items:
        snap = ls.snapshot
        feats = compute_all_uncertainty(policy, snap.observation, snap.valid_actions)
        ls.uncertainty_features = feats
        ls.U = float(feats.get("entropy", 0.0))

    jsonl_path, json_path = save_both(items, args.output_dir, stem="labeled_snapshots")
    logger.info("Wrote %s and %s", jsonl_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
