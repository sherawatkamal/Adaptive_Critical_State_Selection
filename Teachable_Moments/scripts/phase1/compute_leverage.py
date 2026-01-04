#!/usr/bin/env python3
"""Compute leverage (L_local, L_upper) for snapshots in current schema.

Reads either Snapshot dicts or LabeledSnapshot dicts and writes updated
LabeledSnapshot outputs with the `leverage` field filled.

Leverage requires:
  - snapshot.env_state_b64 present (restorable state)
  - teacher_hint present (for forced action)

If teacher_hint is missing you can run `scripts/phase1/generate_hints.py` first
or use `scripts/phase1/run_labeling.py` (end-to-end).
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

from src.label.leverage import LeverageConfig, estimate_leverage
from src.utils.common import setup_logging, set_seed

from scripts.phase1._labeling_io import load_labeled_snapshots, save_both


logger = logging.getLogger(__name__)


class _MockPolicy:
    def get_action(self, observation: str, valid_actions: list[str]) -> str:
        return valid_actions[0] if valid_actions else ""


class _ModelFactoryPolicy:
    def __init__(self, model_factory: Any, model: Any, tokenizer: Any):
        self._factory = model_factory
        self._model = model
        self._tokenizer = tokenizer

    def get_action(self, observation: str, valid_actions: list[str]) -> str:
        action, _, _ = self._factory.decode_action(
            observation=observation,
            valid_actions=valid_actions,
            model=self._model,
            tokenizer=self._tokenizer,
        )
        return action


def _load_policy(checkpoint: str, device: str) -> _ModelFactoryPolicy:
    from src.utils.model_factory import ModelFactory, ModelConfig

    cfg = ModelConfig.from_checkpoint(checkpoint)
    cfg.device = device
    factory = ModelFactory(cfg)
    model, tokenizer = factory.load()
    return _ModelFactoryPolicy(factory, model, tokenizer)


def _build_env_factory(max_steps: int, mock_env: bool):
    from src.utils.model_factory import create_env_factory

    return create_env_factory(mock=mock_env, max_steps=max_steps)


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _configs_from_yaml(cfg: dict) -> tuple[int, LeverageConfig]:
    max_steps = int(cfg.get("environment", {}).get("max_steps", 15))
    labeling = cfg.get("labeling", {}) or {}
    lev = labeling.get("leverage", {}) or {}
    leverage_cfg = LeverageConfig(
        n_force_rollouts=int(lev.get("n_rollouts_A", lev.get("n_force_rollouts", 7))),
        n_expert_rollouts=int(lev.get("n_rollouts_B", lev.get("n_expert_rollouts", 2))),
        max_steps=int(lev.get("max_steps", max_steps)),
    )
    return max_steps, leverage_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute leverage labels")
    p.add_argument("--input", type=Path, required=True, help="Snapshots or labeled snapshots (.json/.jsonl)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))

    p.add_argument("--student-checkpoint", type=str, default=None)
    p.add_argument("--expert-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--mock-env", action="store_true")
    p.add_argument("--mock-policy", action="store_true")

    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    cfg_dict: dict = {}
    if args.config and args.config.exists():
        cfg_dict = _load_yaml(args.config) or {}
    max_steps, leverage_cfg = _configs_from_yaml(cfg_dict)

    items = load_labeled_snapshots(args.input, max_items=args.max_items)
    logger.info("Loaded %d items", len(items))

    if args.mock_policy:
        student_policy: Any = _MockPolicy()
        expert_policy: Any = _MockPolicy()
        logger.info("Using mock policies")
    else:
        if not args.student_checkpoint:
            raise SystemExit("--student-checkpoint required unless --mock-policy")
        student_policy = _load_policy(args.student_checkpoint, device=args.device)
        if args.expert_checkpoint:
            expert_policy = _load_policy(args.expert_checkpoint, device=args.device)
        else:
            expert_policy = student_policy
            logger.warning("No --expert-checkpoint provided; using student as expert (weak L_upper)")

    env_factory = _build_env_factory(max_steps=max_steps, mock_env=args.mock_env)

    n_ok = 0
    for ls in items:
        snap = ls.snapshot
        if snap.env_state_bytes is None:
            logger.warning("%s missing env_state_b64; skipping leverage", snap.id)
            continue
        if snap.teacher_hint is None:
            logger.warning("%s missing teacher_hint; skipping leverage", snap.id)
            continue

        env = env_factory()
        try:
            ls.leverage = estimate_leverage(
                snapshot=snap,
                env=env,
                student_policy=student_policy,
                expert_policy=expert_policy,
                teacher_hint=snap.teacher_hint,
                config=leverage_cfg,
            )
            n_ok += 1
        finally:
            if hasattr(env, "close"):
                env.close()

    logger.info("Computed leverage for %d/%d items", n_ok, len(items))

    jsonl_path, json_path = save_both(items, args.output_dir, stem="labeled_snapshots")
    logger.info("Wrote %s and %s", jsonl_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
