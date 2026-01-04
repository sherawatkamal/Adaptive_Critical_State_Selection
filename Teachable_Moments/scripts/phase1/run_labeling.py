#!/usr/bin/env python3
"""Phase 1 (v8): Label snapshots into the current LabeledSnapshot schema.

This script replaces the legacy Phase-1 labeling scripts.

Input
-----
`--snapshots` should point to a file containing Snapshot.to_dict() objects.
Accepted formats:
  - JSON list:   [ {...snapshot...}, {...snapshot...}, ... ]
  - JSONL:       one JSON object per line

Output
------
Writes labeled snapshots in **current** `LabeledSnapshot.to_dict()` format:
  - <output_dir>/labeled_snapshots.jsonl (streaming, resumable)
  - <output_dir>/labeled_snapshots.json  (JSON list convenience copy)

Notes
-----
* Leverage and CPT require `env_state_b64` to be present in snapshots.
* Use `--mock-env --mock-policy --mock-teacher` for a fast smoke test.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use exported env vars


# Ensure repo root on path so `import src.*` works when running as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.data.snapshot import Snapshot, LabeledSnapshot
from src.label.uncertainty import compute_all_uncertainty
from src.label.leverage import LeverageConfig, estimate_leverage
from src.label.patch_gain import CPTConfig, run_cpt
from src.label.quadrant import assign_quadrant, compute_thresholds
from src.teacher.client import TeacherConfig, create_teacher_client
from src.teacher.structured_hint import generate_teacher_hint
from src.utils.common import setup_logging, set_seed


logger = logging.getLogger(__name__)


class _MockPolicy:
    """Tiny deterministic policy for smoke tests (no transformers required)."""

    def get_action(self, observation: str, valid_actions: list[str]) -> str:
        return valid_actions[0] if valid_actions else ""

    def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
        if not valid_actions:
            return {}
        p = 1.0 / len(valid_actions)
        return {a: p for a in valid_actions}


class _ModelFactoryPolicy:
    """Adapter exposing the PolicyProtocol expected by labeling modules."""

    def __init__(self, model_factory: Any, model: Any, tokenizer: Any):
        self._factory = model_factory
        self._model = model
        self._tokenizer = tokenizer

    def get_action(self, observation: str, valid_actions: list[str]) -> str:
        action, _probs, _raw = self._factory.decode_action(
            observation=observation,
            valid_actions=valid_actions,
            model=self._model,
            tokenizer=self._tokenizer,
        )
        return action

    def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
        return self._factory.get_action_distribution(
            observation=observation,
            valid_actions=valid_actions,
            model=self._model,
            tokenizer=self._tokenizer,
        )


def _iter_json_objects(path: Path) -> Iterator[dict]:
    """Yield dicts from JSON or JSONL."""
    if path.suffix.lower() == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    # JSON (list or object)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
        return

    # Common wrappers
    if isinstance(data, dict):
        for key in ["snapshots", "items", "data"]:
            if key in data and isinstance(data[key], list):
                for obj in data[key]:
                    if isinstance(obj, dict):
                        yield obj
                return

    raise ValueError(f"Unrecognized snapshot file format: {path}")


def _load_snapshots(path: Path, max_items: Optional[int] = None) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    for i, obj in enumerate(_iter_json_objects(path)):
        if max_items is not None and i >= max_items:
            break
        snapshots.append(Snapshot.from_dict(obj))
    return snapshots


def _read_existing_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    ids: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("snapshot", {}).get("id") or obj.get("snapshot_id")
                if sid:
                    ids.add(sid)
            except Exception:
                continue
    return ids


def _save_json_list_from_jsonl(jsonl_path: Path, json_path: Path) -> None:
    items: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    with open(json_path, "w") as f:
        json.dump(items, f, indent=2)


def _load_policy_from_checkpoint(checkpoint: str, device: str = "auto") -> _ModelFactoryPolicy:
    """Load a policy via ModelFactory.

    This is intentionally lazy-importing transformers dependencies.
    """
    from src.utils.model_factory import ModelFactory, ModelConfig

    cfg = ModelConfig.from_checkpoint(checkpoint)
    cfg.device_map = device
    factory = ModelFactory(cfg)
    model, tokenizer = factory.load()
    return _ModelFactoryPolicy(factory, model, tokenizer)


def _build_env_factory(max_steps: int, mock_env: bool):
    from src.utils.model_factory import create_env_factory

    return create_env_factory(mock=mock_env, max_steps=max_steps)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: label snapshots into LabeledSnapshot schema")

    p.add_argument("--snapshots", type=Path, required=True, help="Input snapshots (.json or .jsonl)")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write labeled outputs")
    p.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"), help="Experiment config")

    # Policies
    p.add_argument("--student-checkpoint", type=str, default=None, help="Student checkpoint path")
    p.add_argument("--expert-checkpoint", type=str, default=None, help="Expert checkpoint path (optional)")
    p.add_argument("--device", type=str, default="auto", help="Device for model inference")

    # Mocks (for smoke tests / CI)
    p.add_argument("--mock-env", action="store_true", help="Use MockWebShopEnv")
    p.add_argument("--mock-teacher", action="store_true", help="Use MockTeacherClient")
    p.add_argument("--mock-policy", action="store_true", help="Use a deterministic mock policy")

    # Budget / debug
    p.add_argument("--max-snapshots", type=int, default=None, help="Only process first N snapshots")
    p.add_argument("--resume", action="store_true", help="Resume from existing labeled_snapshots.jsonl")
    p.add_argument("--skip-leverage", action="store_true", help="Skip leverage labeling")
    p.add_argument("--skip-cpt", action="store_true", help="Skip CPT labeling")
    p.add_argument("--skip-teacher", action="store_true", help="Skip teacher hint generation")

    p.add_argument(
        "--primary-uncertainty",
        type=str,
        default=None,
        help="Override which uncertainty feature populates scalar U (e.g., entropy, margin, gini, topk_spread)",
    )

    # Quadrants
    p.add_argument("--assign-quadrants", action="store_true", help="Assign quadrants at end (auto thresholds)")
    p.add_argument("--U-threshold", type=float, default=None, help="Manual U threshold")
    p.add_argument("--L-threshold", type=float, default=None, help="Manual L_local threshold")
    p.add_argument(
        "--threshold-method",
        type=str,
        default="median",
        choices=["median", "percentile_75"],
        help="Threshold heuristic if manual thresholds not provided",
    )

    # Repro / logging
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _configs_from_yaml(cfg: dict) -> tuple[int, str, LeverageConfig, CPTConfig, TeacherConfig]:
    """Map `configs/experiment.yaml` into module configs.

    The repo's YAML uses a nested structure under `labeling:`.
    The core labeling modules take their own dataclass configs.
    """
    max_steps = int(cfg.get("environment", {}).get("max_steps", 15))

    labeling = cfg.get("labeling", {}) or {}
    unc = labeling.get("uncertainty", {}) or {}
    lev = labeling.get("leverage", {}) or {}
    cpt = labeling.get("cpt", {}) or {}
    teacher = cfg.get("teacher", {}) or {}

    uncertainty_primary = str(unc.get("primary", unc.get("primary_estimator", "entropy")))

    leverage_cfg = LeverageConfig(
        n_force_rollouts=int(lev.get("n_rollouts_A", lev.get("n_force_rollouts", 7))),
        n_expert_rollouts=int(lev.get("n_rollouts_B", lev.get("n_expert_rollouts", 2))),
        max_steps=int(lev.get("max_steps", max_steps)),
    )

    cpt_cfg = CPTConfig(
        n_per_condition=int(cpt.get("n_per_condition", 2)),
        max_steps=int(cpt.get("max_steps", max_steps)),
    )

    teacher_cfg = TeacherConfig(
        model=str(teacher.get("model", "gpt-4o")),
        temperature=float(teacher.get("temperature", 0.7)),
        max_tokens=int(teacher.get("max_tokens", 512)),
        cache_dir=str(teacher.get("cache_dir", ".teacher_cache")),
    )

    return max_steps, uncertainty_primary, leverage_cfg, cpt_cfg, teacher_cfg


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "labeled_snapshots.jsonl"
    out_json = args.output_dir / "labeled_snapshots.json"
    meta_path = args.output_dir / "labeling_metadata.json"

    # Load yaml config (optional, but we still run with defaults if missing)
    cfg_dict: dict = {}
    if args.config and args.config.exists():
        cfg_dict = _load_yaml(args.config) or {}
    max_steps, uncertainty_primary, leverage_cfg, cpt_cfg, teacher_cfg = _configs_from_yaml(cfg_dict)
    if args.primary_uncertainty is not None:
        uncertainty_primary = args.primary_uncertainty

    logger.info("Loading snapshots from %s", args.snapshots)
    snapshots = _load_snapshots(args.snapshots, max_items=args.max_snapshots)
    logger.info("Loaded %d snapshots", len(snapshots))

    # Resume support
    done_ids: set[str] = set()
    if args.resume:
        done_ids = _read_existing_ids(out_jsonl)
        if done_ids:
            logger.info("Resume enabled: found %d already-labeled snapshots", len(done_ids))

    # Policies
    if args.mock_policy:
        student_policy: Any = _MockPolicy()
        expert_policy: Any = _MockPolicy()
        logger.info("Using mock policies")
    else:
        if not args.student_checkpoint:
            raise SystemExit("--student-checkpoint is required unless --mock-policy is set")
        student_policy = _load_policy_from_checkpoint(args.student_checkpoint, device=args.device)
        if args.expert_checkpoint:
            expert_policy = _load_policy_from_checkpoint(args.expert_checkpoint, device=args.device)
        else:
            expert_policy = student_policy
            logger.warning("No --expert-checkpoint provided; using student as expert (L_upper will be weak)")

    # Env factory
    env_factory = _build_env_factory(max_steps=max_steps, mock_env=args.mock_env)

    # Teacher client
    teacher_client = None
    if not args.skip_teacher:
        teacher_client = create_teacher_client(teacher_cfg, mock=args.mock_teacher)

    # Metadata (write early)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "input": {
                    "snapshots": str(args.snapshots),
                    "max_snapshots": args.max_snapshots,
                },
                "policy": {
                    "student_checkpoint": args.student_checkpoint,
                    "expert_checkpoint": args.expert_checkpoint,
                    "mock_policy": bool(args.mock_policy),
                },
                "env": {"mock_env": bool(args.mock_env), "max_steps": max_steps},
                "teacher": {
                    "mock_teacher": bool(args.mock_teacher),
                    "enabled": not args.skip_teacher,
                    "config": asdict(teacher_cfg),
                },
                "labeling": {
                    "leverage": asdict(leverage_cfg),
                    "cpt": asdict(cpt_cfg),
                    "skip_leverage": bool(args.skip_leverage),
                    "skip_cpt": bool(args.skip_cpt),
                },
                "quadrants": {
                    "assign": bool(args.assign_quadrants),
                    "method": args.threshold_method,
                    "U_threshold": args.U_threshold,
                    "L_threshold": args.L_threshold,
                },
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    n_total = len(snapshots)
    n_written = 0
    labeled_cache: list[LabeledSnapshot] = []  # only used if assigning quadrants

    # Stream output
    mode = "a" if (args.resume and out_jsonl.exists()) else "w"
    with open(out_jsonl, mode) as f:
        for idx, snap in enumerate(snapshots, start=1):
            if snap.id in done_ids:
                continue

            # Teacher hint
            if (not args.skip_teacher) and (snap.teacher_hint is None):
                try:
                    assert teacher_client is not None
                    snap.teacher_hint = generate_teacher_hint(
                        teacher_client=teacher_client,
                        instruction_text="",  # best-effort; include in agent_prefix if you have it
                        observation=snap.observation,
                        valid_actions=snap.valid_actions,
                        student_action=snap.last_action,
                    )
                except Exception as e:
                    logger.warning("Teacher hint failed for %s: %s", snap.id, e)

            # Uncertainty
            unc = compute_all_uncertainty(student_policy, snap.observation, snap.valid_actions)
            # Primary U (default: entropy)
            U = float(unc.get(uncertainty_primary, unc.get("entropy", 0.0)))

            # Leverage
            lev = None
            if not args.skip_leverage:
                if snap.env_state_bytes is None:
                    logger.warning("Snapshot %s missing env_state_b64; skipping leverage", snap.id)
                elif snap.teacher_hint is None:
                    logger.warning("Snapshot %s missing teacher_hint; skipping leverage", snap.id)
                else:
                    env = env_factory()
                    try:
                        lev = estimate_leverage(
                            snapshot=snap,
                            env=env,
                            student_policy=student_policy,
                            expert_policy=expert_policy,
                            teacher_hint=snap.teacher_hint,
                            config=leverage_cfg,
                        )
                    finally:
                        if hasattr(env, "close"):
                            env.close()

            # CPT
            cpt = None
            if not args.skip_cpt:
                if snap.env_state_bytes is None:
                    logger.warning("Snapshot %s missing env_state_b64; skipping CPT", snap.id)
                elif snap.teacher_hint is None:
                    logger.warning("Snapshot %s missing teacher_hint; skipping CPT", snap.id)
                else:
                    env = env_factory()
                    try:
                        cpt = run_cpt(
                            snapshot=snap,
                            env=env,
                            student_policy=student_policy,
                            teacher_hint=snap.teacher_hint,
                            config=cpt_cfg,
                        )
                    finally:
                        if hasattr(env, "close"):
                            env.close()

            labeled = LabeledSnapshot(
                snapshot=snap,
                U=U,
                uncertainty_features=unc,
                leverage=lev,
                cpt=cpt,
                depth=None,
                quadrant="UNASSIGNED",  # (re)assign via scripts/phase1/assign_quadrants.py
                held_out=False,
                split="train",
            )

            # We only keep in memory if assigning quadrants at end
            if args.assign_quadrants:
                labeled_cache.append(labeled)

            f.write(json.dumps(labeled.to_dict()) + "\n")
            n_written += 1

            if idx % 25 == 0 or idx == n_total:
                logger.info("Labeled %d/%d (wrote %d)", idx, n_total, n_written)

    # Assign quadrants (optional)
    if args.assign_quadrants:
        # Compute thresholds from labeled_cache (only those with leverage)
        snap_dicts = [ls.to_dict() for ls in labeled_cache if ls.leverage is not None]
        if snap_dicts:
            U_thr, L_thr = compute_thresholds(snap_dicts, method=args.threshold_method)
            if args.U_threshold is not None:
                U_thr = float(args.U_threshold)
            if args.L_threshold is not None:
                L_thr = float(args.L_threshold)

            # Re-write jsonl with quadrants updated (single pass)
            tmp_path = out_jsonl.with_suffix(".tmp.jsonl")
            with open(out_jsonl) as fin, open(tmp_path, "w") as fout:
                for line in fin:
                    obj = json.loads(line)
                    lev_obj = obj.get("leverage")
                    if not lev_obj:
                        obj["quadrant"] = "UNASSIGNED"
                    else:
                        U_val = float(obj.get("U", 0.0))
                        L_local = float(lev_obj.get("L_local", 0.0))
                        obj["quadrant"] = assign_quadrant(U_val, L_local, U_thr, L_thr)
                    fout.write(json.dumps(obj) + "\n")
            tmp_path.replace(out_jsonl)

            with open(args.output_dir / "quadrant_thresholds_auto.json", "w") as f:
                json.dump({"U_threshold": U_thr, "L_threshold": L_thr, "method": args.threshold_method}, f, indent=2)
            logger.info("Assigned quadrants using U=%.4f, L=%.4f", U_thr, L_thr)
        else:
            logger.warning("No leverage labels found; skipping quadrant assignment")

    # Convenience JSON list output
    _save_json_list_from_jsonl(out_jsonl, out_json)
    logger.info("Wrote %s and %s", out_jsonl, out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
