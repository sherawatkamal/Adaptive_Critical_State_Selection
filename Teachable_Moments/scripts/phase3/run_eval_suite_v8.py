"""Phase 3 (v8): Unified evaluation suite.

This script is the *v8-compatible* evaluation runner.

It intentionally does **not** depend on older phase-3 scripts that assume a
pre-v8 checkpoint layout and/or a different schema.

What it evaluates
-----------------
1) **Overall task performance** on a sampled set of tasks (end-to-end from reset)
2) **Failure-panel performance** on held-out labeled snapshots (restore env state)
3) **Transfer**: train-quadrant -> eval-quadrant matrix (derived from (2))
4) **Retention**: performance on tasks where the base model succeeds (subset of (1))
5) **Stuckness** diagnostics (repeat/loop rates)

Outputs (in --output-dir)
-------------------------
- task_ids.json
- snapshot_panel.json
- overall_task_results.csv
- per_quadrant_results.csv
- transfer_matrix.csv
- retention_results.csv
- stuckness_results.csv
- eval_config.json

Notes
-----
- This suite is designed to be *small-scale friendly* (pilot runs) and also
  scalable (more tasks/snapshots).
- For statistically robust results, you should generate a held-out snapshot panel
  (different from training data) and pass it via --panel-file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.snapshot import Snapshot
from src.data.webshop_env import WebShopConfig, create_env
from src.policies.model_factory_policy import ModelFactoryPolicy
from src.utils.model_factory import ModelConfig, ModelFactory

logger = logging.getLogger(__name__)


# -----------------------------
# Small helper metrics
# -----------------------------

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_training_summary(path: Path) -> Dict[str, str]:
    """Return run_id -> model_path.

    Accepts either:
      {"model_paths": {...}, ...}
    or a raw mapping {run_id: path}.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "model_paths" in data and isinstance(data["model_paths"], dict):
        return {k: v for k, v in data["model_paths"].items() if v}
    if isinstance(data, dict):
        # raw mapping
        return {k: v for k, v in data.items() if isinstance(v, str) and v}
    raise ValueError(f"Unrecognized training summary format in {path}")


def infer_base_model(training_summary_path: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    meta_path = training_summary_path.parent / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        bm = meta.get("base_model")
        if bm:
            return str(bm)
    raise SystemExit(
        "Could not infer base model. Provide --base-model or ensure training_meta.json exists next to training_summary.json"
    )


def parse_run_id(run_id: str) -> Dict[str, str]:
    """Best-effort parsing for run_id -> {train_quadrant, supervision, family}."""
    if run_id.startswith("B1_"):
        return {"family": "baseline", "train_quadrant": "all", "supervision": "demo"}
    if run_id.startswith("B2_"):
        return {"family": "baseline", "train_quadrant": "all", "supervision": "demo"}
    if run_id.startswith("Q"):
        # Supervision is the last token
        parts = run_id.split("_")
        supervision = parts[-1]
        quadrant = "_".join(parts[:-1])
        return {"family": "quadrant", "train_quadrant": quadrant, "supervision": supervision}
    return {"family": "unknown", "train_quadrant": "unknown", "supervision": "unknown"}


def compute_stuckness(actions: List[str]) -> Dict[str, float]:
    if not actions:
        return {"repeat_rate": 0.0, "loop_rate": 0.0, "unique_actions": 0.0}
    n = len(actions)
    unique = len(set(actions))
    repeat_rate = 1.0 - (unique / n)

    # crude loop detection: does any 2-gram repeat 3+ times?
    bigrams = {}
    for i in range(len(actions) - 1):
        bg = (actions[i], actions[i + 1])
        bigrams[bg] = bigrams.get(bg, 0) + 1
    loop_rate = 1.0 if any(c >= 3 for c in bigrams.values()) else 0.0

    return {"repeat_rate": float(repeat_rate), "loop_rate": float(loop_rate), "unique_actions": float(unique)}


@dataclass
class EpisodeResult:
    model: str
    dataset: str  # "tasks" or "snapshots"
    task_id: str
    snapshot_id: Optional[str]
    quadrant: Optional[str]
    success: int
    total_reward: float
    steps: int
    repeat_rate: float
    loop_rate: float


def run_episode_from_reset(
    policy: ModelFactoryPolicy,
    task_id: str,
    max_steps: int,
    mock_env: bool,
) -> Tuple[int, float, int, List[str]]:
    env = create_env(WebShopConfig(max_steps=max_steps), mock=mock_env)
    obs = env.reset(task_id)
    actions: List[str] = []
    total_reward = 0.0

    for _ in range(max_steps):
        action = policy.act(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        if done:
            break

    success = 1 if env.is_success(total_reward) else 0
    env.close()
    return success, total_reward, len(actions), actions


def run_episode_from_snapshot(
    policy: ModelFactoryPolicy,
    snapshot: Snapshot,
    max_steps: int,
    mock_env: bool,
) -> Tuple[int, float, int, List[str]]:
    env = create_env(WebShopConfig(max_steps=max_steps), mock=mock_env)
    obs = env.reset(snapshot.task_id)
    # restore to snapshot state
    obs = env.set_state(snapshot.env_state_bytes)

    actions: List[str] = []
    total_reward = 0.0

    for _ in range(max_steps):
        action = policy.act(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        if done:
            break

    success = 1 if env.is_success(total_reward) else 0
    env.close()
    return success, total_reward, len(actions), actions


def sample_task_ids(n_tasks: int, max_steps: int, seed: int, mock_env: bool) -> List[str]:
    rnd = random.Random(seed)
    env = create_env(WebShopConfig(max_steps=max_steps), mock=mock_env)
    task_ids: List[str] = []
    seen = set()

    attempts = 0
    while len(task_ids) < n_tasks and attempts < n_tasks * 20:
        attempts += 1
        obs = env.reset(None)
        # Wrapper exposes task_id
        tid = getattr(env, "task_id", None)
        if tid is None:
            # fallback: hash observation
            tid = f"task_{abs(hash(str(obs))) % 10_000_000}"
        tid = str(tid)
        if tid in seen:
            continue
        seen.add(tid)
        task_ids.append(tid)
        # randomize a bit by taking a step sometimes
        if rnd.random() < 0.2:
            try:
                _ = env.step("noop")
            except Exception:
                pass

    env.close()
    if len(task_ids) < n_tasks:
        logger.warning(f"Only sampled {len(task_ids)}/{n_tasks} unique tasks")
    return task_ids


def load_labeled_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def build_snapshot_panel(
    labeled_records: List[Dict[str, Any]],
    n_per_quadrant: int,
    seed: int,
    holdout_frac: float,
) -> List[Dict[str, Any]]:
    """Return a list of dicts with snapshot_id, quadrant, snapshot (nested)."""
    rnd = random.Random(seed)

    def is_holdout(snapshot_id: str) -> bool:
        # deterministic pseudo-split by id
        h = abs(hash(snapshot_id)) % 10_000
        return (h / 10_000.0) < holdout_frac

    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in labeled_records:
        q = r.get("quadrant")
        sid = r.get("snapshot_id")
        snap = r.get("snapshot")
        if not (q and sid and isinstance(snap, dict) and snap.get("env_state_b64")):
            continue
        if not is_holdout(str(sid)):
            continue
        by_q.setdefault(str(q), []).append(r)

    panel: List[Dict[str, Any]] = []
    for q in ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]:
        candidates = by_q.get(q, [])
        if not candidates:
            logger.warning(f"No holdout snapshots found for {q}; falling back to any snapshots")
            candidates = [r for r in labeled_records if r.get("quadrant") == q and r.get("snapshot", {}).get("env_state_b64")]
        if not candidates:
            continue
        if len(candidates) <= n_per_quadrant:
            chosen = candidates
        else:
            chosen = rnd.sample(candidates, n_per_quadrant)
        panel.extend(chosen)

    return panel


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_overall(episodes: List[EpisodeResult]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_model: Dict[str, List[EpisodeResult]] = {}
    for e in episodes:
        if e.dataset != "tasks":
            continue
        by_model.setdefault(e.model, []).append(e)

    for model, eps in sorted(by_model.items()):
        n = len(eps)
        if n == 0:
            continue
        success = sum(e.success for e in eps)
        out.append(
            {
                "model": model,
                "n": n,
                "success_rate": success / n,
                "mean_reward": sum(e.total_reward for e in eps) / n,
                "mean_steps": sum(e.steps for e in eps) / n,
            }
        )
    return out


def aggregate_per_quadrant(episodes: List[EpisodeResult]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], List[EpisodeResult]] = {}
    for e in episodes:
        if e.dataset != "snapshots" or not e.quadrant:
            continue
        by_key.setdefault((e.model, e.quadrant), []).append(e)

    for (model, q), eps in sorted(by_key.items()):
        n = len(eps)
        if n == 0:
            continue
        success = sum(e.success for e in eps)
        out.append(
            {
                "model": model,
                "quadrant": q,
                "n": n,
                "success_rate": success / n,
                "mean_reward": sum(e.total_reward for e in eps) / n,
                "mean_steps": sum(e.steps for e in eps) / n,
            }
        )
    return out


def compute_transfer_matrix(
    per_q_rows: List[Dict[str, Any]],
    base_model_name: str,
) -> List[Dict[str, Any]]:
    """Derive transfer matrix rows from per-quadrant snapshot evaluation.

    We compute Δsuccess vs base for each (train_model -> eval_quadrant).
    """
    base_by_q = {r["quadrant"]: float(r["success_rate"]) for r in per_q_rows if r["model"] == base_model_name}

    out: List[Dict[str, Any]] = []
    for r in per_q_rows:
        model = r["model"]
        q = r["quadrant"]
        if model == base_model_name:
            continue
        base = base_by_q.get(q)
        if base is None:
            continue
        meta = parse_run_id(model)
        out.append(
            {
                "model": model,
                "train_quadrant": meta.get("train_quadrant"),
                "supervision": meta.get("supervision"),
                "eval_quadrant": q,
                "success_rate": float(r["success_rate"]),
                "base_success": float(base),
                "delta_success": float(r["success_rate"]) - float(base),
            }
        )
    return out


def compute_retention(
    episodes: List[EpisodeResult],
    base_model_name: str,
) -> List[Dict[str, Any]]:
    """Retention is computed on tasks where the base succeeds."""
    base_eps = [e for e in episodes if e.dataset == "tasks" and e.model == base_model_name]
    base_success_tasks = {e.task_id for e in base_eps if e.success == 1}

    out: List[Dict[str, Any]] = []
    by_model: Dict[str, List[EpisodeResult]] = {}
    for e in episodes:
        if e.dataset != "tasks" or e.task_id not in base_success_tasks:
            continue
        by_model.setdefault(e.model, []).append(e)

    # base retention is, by definition, 1.0 on this subset (unless stochastic), but we compute it anyway.
    base_rate = None
    if base_model_name in by_model:
        eps = by_model[base_model_name]
        base_rate = sum(e.success for e in eps) / max(1, len(eps))

    for model, eps in sorted(by_model.items()):
        n = len(eps)
        if n == 0:
            continue
        rate = sum(e.success for e in eps) / n
        out.append(
            {
                "model": model,
                "n": n,
                "retention_success_rate": rate,
                "base_retention_success_rate": base_rate,
                "retention_drop": (base_rate - rate) if base_rate is not None else None,
            }
        )
    return out


def aggregate_stuckness(episodes: List[EpisodeResult]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], List[EpisodeResult]] = {}
    for e in episodes:
        by_key.setdefault((e.model, e.dataset), []).append(e)

    for (model, dataset), eps in sorted(by_key.items()):
        n = len(eps)
        if n == 0:
            continue
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "n": n,
                "mean_repeat_rate": sum(e.repeat_rate for e in eps) / n,
                "mean_loop_rate": sum(e.loop_rate for e in eps) / n,
                "mean_steps": sum(e.steps for e in eps) / n,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v8 evaluation suite")
    ap.add_argument("--training-summary", type=Path, required=True)
    ap.add_argument("--labeled-snapshots", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/phase3/v8"))

    ap.add_argument("--n-tasks", type=int, default=100)
    ap.add_argument("--n-snapshots-per-quadrant", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mock-env", action="store_true")
    ap.add_argument("--base-model", type=str, default=None, help="Override base model used to load adapters")

    ap.add_argument("--panel-file", type=Path, default=None, help="Optional pre-built snapshot panel JSON")
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="If building panel from labeled set, use this holdout fraction")

    # model loading
    ap.add_argument("--load-in-8bit", action="store_true", help="Load models in 8-bit (bitsandbytes)")
    ap.add_argument("--load-in-4bit", action="store_true", help="Load models in 4-bit (bitsandbytes)")

    args = ap.parse_args()
    setup_logging()
    set_seed(args.seed)

    safe_mkdir(args.output_dir)

    model_paths = parse_training_summary(args.training_summary)
    base_model = infer_base_model(args.training_summary, args.base_model)

    # Always include base model for deltas
    models: Dict[str, Optional[str]] = {"BASE": None}
    models.update(model_paths)

    labeled_records = load_labeled_records(args.labeled_snapshots)

    # Snapshot panel
    if args.panel_file:
        panel_records = json.loads(args.panel_file.read_text())
        if not isinstance(panel_records, list):
            raise SystemExit("--panel-file must be a JSON list")
    else:
        panel_records = build_snapshot_panel(
            labeled_records,
            n_per_quadrant=args.n_snapshots_per_quadrant,
            seed=args.seed,
            holdout_frac=args.holdout_frac,
        )

    # Persist sampled artifacts
    with open(args.output_dir / "snapshot_panel.json", "w") as f:
        json.dump(
            [
                {
                    "snapshot_id": r.get("snapshot_id"),
                    "quadrant": r.get("quadrant"),
                    "task_id": r.get("snapshot", {}).get("task_id"),
                    "step_idx": r.get("snapshot", {}).get("step_idx"),
                }
                for r in panel_records
            ],
            f,
            indent=2,
        )

    task_ids = sample_task_ids(args.n_tasks, args.max_steps, args.seed, args.mock_env)
    with open(args.output_dir / "task_ids.json", "w") as f:
        json.dump(task_ids, f, indent=2)

    with open(args.output_dir / "eval_config.json", "w") as f:
        json.dump(
            {
                "base_model": base_model,
                "n_models": len(models),
                "n_tasks": len(task_ids),
                "n_panel": len(panel_records),
                "n_per_quadrant": args.n_snapshots_per_quadrant,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "mock_env": args.mock_env,
                "load_in_8bit": args.load_in_8bit,
                "load_in_4bit": args.load_in_4bit,
            },
            f,
            indent=2,
        )

    episodes: List[EpisodeResult] = []

    for model_name, lora_path in models.items():
        logger.info(f"Evaluating model {model_name}")

        cfg = ModelConfig(
            model_name=model_name,
            model_path=base_model,
            lora_path=lora_path,
            load_in_8bit=bool(args.load_in_8bit),
            load_in_4bit=bool(args.load_in_4bit),
        )
        factory = ModelFactory(cfg)
        policy = ModelFactoryPolicy(factory)

        # Task evaluation
        for tid in task_ids:
            try:
                success, total_reward, steps, actions = run_episode_from_reset(
                    policy=policy,
                    task_id=tid,
                    max_steps=args.max_steps,
                    mock_env=args.mock_env,
                )
            except Exception as e:
                logger.warning(f"Task eval failed for {model_name} on {tid}: {e}")
                success, total_reward, steps, actions = 0, 0.0, 0, []

            s = compute_stuckness(actions)
            episodes.append(
                EpisodeResult(
                    model=model_name,
                    dataset="tasks",
                    task_id=tid,
                    snapshot_id=None,
                    quadrant=None,
                    success=success,
                    total_reward=total_reward,
                    steps=steps,
                    repeat_rate=s["repeat_rate"],
                    loop_rate=s["loop_rate"],
                )
            )

        # Snapshot panel evaluation
        for r in panel_records:
            sid = str(r.get("snapshot_id"))
            q = r.get("quadrant")
            snap_dict = r.get("snapshot")
            if not isinstance(snap_dict, dict):
                continue
            try:
                snap = Snapshot.from_dict(snap_dict)
            except Exception:
                continue

            try:
                success, total_reward, steps, actions = run_episode_from_snapshot(
                    policy=policy,
                    snapshot=snap,
                    max_steps=args.max_steps,
                    mock_env=args.mock_env,
                )
            except Exception as e:
                logger.warning(f"Snapshot eval failed for {model_name} on {sid}: {e}")
                success, total_reward, steps, actions = 0, 0.0, 0, []

            s = compute_stuckness(actions)
            episodes.append(
                EpisodeResult(
                    model=model_name,
                    dataset="snapshots",
                    task_id=str(snap.task_id),
                    snapshot_id=sid,
                    quadrant=str(q) if q is not None else None,
                    success=success,
                    total_reward=total_reward,
                    steps=steps,
                    repeat_rate=s["repeat_rate"],
                    loop_rate=s["loop_rate"],
                )
            )

    # Aggregates
    overall_rows = aggregate_overall(episodes)
    per_q_rows = aggregate_per_quadrant(episodes)
    transfer_rows = compute_transfer_matrix(per_q_rows, base_model_name="BASE")
    retention_rows = compute_retention(episodes, base_model_name="BASE")
    stuck_rows = aggregate_stuckness(episodes)

    write_csv(
        args.output_dir / "overall_task_results.csv",
        overall_rows,
        fieldnames=["model", "n", "success_rate", "mean_reward", "mean_steps"],
    )
    write_csv(
        args.output_dir / "per_quadrant_results.csv",
        per_q_rows,
        fieldnames=["model", "quadrant", "n", "success_rate", "mean_reward", "mean_steps"],
    )
    write_csv(
        args.output_dir / "transfer_matrix.csv",
        transfer_rows,
        fieldnames=[
            "model",
            "train_quadrant",
            "supervision",
            "eval_quadrant",
            "success_rate",
            "base_success",
            "delta_success",
        ],
    )
    write_csv(
        args.output_dir / "retention_results.csv",
        retention_rows,
        fieldnames=["model", "n", "retention_success_rate", "base_retention_success_rate", "retention_drop"],
    )
    write_csv(
        args.output_dir / "stuckness_results.csv",
        stuck_rows,
        fieldnames=["model", "dataset", "n", "mean_repeat_rate", "mean_loop_rate", "mean_steps"],
    )

    logger.info(f"Wrote eval artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
