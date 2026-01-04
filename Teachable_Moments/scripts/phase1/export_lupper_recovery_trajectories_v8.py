#!/usr/bin/env python3
"""Export L_upper recovery trajectories for SFT.

Why this exists
---------------
In the blueprint, L_upper is the "expert-takeover" probability of success from a
failure snapshot. Computing it everywhere is expensive. A useful compromise is:

  (1) Compute L_upper (or approximate it) on a subset of snapshots.
  (2) Store a *single successful expert continuation* (if found).
  (3) Reuse that continuation as a trajectory-level demonstration dataset.

This script does (2) and produces a JSONL file with step-by-step (obs, valid_actions, action)
that can be consumed as "recovery trajectory imitation" data.

It supports both:
  - local expert checkpoints (ModelFactoryPolicy)
  - OpenAI teacher model as expert (TeacherClient)

Output format (one JSON object per trajectory)
---------------------------------------------
{
  "snapshot_id": "...",
  "task_id": "...",
  "quadrant": "Q1_highU_highL",
  "L_upper": 0.73,
  "expert": {"type": "teacher", "model": "gpt-4.1-mini"},
  "success": true,
  "total_reward": 0.9,
  "n_steps": 7,
  "steps": [
     {"t": 0, "observation": "...", "valid_actions": [...], "action": "search[...]", "reward": 0.0, "done": false},
     ...
  ]
}
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported file type: {path}")


def _snapshot_dict(rec: Dict[str, Any]) -> Dict[str, Any]:
    return rec.get("snapshot") if isinstance(rec.get("snapshot"), dict) else rec


class _TeacherPolicy:
    def __init__(self, model: str, temperature: float, max_tokens: int, cache_dir: str):
        # Lazy import so this script can run without openai installed when using local.
        from src.teacher.client import TeacherClient, TeacherConfig

        cfg = TeacherConfig(model=model, temperature=temperature, max_tokens=max_tokens, cache_dir=cache_dir)
        self.client = TeacherClient(cfg)

    def get_action(self, observation: str, valid_actions: List[str], instruction_text: Optional[str] = None) -> str:
        hint = self.client.get_teacher_hint(observation, valid_actions, instruction_text=instruction_text)
        return hint.suggested_action


def _build_expert_policy(args) -> Any:
    if args.expert_type == "teacher":
        return _TeacherPolicy(
            model=args.teacher_model,
            temperature=float(args.teacher_temperature),
            max_tokens=int(args.teacher_max_tokens),
            cache_dir=str(args.teacher_cache_dir),
        )

    # Local checkpoint
    from src.utils.model_factory import ModelConfig, ModelFactory
    from src.policies.model_factory_policy import ModelFactoryPolicy

    cfg = ModelConfig.from_checkpoint(str(args.expert_checkpoint))
    if args.device:
        cfg.device = args.device
    if args.load_in_8bit:
        cfg.load_in_8bit = True
    if args.load_in_4bit:
        cfg.load_in_4bit = True
    mf = ModelFactory(cfg)
    _model, _tok = mf.load()
    return ModelFactoryPolicy(mf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export L_upper recovery trajectories (v8)")
    ap.add_argument("--labeled-snapshots", type=Path, required=True, help="labeled_snapshots.jsonl or .json")
    ap.add_argument("--output", type=Path, default=Path("results/phase1/v8/recovery_trajectories.jsonl"))

    # Subset selection
    ap.add_argument("--max-trajectories", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quadrants", nargs="+", default=None, help="Optional list of quadrants to include")
    ap.add_argument("--min-l-upper", type=float, default=None, help="Filter: require L_upper >= threshold")
    ap.add_argument(
        "--sort-by",
        type=str,
        default="L_upper",
        choices=["L_upper", "ELP_net", "random"],
        help="How to pick trajectories from the pool",
    )

    # Expert policy
    ap.add_argument("--expert-type", type=str, default="local", choices=["local", "teacher"])
    ap.add_argument("--expert-checkpoint", type=Path, default=None, help="Local expert checkpoint (LoRA dir)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")

    # Teacher expert settings
    ap.add_argument("--teacher-model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--teacher-temperature", type=float, default=0.2)
    ap.add_argument("--teacher-max-tokens", type=int, default=256)
    ap.add_argument("--teacher-cache-dir", type=Path, default=Path(".teacher_cache"))

    # Rollout
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--mock-env", action="store_true", help="Use mock env (for CI/pilots)")

    args = ap.parse_args()
    setup_logging()

    if args.expert_type == "local" and args.expert_checkpoint is None:
        raise SystemExit("--expert-checkpoint is required when --expert-type local")

    rng = random.Random(args.seed)

    # Imports here so docs-only operations don't require WebShop installed.
    from src.data.snapshot import Snapshot
    from src.data.webshop_env import WebShopConfig, create_env

    records = _load_records(args.labeled_snapshots)
    logger.info(f"Loaded {len(records)} labeled records")

    # Filter subset
    pool: List[Dict[str, Any]] = []
    for r in records:
        if r.get("split") not in (None, "train"):
            continue
        if r.get("held_out"):
            continue
        q = r.get("quadrant")
        if args.quadrants and q not in set(args.quadrants):
            continue
        if args.min_l_upper is not None:
            lu = (((r.get("leverage") or {}).get("L_upper")) or 0.0)
            if float(lu) < float(args.min_l_upper):
                continue
        pool.append(r)

    if not pool:
        raise SystemExit("No snapshots after filtering")

    # Sorting/selection
    if args.sort_by == "random":
        rng.shuffle(pool)
    elif args.sort_by == "ELP_net":
        pool.sort(key=lambda r: float(((r.get("cpt") or {}).get("ELP_net")) or 0.0), reverse=True)
    else:  # L_upper
        pool.sort(key=lambda r: float(((r.get("leverage") or {}).get("L_upper")) or 0.0), reverse=True)

    pool = pool[: int(args.max_trajectories)]
    logger.info(f"Exporting up to {len(pool)} recovery trajectories")

    expert = _build_expert_policy(args)
    env = create_env(WebShopConfig(max_steps=int(args.max_steps)), mock=bool(args.mock_env))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_success = 0
    with open(args.output, "w") as f:
        for idx, r in enumerate(pool):
            snap_dict = _snapshot_dict(r)
            try:
                snap = Snapshot.from_dict(snap_dict)
            except Exception as e:
                logger.warning(f"Skip malformed snapshot at idx={idx}: {e}")
                continue

            # Restore state
            try:
                # env.set_state returns obs dict with observation, valid_actions, instruction_text
                obs_dict = env.set_state(snap.env_state_bytes)
            except Exception as e:
                logger.warning(f"Failed to restore state for snapshot {snap.id}: {e}")
                continue

            traj_steps: List[Dict[str, Any]] = []
            done = False
            total_reward = 0.0
            steps = 0
            # env.set_state returned current observation
            cur_obs = obs_dict
            while not done and steps < int(args.max_steps):
                observation = cur_obs.get("observation", "")
                valid_actions = cur_obs.get("valid_actions", [])
                instr = cur_obs.get("instruction_text", None)

                action = expert.get_action(observation, valid_actions, instruction_text=instr)
                next_obs, reward, done, _info = env.step(action)
                total_reward += float(reward)
                traj_steps.append(
                    {
                        "t": steps,
                        "observation": observation,
                        "valid_actions": valid_actions,
                        "action": action,
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
                cur_obs = next_obs
                steps += 1

            success = env.is_success(total_reward)
            n_success += int(success)

            out = {
                "snapshot_id": snap.id,
                "task_id": snap.task_id,
                "quadrant": r.get("quadrant"),
                "L_upper": float(((r.get("leverage") or {}).get("L_upper")) or 0.0),
                "ELP_net": float(((r.get("cpt") or {}).get("ELP_net")) or 0.0),
                "expert": {
                    "type": args.expert_type,
                    "checkpoint": str(args.expert_checkpoint) if args.expert_type == "local" else None,
                    "model": args.teacher_model if args.expert_type == "teacher" else None,
                },
                "success": bool(success),
                "total_reward": float(total_reward),
                "n_steps": len(traj_steps),
                "steps": traj_steps,
            }
            f.write(json.dumps(out) + "\n")

            if (idx + 1) % 25 == 0:
                logger.info(f"Processed {idx+1}/{len(pool)} trajectories (successes so far: {n_success})")

    logger.info(f"Wrote {args.output} with {len(pool)} trajectories; success fraction={n_success/max(1,len(pool)):.3f}")


if __name__ == "__main__":
    main()
