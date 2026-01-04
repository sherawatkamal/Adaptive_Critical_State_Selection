#!/usr/bin/env python3
"""Build the Phase 1 teachability dataset from student rollouts.

This script is the practical glue between:
- Phase 0 rollouts (saved as rollouts.json)
- Replayable failure snapshots (Snapshot objects with env_state_bytes)
- Labels: uncertainty (UQ), leverage (actionability), CPT (patch gain)
- Quadrant assignment + stratified panel selection

Two common modes:

1) Smoke-test mode (no large models):
   - --student-policy first
   - --teacher-hints mock
   - --mock-env

2) Research mode:
   - --student-policy model_factory --student-checkpoint <ckpt>
   - optional: --teacher-hints llm

Outputs (in --output-dir):
- snapshots.json
- labeled_snapshots.json
- thresholds.json
- panel.json
- summary.json

Note on prompts:
- The environment wrapper returns instruction_text on reset/set_state, but not always on step().
  To keep prompts consistent for CPT/leverage, we wrap envs so *every* observation is prefixed with:
    "Task: <instruction_text>"
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.snapshot import Snapshot, TeacherHint, ErrorType
from src.label.uncertainty import compute_all_uncertainty
from src.label.quadrant import assign_quadrant, compute_thresholds
from src.features.tier1_structural import extract_structural_features

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))


def _safe_pickle_loads(b: bytes) -> dict:
    try:
        obj = pickle.loads(b)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _extract_instruction_text(snapshot: Snapshot) -> str:
    if not snapshot.env_state_bytes:
        return ""
    st = _safe_pickle_loads(snapshot.env_state_bytes)
    return str(st.get("instruction_text", "") or "")


def _prefix_task(instr: str, obs: str) -> str:
    instr = (instr or "").strip()
    if not instr:
        return obs
    if "Task:" in obs[:80] or instr in obs:
        return obs
    return f"Task: {instr}\n\n{obs}"


class _TaskObsEnvWrapper:
    """Wrap an env so observations always contain the task instruction."""

    def __init__(self, env: Any):
        self._env = env
        self._instruction_text: str = ""

    def _massage_obs(self, obs: Any) -> Any:
        if not isinstance(obs, dict):
            return obs
        if obs.get("instruction_text"):
            self._instruction_text = str(obs.get("instruction_text") or "")
        if "observation" in obs and self._instruction_text:
            obs = dict(obs)
            obs["observation"] = _prefix_task(self._instruction_text, obs["observation"])
        return obs

    def reset(self, task_id: Optional[str] = None):
        return self._massage_obs(self._env.reset(task_id))

    def step(self, action: str):
        obs, reward, done, info = self._env.step(action)
        return self._massage_obs(obs), reward, done, info

    def get_state(self) -> bytes:
        return self._env.get_state()

    def set_state(self, state_bytes: bytes):
        return self._massage_obs(self._env.set_state(state_bytes))

    def is_success(self, total_reward: float) -> bool:
        return self._env.is_success(total_reward)

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)


# ---------------------------
# Policies
# ---------------------------

class FirstActionPolicy:
    """Deterministic dependency-free policy (always pick first action).

    For uncertainty metrics we return a uniform distribution so entropy is defined.
    """

    def get_action(self, observation: str, valid_actions: List[str]) -> str:
        return valid_actions[0] if valid_actions else ""

    def get_action_distribution(self, observation: str, valid_actions: List[str]) -> Dict[str, float]:
        if not valid_actions:
            return {}
        p = 1.0 / float(len(valid_actions))
        return {a: p for a in valid_actions}


def _make_student_policy(args) -> Any:
    if args.student_policy == "first":
        return FirstActionPolicy()

    if args.student_policy == "random":
        from src.policies import RandomPolicy
        return RandomPolicy(seed=args.seed)

    if args.student_policy == "model_factory":
        if not args.student_checkpoint:
            raise ValueError("--student-checkpoint is required for --student-policy model_factory")
        from src.utils.model_factory import ModelFactory, ModelConfig
        from src.policies import ModelFactoryPolicy
        model_cfg = ModelConfig.from_checkpoint(args.student_checkpoint)
        mf = ModelFactory(model_cfg)
        return ModelFactoryPolicy(mf)

    raise ValueError(f"Unknown --student-policy: {args.student_policy}")


def _make_expert_policy(args, student_policy: Any) -> Any:
    if args.expert_policy in ("student", None):
        return student_policy
    if args.expert_policy == "first":
        return FirstActionPolicy()
    if args.expert_policy == "random":
        from src.policies import RandomPolicy
        return RandomPolicy(seed=args.seed + 1337)
    if args.expert_policy == "teacher":
        from src.teacher.client import TeacherClient, TeacherConfig

        cfg = TeacherConfig(
            model=args.teacher_model,
            temperature=args.teacher_temperature,
            max_tokens=args.teacher_max_tokens,
            use_cache=True,
        )
        client = TeacherClient(cfg)

        class TeacherActionPolicy:
            def get_action(self, observation: str, valid_actions: List[str]) -> str:
                actions_str = "\n".join(f"- {a}" for a in valid_actions[:50])
                prompt = (
                    "You are an expert web-shopping agent.\n\n"
                    "Choose the SINGLE BEST next action to make progress.\n\n"
                    f"OBSERVATION:\n{observation[:2000]}\n\n"
                    f"VALID_ACTIONS:\n{actions_str}\n\n"
                    "Return JSON only: {\"action\": \"...\"} where action EXACTLY matches one of VALID_ACTIONS."
                )
                resp = client.generate(prompt=prompt, system_prompt="Return valid JSON only.")
                # Extract action
                try:
                    import re, json as _json
                    m = re.search(r"\{.*\}", resp, re.DOTALL)
                    data = _json.loads(m.group()) if m else {}
                    act = str(data.get("action", "")).strip()
                except Exception:
                    act = ""

                if act in valid_actions:
                    return act
                # Fallback: substring match
                low = resp.lower()
                for a in valid_actions:
                    if a.lower() in low:
                        return a
                return valid_actions[0] if valid_actions else ""

        return TeacherActionPolicy()

    raise ValueError(f"Unknown --expert-policy: {args.expert_policy}")


# ---------------------------
# Teacher hints
# ---------------------------

def _mock_teacher_hint(snapshot: Snapshot) -> TeacherHint:
    suggested = snapshot.valid_actions[0] if snapshot.valid_actions else ""
    return TeacherHint(
        suggested_action=suggested,
        rationale="[mock] choose first valid action",
        confidence="low",
        error_type=ErrorType.PLANNING_ERROR,
    )


def _llm_teacher_hint(snapshot: Snapshot, instruction_text: str, student_action: str, args) -> TeacherHint:
    from src.teacher.client import TeacherClient, TeacherConfig
    from src.teacher.structured_hint import build_teacher_hint_prompt

    cfg = TeacherConfig(
        model=args.teacher_model,
        temperature=args.teacher_temperature,
        max_tokens=args.teacher_max_tokens,
        use_cache=True,
    )
    client = TeacherClient(cfg)

    prompt = build_teacher_hint_prompt(
        instruction_text=instruction_text,
        observation=_prefix_task(instruction_text, snapshot.observation),
        valid_actions=snapshot.valid_actions,
        student_action=student_action,
    )
    resp = client.generate(prompt=prompt, system_prompt="Return JSON only.")

    import re, json as _json

    data: Dict[str, Any] = {}
    try:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        data = _json.loads(m.group()) if m else {}
    except Exception:
        data = {}

    suggested = str(data.get("suggested_action", "") or "").strip()
    rationale = str(data.get("rationale", "") or "").strip()
    err = str(data.get("error_type", "planning_error") or "planning_error").strip().lower()
    conf = str(data.get("confidence", "medium") or "medium").strip().lower()

    err_map = {
        "affordance_miss": ErrorType.AFFORDANCE_MISS,
        "attribute_confusion": ErrorType.ATTRIBUTE_CONFUSION,
        "planning_error": ErrorType.PLANNING_ERROR,
        "exploration_failure": ErrorType.EXPLORATION_FAILURE,
        "unknown": ErrorType.UNKNOWN,
    }
    error_type = err_map.get(err, ErrorType.UNKNOWN)

    if suggested not in snapshot.valid_actions and snapshot.valid_actions:
        s_low = suggested.lower()
        for a in snapshot.valid_actions:
            if s_low and s_low in a.lower():
                suggested = a
                break
        if suggested not in snapshot.valid_actions:
            suggested = snapshot.valid_actions[0]

    return TeacherHint(
        suggested_action=suggested,
        rationale=rationale or "(no rationale)",
        confidence=conf if conf in ("low", "medium", "high") else "medium",
        error_type=error_type,
        model=args.teacher_model,
    )


# ---------------------------
# Snapshot mining
# ---------------------------

def _mine_failure_snapshots(
    rollouts_data: dict,
    offsets: List[int],
    max_snapshots: Optional[int],
    include_success: bool,
) -> Tuple[List[Snapshot], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return (snapshots, stats, extras_by_id).

    extras_by_id stores fields not present in Snapshot schema (e.g., student action, action_probs).
    """
    results = rollouts_data.get("results") or []
    mined: List[Snapshot] = []
    extras_by_id: Dict[str, Dict[str, Any]] = {}

    stats = {
        "n_rollouts": len(results),
        "n_rollouts_used": 0,
        "skipped_missing_rollout_states": 0,
        "skipped_missing_env_state": 0,
        "skipped_success": 0,
    }

    for r in results:
        if not include_success and r.get("success") is True:
            stats["skipped_success"] += 1
            continue

        rollout_states = r.get("rollout_states") or []
        if not rollout_states:
            stats["skipped_missing_rollout_states"] += 1
            continue

        failure_events = r.get("failures") or []
        failure_steps = [int(f.get("step_idx")) for f in failure_events if f.get("step_idx") is not None]
        failure_step = min(failure_steps) if failure_steps else (len(rollout_states) - 1)

        traj_id = r.get("trajectory_id") or f"traj_{stats['n_rollouts_used']:06d}"
        task_id = r.get("task_id") or r.get("env_task_id") or traj_id

        stats["n_rollouts_used"] += 1

        for k in offsets:
            idx = max(0, failure_step - int(k))
            rs = rollout_states[idx] if idx < len(rollout_states) else rollout_states[-1]

            env_state_b64 = rs.get("env_state_b64") or rs.get("env_state")
            if not env_state_b64:
                stats["skipped_missing_env_state"] += 1
                continue

            try:
                env_state_bytes = _b64_to_bytes(env_state_b64)
            except Exception:
                stats["skipped_missing_env_state"] += 1
                continue

            snap_id = f"{traj_id}_step{idx:04d}_k{int(k)}"

            snap = Snapshot(
                id=snap_id,
                task_id=str(task_id),
                step_idx=int(idx),
                trajectory_id=str(traj_id),
                env_state_bytes=env_state_bytes,
                observation=str(rs.get("observation", "")),
                valid_actions=list(rs.get("valid_actions", []) or []),
                agent_prefix=None,
                last_action=str((rs.get("action_taken") or rs.get("last_action", "")) or "") or None,
                action_type="student",
                teacher_hint=None,
                trajectory_outcome={
                    "success": bool(r.get("success")),
                    "total_reward": r.get("total_reward"),
                },
                timestamp=float(r.get("timestamp", time.time())),
            )
            mined.append(snap)
            extras_by_id[snap_id] = {
                "student_action": str(rs.get("action_taken", "") or ""),
                "confidence": rs.get("confidence"),
                "action_probs": rs.get("action_probs"),
                "failure_step": int(failure_step),
                "offset_k": int(k),
            }

            if max_snapshots is not None and len(mined) >= int(max_snapshots):
                break

        if max_snapshots is not None and len(mined) >= int(max_snapshots):
            break

    return mined, stats, extras_by_id


# ---------------------------
# Panel selection
# ---------------------------

def _select_panel(
    labeled_records: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    buckets: Dict[str, List[str]] = {}
    for rec in labeled_records:
        q = rec.get("quadrant") or "UNKNOWN"
        sid = rec.get("id")
        if not sid:
            continue
        buckets.setdefault(q, []).append(sid)

    quads = [q for q in buckets.keys() if q != "UNKNOWN"]
    quads = sorted(quads)
    targets: Dict[str, int] = {}

    if len(quads) >= 4:
        per = n // 4
        for q in quads[:4]:
            targets[q] = per
        rem = n - per * 4
        if rem > 0:
            biggest = max(quads[:4], key=lambda q: len(buckets[q]))
            targets[biggest] += rem
    else:
        total = sum(len(v) for v in buckets.values())
        for q, ids in buckets.items():
            targets[q] = max(1, int(n * len(ids) / max(total, 1)))

    panel: List[str] = []
    for q, n_q in targets.items():
        ids = list(buckets.get(q, []))
        rng.shuffle(ids)
        panel.extend(ids[:n_q])

    rng.shuffle(panel)
    panel = panel[:n]
    return {
        "n": len(panel),
        "seed": seed,
        "ids": panel,
        "targets": targets,
        "bucket_sizes": {k: len(v) for k, v in buckets.items()},
    }


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    _setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--snapshot-offsets", default="3", help="Comma-separated k offsets: snapshot at failure_step-k")  # noqa
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--include-success", action="store_true")

    ap.add_argument("--mock-env", action="store_true")
    ap.add_argument("--env-max-steps", type=int, default=50)

    ap.add_argument("--student-policy", choices=["model_factory", "first", "random"], default="first")
    ap.add_argument("--student-checkpoint", default=None)

    ap.add_argument("--expert-policy", choices=["student", "teacher", "first", "random"], default="student")

    ap.add_argument("--teacher-hints", choices=["mock", "llm"], default="mock")
    ap.add_argument("--teacher-model", default="gpt-4o-mini")
    ap.add_argument("--teacher-temperature", type=float, default=0.2)
    ap.add_argument("--teacher-max-tokens", type=int, default=400)

    # Leverage knobs (mapped to src/label/leverage.py config)
    ap.add_argument("--skip-leverage", action="store_true")
    ap.add_argument("--leverage-n-policy", type=int, default=1, help="Alias for baseline rollouts (kept for compatibility)")  # noqa
    ap.add_argument("--leverage-n-force", type=int, default=1, help="Alias for baseline/forced rollouts (kept for compatibility)")  # noqa
    ap.add_argument("--leverage-n-expert", type=int, default=0)
    ap.add_argument("--leverage-horizon", type=int, default=8, help="Alias for max_steps")  # noqa
    ap.add_argument("--leverage-parallel", type=int, default=4)

    # CPT knobs (mapped to src/label/patch_gain.py config)
    ap.add_argument("--skip-cpt", action="store_true")
    ap.add_argument("--cpt-n-per-condition", type=int, default=1, help="Mapped to CPTConfig.max_rollouts")  # noqa
    ap.add_argument("--cpt-horizon", type=int, default=None, help="(deprecated) no-op; use --cpt-max-steps")
    ap.add_argument("--cpt-max-steps", type=int, default=10)
    ap.add_argument("--cpt-parallel", type=int, default=4)

    # Panel
    ap.add_argument("--panel-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(args.rollouts, "r") as f:
        rollouts_data = json.load(f)

    offsets = [int(x) for x in str(args.snapshot_offsets).split(",") if str(x).strip()]
    snapshots, mine_stats, extras_by_id = _mine_failure_snapshots(
        rollouts_data=rollouts_data,
        offsets=offsets,
        max_snapshots=args.max_snapshots,
        include_success=bool(args.include_success),
    )
    logger.info("Mined %d snapshots", len(snapshots))

    snapshots_path = out_dir / "snapshots.json"
    with open(snapshots_path, "w") as f:
        json.dump([s.to_dict() for s in snapshots], f, indent=2)
    logger.info("Wrote %s", snapshots_path)

    if not snapshots:
        raise SystemExit(
            "No snapshots mined. Ensure rollouts.json contains env_state_b64 in rollout_states (apply fixpack student_rollout)."  # noqa
        )

    # Env factory (wrapped)
    from src.data.webshop_env import create_env, WebShopConfig

    env_cfg = WebShopConfig(max_steps=args.env_max_steps)

    def env_factory():
        return _TaskObsEnvWrapper(create_env(env_cfg, mock=args.mock_env))

    # Policies
    student_policy = _make_student_policy(args)
    expert_policy = _make_expert_policy(args, student_policy)

    # Teacher hints per snapshot
    teacher_hints: Dict[str, TeacherHint] = {}
    for s in snapshots:
        instr = _extract_instruction_text(s)
        student_action = (extras_by_id.get(s.id, {}) or {}).get("student_action", "")
        if args.teacher_hints == "mock":
            teacher_hints[s.id] = _mock_teacher_hint(s)
        else:
            teacher_hints[s.id] = _llm_teacher_hint(s, instr, str(student_action), args)

    # Uncertainty + structural features (cheap)
    labeled_records: List[Dict[str, Any]] = []
    for s in snapshots:
        instr = _extract_instruction_text(s)
        obs_uq = _prefix_task(instr, s.observation)
        unc = compute_all_uncertainty(student_policy, obs_uq, s.valid_actions)
        U = float(unc.get("entropy", 0.0))

        extra = extras_by_id.get(s.id, {}) or {}
        flat_snap = s.to_dict()
        flat_snap["instruction_text"] = instr
        apb = extra.get("action_probs")
        if apb is not None:
            flat_snap["policy_outputs"] = {"action_probs": apb}
        struct = extract_structural_features(flat_snap).to_dict()

        labeled_records.append(
            {
                "id": s.id,
                "task_id": s.task_id,
                "trajectory_id": s.trajectory_id,
                "step_idx": s.step_idx,
                "instruction_text": instr,
                "snapshot": s.to_dict(),
                "student_action": extra.get("student_action"),
                "student_confidence": extra.get("confidence"),
                "student_action_probs": extra.get("action_probs"),
                "teacher_hint": teacher_hints[s.id].to_dict(),
                "uncertainty": unc,
                "U": U,
                "structural_features": struct,
                "leverage": None,
                "cpt": None,
                "quadrant": None,
            }
        )

    # Leverage
    if not args.skip_leverage:
        from src.label.leverage import LeverageConfig, estimate_leverage_batch

        # This leverage implementation uses one budget (n_force_rollouts) for baseline and forced
        n_force = max(int(args.leverage_n_policy), int(args.leverage_n_force))
        lev_cfg = LeverageConfig(
            n_force_rollouts=n_force,
            n_expert_rollouts=int(args.leverage_n_expert),
            max_steps=int(args.leverage_horizon),
            n_parallel=int(args.leverage_parallel),
        )
        logger.info("Running leverage labeling (%d snapshots)...", len(snapshots))
        lev_map = estimate_leverage_batch(
            snapshots=snapshots,
            env_factory=env_factory,
            student_policy=student_policy,
            expert_policy=expert_policy,
            teacher_hints=teacher_hints,
            config=lev_cfg,
        )
        for rec in labeled_records:
            sid = rec["id"]
            lev = lev_map.get(sid)
            rec["leverage"] = lev.to_dict() if lev else None
    else:
        logger.info("Skipping leverage (--skip-leverage)")

    # CPT
    if not args.skip_cpt:
        from src.label.patch_gain import CPTConfig, run_cpt_batch

        cpt_cfg = CPTConfig(
            n_per_condition=int(args.cpt_n_per_condition),
            max_steps=int(args.cpt_max_steps),
        )
        logger.info("Running CPT labeling (%d snapshots)...", len(snapshots))
        cpt_map = run_cpt_batch(
            snapshots=snapshots,
            env_factory=env_factory,
            student_policy=student_policy,
            teacher_hints=teacher_hints,
            config=cpt_cfg,
            n_parallel=int(args.cpt_parallel),
        )
        for rec in labeled_records:
            sid = rec["id"]
            cpt = cpt_map.get(sid)
            rec["cpt"] = cpt.to_dict() if cpt else None
    else:
        logger.info("Skipping CPT (--skip-cpt)")

    # Quadrants (need both U and leverage)
    try:
        U_th, L_th = compute_thresholds(labeled_records)
        for rec in labeled_records:
            lev = rec.get("leverage") or {}
            if not isinstance(lev, dict):
                continue
            if lev.get("L_local") is None:
                continue
            rec["quadrant"] = assign_quadrant(float(rec["U"]), float(lev["L_local"]), U_th, L_th)
    except Exception as e:
        U_th, L_th = None, None
        logger.warning("Could not compute quadrants: %s", e)

    labeled_path = out_dir / "labeled_snapshots.json"
    with open(labeled_path, "w") as f:
        json.dump(labeled_records, f, indent=2)
    logger.info("Wrote %s", labeled_path)

    thresholds_path = out_dir / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"U_threshold": U_th, "L_threshold": L_th}, f, indent=2)
    logger.info("Wrote %s", thresholds_path)

    panel = _select_panel(labeled_records, n=int(args.panel_n), seed=int(args.seed))
    panel_path = out_dir / "panel.json"
    with open(panel_path, "w") as f:
        json.dump(panel, f, indent=2)
    logger.info("Wrote %s", panel_path)

    quad_counts: Dict[str, int] = {}
    for rec in labeled_records:
        q = rec.get("quadrant") or "UNKNOWN"
        quad_counts[q] = quad_counts.get(q, 0) + 1

    summary = {
        "args": vars(args),
        "mine_stats": mine_stats,
        "n_snapshots": len(snapshots),
        "quadrant_counts": quad_counts,
        "elapsed_sec": time.time() - t0,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_path)

    logger.info("Done in %.1fs", summary["elapsed_sec"])


if __name__ == "__main__":
    main()
