#!/usr/bin/env python3
"""
IDT Core: shared logic for patch-teachability experiments.

Provides run_idt_experiment() used by run_idt_patch.py and run_all_experiments.py.
"""

from __future__ import annotations

import os
import sys
import json
import random
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BASELINE_DIR, ".."))
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from idt_teachability.patch_simulator import PatchSimulator
from idt_teachability.patchers import make_patcher, PatchProposal
from idt_teachability.step_selectors import (
    select_steps_last_n,
    select_steps_search_only,
    select_steps_random,
)
from eef_detailed_with_diagnosis import (
    setup_environment,
    setup_model,
    Agent,
    DiagnosisModelSelector,
    select_critical_states_baseline,
    select_critical_states_entropy,
    select_critical_states_stratified_entropy,
    select_critical_states_diagnosis,
)


def _stable_int(x: Any) -> int:
    if x is None:
        return 0
    try:
        return int(x)
    except Exception:
        return abs(hash(str(x))) % 1_000_000_000


def set_all_seeds(seed: int) -> None:
    seed = int(seed) % 2_000_000_000
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_base_seed(global_seed: int, task_id: Any, traj_idx: int, step_idx: int) -> int:
    base = (
        _stable_int(global_seed) * 1_000_003
        + _stable_int(task_id) * 9_173
        + int(traj_idx) * 1_003
        + int(step_idx) * 97
    )
    return int(base % 2_000_000_000)


def load_failures(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        out: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("failures", "trajectories", "data", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    raise ValueError(f"Unsupported failure dataset format at {path}")


def _is_action_valid_at_state(action: str, valid_actions: List[str], *, allow_any_search: bool) -> bool:
    if not isinstance(action, str) or not action:
        return False
    if action in valid_actions:
        return True
    if allow_any_search and action.startswith("search["):
        return True
    return False


def _forced_was_used(out) -> bool:
    try:
        if not out.sim_traj:
            return False
        info = out.sim_traj[0].get("action_info", {}) or {}
        return info.get("type") == "forced"
    except Exception:
        return False


def _get_forced_skip_reason(out) -> Optional[str]:
    """Return forced_skip_reason from RolloutOutcome when forced was not used."""
    return getattr(out, "forced_skip_reason", None)


def _select_states(
    trajectory: Dict[str, Any],
    *,
    strategy: str,
    M: int,
    agent: Agent,
    diagnosis_model: Optional[DiagnosisModelSelector] = None,
    diagnosis_window: int = 1,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    if strategy == "diagnosis":
        if diagnosis_model is None:
            raise ValueError("diagnosis_model required for strategy=diagnosis")
        return select_critical_states_diagnosis(
            trajectory, M=M, diagnosis_model=diagnosis_model, window=diagnosis_window, agent=agent
        )
    if strategy == "entropy":
        return select_critical_states_entropy(trajectory, M=M, agent=agent)
    if strategy == "stratified_entropy":
        return select_critical_states_stratified_entropy(trajectory, M=M, agent=agent)
    if strategy == "baseline":
        return select_critical_states_baseline(trajectory, M=M, agent=agent)
    if strategy == "last_n":
        return select_steps_last_n(trajectory, M=M)
    if strategy == "search_steps":
        return select_steps_search_only(trajectory, M=M)
    if strategy == "random_steps":
        return select_steps_random(trajectory, M=M, seed=seed)
    raise ValueError(f"Unknown strategy: {strategy}")


def _classify_patch_type(action: str) -> str:
    """Classify patch action for EXP6 histogram."""
    if not action or not isinstance(action, str):
        return "unknown"
    a = action.lower()
    if a.startswith("search["):
        return "search"
    if a.startswith("click[buy now]") or "buy now" in a:
        return "buy"
    if "back to search" in a or "back to search" in a:
        return "backtrack"
    if "< prev" in a or "prev" in a:
        return "navigation"
    if "next >" in a or "next" in a:
        return "navigation"
    if a.startswith("click[item - "):
        return "item"
    if any(x in a for x in ["color", "size", "description", "features", "reviews"]):
        return "attribute"
    return "other"


def run_idt_experiment(
    *,
    failures: List[Dict[str, Any]],
    env,
    agent: Agent,
    strategy: str = "diagnosis",
    M: int = 3,
    patcher: str = "agent_topk",
    patch_k: int = 5,
    diagnosis_model: Optional[DiagnosisModelSelector] = None,
    diagnosis_window: int = 1,
    baseline_attempts: int = 3,
    patch_attempts: Optional[int] = None,
    paired_seeds: bool = True,
    max_steps: int = 50,
    allow_any_search: bool = False,
    allow_invalid_patch_actions: bool = False,
    stop_on_success: bool = True,
    greedy: bool = False,
    seed: int = 0,
    simulation_budget: int = 999999,
    verbose: bool = True,
    compute_matched_baseline_attempts: Optional[int] = None,  # EXP2: baseline gets A*C attempts
    use_random_patch: bool = False,  # EXP3: pick one random patch instead of best-of
    baseline_only_attempts: Optional[int] = None,  # EXP2: run only baseline with this many attempts (no patch)
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run IDT patch experiment. Returns (all_records, stats, training_samples).

    Coverage/denominator and compute counters are included in stats.
    """
    if patch_attempts is None:
        patch_attempts = baseline_attempts
    action_method = "greedy" if greedy else "softmax"

    patcher_obj = make_patcher(patcher, seed=seed)
    simulator = PatchSimulator(env, agent, max_steps=max_steps, debug=verbose)

    # Coverage counters
    coverage = {
        "N_traj_total": len(failures),
        "N_traj_processed": 0,
        "N_step_candidates_total": 0,
        "N_step_replay_ok": 0,
        "N_step_with_patch_candidates": 0,
        "skip_reasons": {
            "replay_failed": 0,
            "done_before_patch": 0,
            "no_alt_actions": 0,
            "invalid_patch_action": 0,
            "forced_not_used": 0,
            "exception_fallback": 0,
        },
        "forced_skip_reason_counts": {},
    }
    # Compute counters
    compute = {
        "env_steps_baseline": 0,
        "env_steps_patched": 0,
        "model_calls_baseline": 0,
        "model_calls_patched": 0,
        "number_of_patch_candidates_evaluated": 0,
    }

    all_records: List[Dict[str, Any]] = []
    training_samples: List[Dict[str, Any]] = []
    rollouts_used = 0

    for traj_idx, traj in enumerate(failures):
        if rollouts_used >= simulation_budget:
            break

        task_id = traj.get("task_id")
        goal = traj.get("goal", "")
        steps = traj.get("steps", [])
        traj_len = len(steps)
        original_reward = float(traj.get("reward", 0.0))
        coverage["N_traj_processed"] += 1

        candidate_steps, selection_info = _select_states(
            traj,
            strategy=strategy,
            M=M,
            agent=agent,
            diagnosis_model=diagnosis_model,
            diagnosis_window=diagnosis_window,
            seed=seed,
        )
        coverage["N_step_candidates_total"] += len(candidate_steps)

        if not candidate_steps:
            continue

        if verbose:
            print(f"\n[{traj_idx+1}/{len(failures)}] task_id={task_id} len={traj_len} orig_reward={original_reward:.0f}")
            print("  candidate_steps:", candidate_steps)

        for step_idx in candidate_steps:
            if rollouts_used >= simulation_budget:
                break
            if step_idx < 0 or step_idx >= traj_len:
                continue

            orig_action = steps[step_idx].get("action_taken") or steps[step_idx].get("action")
            replay = simulator.replay_prefix(traj, step_idx)

            if not replay.ok:
                coverage["skip_reasons"]["replay_failed"] += 1
                if "done" in (replay.error or "").lower() or "early" in (replay.error or "").lower():
                    coverage["skip_reasons"]["done_before_patch"] += 1
                all_records.append({
                    "task_id": task_id,
                    "recovery_step": step_idx,
                    "replay_ok": False,
                    "replay_error": replay.error,
                    "strategy": strategy,
                    "patcher": patcher,
                })
                continue

            coverage["N_step_replay_ok"] += 1
            state_obs = replay.obs
            state_info = replay.info
            state_goal = state_info.get("goal", goal)
            valid_actions = state_info.get("valid", []) or []

            base_seed = derive_base_seed(seed, task_id, traj_idx, step_idx)
            baseline_seeds = [base_seed + i for i in range(max(1, baseline_attempts))]
            patch_seeds = [base_seed + i for i in range(max(1, patch_attempts))]

            # Determine baseline attempt count (EXP2: compute-matched or baseline-only)
            actual_baseline_attempts = baseline_attempts
            if baseline_only_attempts is not None:
                actual_baseline_attempts = baseline_only_attempts
            elif compute_matched_baseline_attempts is not None:
                actual_baseline_attempts = compute_matched_baseline_attempts

            # ----- Baseline rollouts -----
            baseline_best_reward = -1.0
            baseline_best_success = False
            baseline_best_seed: Optional[int] = None
            baseline_env_steps = 0
            baseline_model_calls = 0
            baseline_first_actions: List[str] = []  # EXP2 validation: first action per rollout

            for a in range(actual_baseline_attempts):
                if rollouts_used >= simulation_budget:
                    break
                set_all_seeds(baseline_seeds[a % len(baseline_seeds)])
                out = simulator.rollout_from_state(traj, step_idx, method=action_method)
                rollouts_used += 1
                baseline_env_steps += getattr(out, "env_steps", 0) or len(out.sim_traj)
                baseline_model_calls += getattr(out, "model_calls", 0)
                # Log first action for diversity validation (EXP2)
                if out.sim_traj:
                    first_act = out.sim_traj[0].get("action_taken")
                    if first_act is not None:
                        baseline_first_actions.append(first_act)
                if out.reward > baseline_best_reward:
                    baseline_best_reward = out.reward
                    baseline_best_success = out.success
                    baseline_best_seed = baseline_seeds[a % len(baseline_seeds)]
                if stop_on_success and out.success:
                    break

            compute["env_steps_baseline"] += baseline_env_steps
            compute["model_calls_baseline"] += baseline_model_calls

            # EXP2 baseline-only: skip patch, patched = baseline; add first-action diversity
            if baseline_only_attempts is not None:
                unique_first_actions = len(set(baseline_first_actions))
                rec = {
                    "task_id": task_id,
                    "goal": state_goal[:500] if isinstance(state_goal, str) else state_goal,
                    "recovery_step": step_idx,
                    "traj_len": traj_len,
                    "original_reward": original_reward,
                    "replay_ok": True,
                    "baseline_best_reward": baseline_best_reward,
                    "baseline_success": baseline_best_success,
                    "patched_best_reward": baseline_best_reward,
                    "patched_success": baseline_best_success,
                    "best_patch_action": None,
                    "improvement": 0.0,
                    "strategy": strategy,
                    "patcher": patcher,
                    "baseline_only": True,
                    # EXP2 validation: ensure 15-rollout baseline explores
                    "baseline_attempts_used": len(baseline_first_actions),
                    "baseline_first_actions": baseline_first_actions,
                    "baseline_unique_first_actions": unique_first_actions,
                }
                all_records.append(rec)
                continue

            # ----- Patch proposals -----
            diag_resp = None
            if patcher == "diagnosis_text":
                srec = next((s for s in selection_info if s.get("state_idx") == step_idx), None)
                if srec:
                    diag_resp = srec.get("model_response")
            if patcher == "diagnosis_text" and diag_resp:
                proposals = patcher_obj.propose(
                    state_obs, state_goal, valid_actions,
                    original_action=orig_action, agent=agent, k=patch_k, diagnosis_response=diag_resp
                )
            else:
                proposals = patcher_obj.propose(
                    state_obs, state_goal, valid_actions,
                    original_action=orig_action, agent=agent, k=patch_k
                )

            num_proposals_raw = len(proposals)
            if not allow_invalid_patch_actions:
                proposals = [
                    p for p in proposals
                    if _is_action_valid_at_state(p.action, valid_actions, allow_any_search=allow_any_search)
                ]

            if not proposals:
                coverage["skip_reasons"]["no_alt_actions"] += 1
                all_records.append({
                    "task_id": task_id,
                    "goal": state_goal[:500] if isinstance(state_goal, str) else state_goal,
                    "recovery_step": step_idx,
                    "traj_len": traj_len,
                    "original_reward": original_reward,
                    "replay_ok": True,
                    "baseline_best_reward": baseline_best_reward,
                    "baseline_success": baseline_best_success,
                    "patched_best_reward": baseline_best_reward,
                    "patched_success": baseline_best_success,
                    "best_patch_action": None,
                    "improvement": 0.0,
                    "strategy": strategy,
                    "patcher": patcher,
                    "num_proposals_raw": num_proposals_raw,
                    "num_proposals_eval": 0,
                })
                continue

            coverage["N_step_with_patch_candidates"] += 1

            # ----- Evaluate patches -----
            if use_random_patch:
                import random as _r
                prop = _r.Random(base_seed).choice(proposals)
                proposals = [prop]

            patched_best_reward = -1.0
            patched_best_success = False
            best_patch_action: Optional[str] = None
            best_patch_meta: Optional[Dict[str, Any]] = None
            best_patch_valid: Optional[bool] = None
            best_patch_forced_used: Optional[bool] = None
            best_patch_forced_skip_reason: Optional[str] = None
            best_patch_executed_action0: Optional[str] = None
            patched_env_steps = 0
            patched_model_calls = 0

            for prop_idx, proposal in enumerate(proposals):
                if rollouts_used >= simulation_budget:
                    break
                action_to_force = proposal.action
                patch_valid = _is_action_valid_at_state(action_to_force, valid_actions, allow_any_search=allow_any_search)

                for pa in range(patch_attempts):
                    if rollouts_used >= simulation_budget:
                        break
                    s = patch_seeds[pa] if paired_seeds else base_seed + 10_000 + prop_idx * 101 + pa
                    set_all_seeds(s)
                    out = simulator.rollout_from_state(
                        traj, step_idx, method=action_method,
                        forced_first_action=action_to_force,
                        allow_any_search=allow_any_search,
                    )
                    rollouts_used += 1
                    compute["number_of_patch_candidates_evaluated"] += 1
                    patched_env_steps += getattr(out, "env_steps", 0) or len(out.sim_traj)
                    patched_model_calls += getattr(out, "model_calls", 0)

                    forced_used = _forced_was_used(out)
                    skip_reason = _get_forced_skip_reason(out)

                    if out.reward > patched_best_reward:
                        patched_best_reward = out.reward
                        patched_best_success = out.success
                        best_patch_action = action_to_force
                        best_patch_valid = patch_valid
                        best_patch_forced_used = forced_used
                        best_patch_forced_skip_reason = skip_reason
                        best_patch_executed_action0 = out.sim_traj[0].get("action_taken") if out.sim_traj else None
                        best_patch_meta = {
                            "proposal_score": proposal.score,
                            "patch_valid_at_state": patch_valid,
                            "forced_used": forced_used,
                            "forced_skip_reason": skip_reason,
                        }
                    if stop_on_success and out.success:
                        break
                if stop_on_success and patched_best_success:
                    break

            compute["env_steps_patched"] += patched_env_steps
            compute["model_calls_patched"] += patched_model_calls

            improvement = patched_best_reward - baseline_best_reward
            rec = {
                "task_id": task_id,
                "goal": state_goal[:500] if isinstance(state_goal, str) else state_goal,
                "recovery_step": step_idx,
                "traj_len": traj_len,
                "original_reward": original_reward,
                "replay_ok": True,
                "num_valid_actions": len(valid_actions),
                "baseline_best_reward": baseline_best_reward,
                "baseline_success": baseline_best_success,
                "patched_best_reward": patched_best_reward,
                "patched_success": patched_best_success,
                "best_patch_action": best_patch_action,
                "best_patch_meta": best_patch_meta,
                "best_patch_valid_at_state": best_patch_valid,
                "best_patch_forced_used": best_patch_forced_used,
                "best_patch_forced_skip_reason": best_patch_forced_skip_reason,
                "best_patch_executed_action0": best_patch_executed_action0,
                "improvement": improvement,
                "strategy": strategy,
                "patcher": patcher,
                "original_action": orig_action,
                "paired_seeds": paired_seeds,
                "num_proposals_raw": num_proposals_raw,
                "num_proposals_eval": len(proposals),
            }
            all_records.append(rec)

            if best_patch_action and (patched_best_success or improvement > 0):
                training_samples.append({
                    "state": state_obs,
                    "goal": state_goal,
                    "action": best_patch_action,
                    "valid_actions": valid_actions,
                    "task_id": task_id,
                    "recovery_step": step_idx,
                    "final_reward": patched_best_reward,
                    "source": "patch_success" if patched_best_success else "patch_improvement",
                    "original_action": orig_action,
                    "patch_type": _classify_patch_type(best_patch_action),
                })

    # Aggregate stats
    ok_records = [r for r in all_records if r.get("replay_ok")]
    n = len(ok_records)
    baseline_success_count = sum(1 for r in ok_records if r.get("baseline_success"))
    patched_success_count = sum(1 for r in ok_records if r.get("patched_success"))
    improved = sum(1 for r in ok_records if r.get("improvement") is not None and r.get("improvement") > 0)
    baseline_failed = [r for r in ok_records if not r.get("baseline_success")]
    rescue_count = sum(1 for r in baseline_failed if r.get("patched_success"))
    rescue_rate = float(rescue_count) / max(len(baseline_failed), 1)
    baseline_succeeded = [r for r in ok_records if r.get("baseline_success")]
    break_count = sum(1 for r in baseline_succeeded if not r.get("patched_success"))
    break_rate = float(break_count) / max(len(baseline_succeeded), 1)
    deltas = [r["improvement"] for r in ok_records if r.get("improvement") is not None]
    avg_improvement = float(sum(deltas) / len(deltas)) if deltas else 0.0
    median_improvement = float(np.median(deltas)) if deltas else 0.0

    # EXP2 validation: aggregate unique-first-action diversity (baseline_only runs)
    baseline_only_records = [r for r in all_records if r.get("baseline_only")]
    if baseline_only_records:
        diversity_counts = [r["baseline_unique_first_actions"] for r in baseline_only_records if r.get("baseline_unique_first_actions") is not None]
        attempts_used = [r.get("baseline_attempts_used") for r in baseline_only_records if r.get("baseline_attempts_used") is not None]
        coverage["exp2_baseline_first_action_diversity"] = {
            "n_steps": len(baseline_only_records),
            "mean_unique_first_actions": float(np.mean(diversity_counts)) if diversity_counts else 0,
            "min_unique_first_actions": int(min(diversity_counts)) if diversity_counts else 0,
            "max_unique_first_actions": int(max(diversity_counts)) if diversity_counts else 0,
            "mean_attempts_used": float(np.mean(attempts_used)) if attempts_used else 0,
            "histogram_unique_first_actions": {int(k): int(v) for k, v in zip(*np.unique(diversity_counts, return_counts=True))} if diversity_counts else {},
        }

    with_patch = [r for r in ok_records if r.get("best_patch_action") is not None]
    n_with_patch = len(with_patch)
    patch_valid_count = sum(1 for r in with_patch if r.get("best_patch_valid_at_state"))
    forced_used_count = sum(1 for r in with_patch if r.get("best_patch_forced_used"))
    # Count forced_skip_reason when forced was not used
    for r in with_patch:
        if not r.get("best_patch_forced_used") and r.get("best_patch_forced_skip_reason"):
            coverage["skip_reasons"]["forced_not_used"] += 1
            rs = r["best_patch_forced_skip_reason"]
            coverage["forced_skip_reason_counts"][rs] = coverage["forced_skip_reason_counts"].get(rs, 0) + 1

    stats = {
        "failures_processed": coverage["N_traj_processed"],
        "steps_evaluated": coverage["N_step_candidates_total"],
        "replay_ok_steps": n,
        "rollouts_used": rollouts_used,
        "simulation_budget": simulation_budget,
        "baseline_attempts": baseline_attempts,
        "patch_attempts": patch_attempts,
        "paired_seeds": paired_seeds,
        "baseline_success_rate": float(baseline_success_count) / max(n, 1),
        "patched_success_rate": float(patched_success_count) / max(n, 1),
        "improvement_rate": float(improved) / max(n, 1),
        "avg_improvement": avg_improvement,
        "median_improvement": median_improvement,
        "rescue_rate": rescue_rate,
        "rescue_count": rescue_count,
        "n_baseline_failed": len(baseline_failed),
        "break_rate": break_rate,
        "break_count": break_count,
        "n_baseline_succeeded": len(baseline_succeeded),
        "n_with_patch": n_with_patch,
        "patch_valid_count": patch_valid_count,
        "patch_valid_rate": float(patch_valid_count) / max(n_with_patch, 1),
        "forced_used_count": forced_used_count,
        "forced_used_rate": float(forced_used_count) / max(n_with_patch, 1),
        "strategy": strategy,
        "patcher": patcher,
        "patch_k": patch_k,
        "coverage": coverage,
        "compute": compute,
        "simulator_stats": simulator.stats,
        "fairness": {
            "baseline_attempts": baseline_attempts,
            "patch_attempts": patch_attempts,
            "paired_seeds": paired_seeds,
        },
    }

    return all_records, stats, training_samples
