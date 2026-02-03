#!/usr/bin/env python3
"""
run_idt_patch.py

IDT-style "patch teachability" evaluation on top of your current EEF pipeline.

Concept (simple):
  - Pick candidate steps in each failed trajectory (baseline/entropy/diagnosis).
  - At each step:
      (a) baseline: restart from that step and let the agent explore.
      (b) patched: restart from that step, FORCE an alternative first action, then let the agent explore.
  - Compare success/reward with fixed compute budgets.

This operationalizes the question:
  "Is this step a teachable moment *because a small patch at this step can recover*?"

Outputs:
  - patch_results_*.json: per-(trajectory, step) evaluation records
  - patch_stats_*.json: summary statistics
  - patch_training_samples_*.json: (state, goal, action, valid_actions) samples for patch-supervision
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------------------------------------------------------
# NOTE ON PATHS
#
# This code is intended to live under:
#   baseline_models/idt_teachability/
# alongside:
#   baseline_models/ckpts/
#   baseline_models/simulation/Qwen2.5/qwen25_instruct_v1/
#
# Users may run this script from either the repo root or baseline_models/.
# We therefore add BOTH the repo root and baseline_models/ to sys.path.
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # baseline_models/
REPO_ROOT = os.path.abspath(os.path.join(BASELINE_DIR, ".."))  # repo root

# Prefer local baseline_models modules (eef_detailed_with_diagnosis.py is often stored there)
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from idt_teachability.patch_simulator import PatchSimulator
from idt_teachability.patchers import make_patcher

# Reuse your existing environment/model setup + state selectors
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


def load_failures(path: str) -> List[Dict[str, Any]]:
    """Load a failure dataset.

    Supports:
      - JSON list: [ {traj}, {traj}, ... ]
      - JSON dict with a top-level key like 'failures' or 'trajectories'
      - JSONL (one JSON object per line) if file ends with .jsonl
    """
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
        # Fallback: if dict maps ids -> trajectories
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())

    raise ValueError(
        f"Unsupported failure dataset format at {path}. "
        "Expected JSON list/dict or JSONL."
    )


def _select_states(
    trajectory: Dict[str, Any],
    *,
    strategy: str,
    M: int,
    agent: Agent,
    diagnosis_model: Optional[DiagnosisModelSelector] = None,
    diagnosis_window: int = 1,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    strategy = strategy.lower().strip()
    if strategy == "diagnosis":
        if diagnosis_model is None:
            raise ValueError("diagnosis_model is required for strategy=diagnosis")
        return select_critical_states_diagnosis(
            trajectory, M=M, diagnosis_model=diagnosis_model, window=diagnosis_window, agent=agent
        )
    if strategy == "entropy":
        return select_critical_states_entropy(trajectory, M=M, agent=agent)
    if strategy == "stratified_entropy":
        return select_critical_states_stratified_entropy(trajectory, M=M, agent=agent)
    return select_critical_states_baseline(trajectory, M=M, agent=agent)


def main():
    parser = argparse.ArgumentParser(description="IDT Patch Teachability Evaluation (on top of EEF/WebShop)")
    parser.add_argument("--failure_data", type=str, required=True, help="Path to pre-collected failure trajectories (json)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(THIS_DIR, "outputs"),
        help="Where to write outputs (default: baseline_models/idt_teachability/outputs)",
    )

    # State selection
    parser.add_argument("--strategy", type=str, default="diagnosis",
                        choices=["baseline", "entropy", "stratified_entropy", "diagnosis"],
                        help="How to select candidate steps inside a failure trajectory")
    parser.add_argument("--M", type=int, default=3, help="Number of candidate steps per trajectory")

    # Agent checkpoint
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(BASELINE_DIR, "ckpts", "web_click", "epoch_9", "model.pth"),
        help="Path to BERT click model (default: baseline_models/ckpts/web_click/epoch_9/model.pth)",
    )

    # Diagnosis model (optional)
    parser.add_argument(
        "--diagnosis_model_path",
        type=str,
        default=os.path.join(BASELINE_DIR, "simulation", "Qwen2.5", "qwen25_instruct_v1"),
        help=(
            "Path to LoRA adapter for diagnosis model. "
            "Default matches repo layout: baseline_models/simulation/Qwen2.5/qwen25_instruct_v1"
        ),
    )
    parser.add_argument("--diagnosis_base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name for diagnosis model")
    parser.add_argument("--diagnosis_window", type=int, default=1,
                        help="If strategy=diagnosis, evaluate steps in [pred-window, pred+window]")

    # Rollout budgets
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per rollout after the intervention step")
    parser.add_argument("--baseline_attempts", type=int, default=3, help="How many baseline rollouts per (traj, step)")
    parser.add_argument("--patch_attempts", type=int, default=1, help="How many rollouts per proposed patch action")
    parser.add_argument("--simulation_budget", type=int, default=999999, help="Total rollout budget across dataset")

    # Action sampling
    parser.add_argument("--greedy", action="store_true", default=False, help="Use greedy instead of softmax sampling")
    parser.add_argument("--stop_on_success", action="store_true", default=True, help="Stop early when a rollout succeeds")

    # Patcher
    parser.add_argument("--patcher", type=str, default="agent_topk",
                        choices=["agent_topk", "random", "diagnosis_text"],
                        help="How to propose alternative first actions")
    parser.add_argument("--patch_k", type=int, default=5, help="How many patch actions to evaluate per step")
    parser.add_argument("--allow_any_search", action="store_true", default=False,
                        help="Allow forced search[query] even if not listed in info['valid'].")

    # Book-keeping
    parser.add_argument("--num_trajectories", type=int, default=None, help="Process only first N trajectories")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_full_trajectories", action="store_true", default=False,
                        help="Include full replay+rollout trajectories in outputs (large files)")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    action_method = "greedy" if args.greedy else "softmax"

    print("=" * 80)
    print("IDT PATCH TEACHABILITY EVALUATION")
    print("=" * 80)
    print(f"Failure data:      {args.failure_data}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Strategy:          {args.strategy} (M={args.M})")
    print(f"Student model:     {args.model_path}")
    print(f"Patcher:           {args.patcher} (k={args.patch_k})")
    print(f"Rollouts:          baseline_attempts={args.baseline_attempts}, patch_attempts={args.patch_attempts}")
    print(f"Action method:     {action_method}")
    print(f"Budget:            {args.simulation_budget}")
    print("=" * 80)

    # Setup environment + student
    env = setup_environment()
    models = setup_model(args.model_path)
    agent = Agent(models)

    # Diagnosis model if needed
    diagnosis_model = None
    if args.strategy == "diagnosis" or args.patcher == "diagnosis_text":
        if args.diagnosis_model_path is None:
            raise ValueError("--diagnosis_model_path required when using diagnosis strategy or diagnosis_text patcher")
        if not os.path.exists(args.diagnosis_model_path):
            raise FileNotFoundError(
                f"Diagnosis model path not found: {args.diagnosis_model_path}. "
                "Expected repo layout: baseline_models/simulation/Qwen2.5/qwen25_instruct_v1 "
                "(or pass --diagnosis_model_path explicitly)."
            )
        diagnosis_model = DiagnosisModelSelector(args.diagnosis_model_path, args.diagnosis_base_model)

    patcher = make_patcher(args.patcher, seed=args.seed)
    simulator = PatchSimulator(env, agent, max_steps=args.max_steps, debug=args.verbose)

    # Load failures
    failures = load_failures(args.failure_data)
    if args.num_trajectories is not None:
        failures = failures[: args.num_trajectories]

    # Results
    all_records: List[Dict[str, Any]] = []
    training_samples: List[Dict[str, Any]] = []

    rollouts_used = 0
    total_steps_evaluated = 0

    for traj_idx, traj in enumerate(failures):
        if rollouts_used >= args.simulation_budget:
            break

        task_id = traj.get("task_id")
        goal = traj.get("goal", "")
        steps = traj.get("steps", [])
        traj_len = len(steps)
        original_reward = float(traj.get("reward", 0.0))

        # Pick candidate steps
        candidate_steps, selection_info = _select_states(
            traj,
            strategy=args.strategy,
            M=args.M,
            agent=agent,
            diagnosis_model=diagnosis_model,
            diagnosis_window=args.diagnosis_window,
        )

        if not candidate_steps:
            continue

        if args.verbose:
            print(f"\n[{traj_idx+1}/{len(failures)}] task_id={task_id} len={traj_len} orig_reward={original_reward:.0f}")
            print("  candidate_steps:", candidate_steps)
            if args.strategy == "diagnosis" and selection_info:
                pred = selection_info[0].get("predicted_mistake_step", "?")
                print("  diagnosis prediction:", pred)

        for step_idx in candidate_steps:
            if rollouts_used >= args.simulation_budget:
                break
            if step_idx < 0 or step_idx >= traj_len:
                continue

            total_steps_evaluated += 1

            # Gather original action (to exclude from patch proposals)
            orig_action = steps[step_idx].get("action_taken")
            if orig_action is None:
                orig_action = steps[step_idx].get("action")

            # We need state obs + valid actions at that state to propose patches.
            replay = simulator.replay_prefix(traj, step_idx)
            if not replay.ok:
                # Can't evaluate this step
                rec = {
                    "task_id": task_id,
                    "recovery_step": step_idx,
                    "traj_len": traj_len,
                    "original_reward": original_reward,
                    "replay_ok": False,
                    "replay_error": replay.error,
                    "baseline_best_reward": None,
                    "baseline_success": False,
                    "patched_best_reward": None,
                    "patched_success": False,
                    "best_patch_action": None,
                    "improvement": None,
                    "strategy": args.strategy,
                    "patcher": args.patcher,
                }
                all_records.append(rec)
                continue

            state_obs = replay.obs
            state_info = replay.info
            state_goal = state_info.get("goal", goal)
            valid_actions = state_info.get("valid", []) or []

            # Baseline rollouts
            baseline_best_reward = -1.0
            baseline_best_success = False

            for a in range(args.baseline_attempts):
                if rollouts_used >= args.simulation_budget:
                    break
                out = simulator.rollout_from_state(traj, step_idx, method=action_method)
                rollouts_used += 1

                if out.reward > baseline_best_reward:
                    baseline_best_reward = out.reward
                    baseline_best_success = out.success

                if args.stop_on_success and out.success:
                    break

            # Patch proposals
            # If we have diagnosis info, attach the model response for diagnosis_text patcher.
            diag_resp = None
            if args.patcher == "diagnosis_text":
                # Find selection_info record for this step (if any)
                # In select_critical_states_diagnosis, only the predicted step has model_response populated.
                srec = next((s for s in selection_info if s.get("state_idx") == step_idx), None)
                if srec:
                    diag_resp = srec.get("model_response") or None

                proposals = patcher.propose(
                    state_obs, state_goal, valid_actions,
                    original_action=orig_action,
                    agent=agent,
                    k=args.patch_k,
                    diagnosis_response=diag_resp,
                )
            else:
                proposals = patcher.propose(
                    state_obs, state_goal, valid_actions,
                    original_action=orig_action,
                    agent=agent,
                    k=args.patch_k,
                )

            patched_best_reward = -1.0
            patched_best_success = False
            best_patch_action = None
            best_patch_meta = None

            for proposal in proposals:
                if rollouts_used >= args.simulation_budget:
                    break

                action_to_force = proposal.action

                # Run patch_attempts rollouts for this forced action
                local_best_reward = -1.0
                local_best_success = False

                for pa in range(args.patch_attempts):
                    if rollouts_used >= args.simulation_budget:
                        break
                    out = simulator.rollout_from_state(
                        traj, step_idx, method=action_method,
                        forced_first_action=action_to_force,
                        allow_any_search=args.allow_any_search,
                    )
                    rollouts_used += 1

                    if out.reward > local_best_reward:
                        local_best_reward = out.reward
                        local_best_success = out.success

                    if args.stop_on_success and out.success:
                        break

                # Compare against global patched best
                if local_best_reward > patched_best_reward:
                    patched_best_reward = local_best_reward
                    patched_best_success = local_best_success
                    best_patch_action = action_to_force
                    best_patch_meta = {"proposal_score": proposal.score, "proposal_meta": proposal.meta}

                if args.stop_on_success and patched_best_success:
                    break

            # If there were no proposals, mark patched as baseline (no extra)
            if not proposals:
                patched_best_reward = baseline_best_reward
                patched_best_success = baseline_best_success

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
                "improvement": improvement,
                "strategy": args.strategy,
                "patcher": args.patcher,
                "original_action": orig_action,
            }

            # Keep selection details if available
            sel = next((s for s in selection_info if s.get("state_idx") == step_idx), None)
            if sel:
                for k in ["predicted_mistake_step", "offset_from_prediction", "is_predicted_step", "true_entropy", "normalized_entropy", "combined_score", "method"]:
                    if k in sel:
                        rec[k] = sel[k]

            if args.save_full_trajectories:
                # WARNING: big outputs
                rec["state_observation"] = state_obs[:2000]
                rec["valid_actions"] = valid_actions

            all_records.append(rec)

            # Training sample: only if patched beats baseline (or achieves success)
            if best_patch_action is not None and (patched_best_success or improvement > 0.0):
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
                    "baseline_best_reward": baseline_best_reward,
                    "patched_best_reward": patched_best_reward,
                })

            if args.verbose:
                print(f"  step={step_idx:>2} baseline={baseline_best_reward:>5.0f} patched={patched_best_reward:>5.0f} "
                      f"Δ={improvement:>+5.0f} patch={best_patch_action}")

    # Aggregate stats
    n = len([r for r in all_records if r.get("replay_ok")])
    baseline_success = sum(1 for r in all_records if r.get("replay_ok") and r.get("baseline_success"))
    patched_success = sum(1 for r in all_records if r.get("replay_ok") and r.get("patched_success"))
    improved = sum(1 for r in all_records if r.get("replay_ok") and (r.get("improvement") is not None and r.get("improvement") > 0))

    avg_improvement = 0.0
    if n > 0:
        deltas = [r["improvement"] for r in all_records if r.get("replay_ok") and r.get("improvement") is not None]
        if deltas:
            avg_improvement = sum(deltas) / len(deltas)

    stats = {
        "failures_processed": len(failures),
        "steps_evaluated": total_steps_evaluated,
        "replay_ok_steps": n,
        "rollouts_used": rollouts_used,
        "simulation_budget": args.simulation_budget,
        "baseline_success_rate": float(baseline_success) / max(n, 1),
        "patched_success_rate": float(patched_success) / max(n, 1),
        "improvement_rate": float(improved) / max(n, 1),
        "avg_improvement": avg_improvement,
        "strategy": args.strategy,
        "patcher": args.patcher,
        "baseline_attempts": args.baseline_attempts,
        "patch_attempts": args.patch_attempts,
        "patch_k": args.patch_k,
        "action_method": action_method,
        "simulator_stats": simulator.stats,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"patch_results_{args.strategy}_{args.patcher}_{timestamp}.json")
    stats_path = os.path.join(args.output_dir, f"patch_stats_{args.strategy}_{args.patcher}_{timestamp}.json")
    train_path = os.path.join(args.output_dir, f"patch_training_samples_{args.strategy}_{args.patcher}_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump(all_records, f, indent=2)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    with open(train_path, "w") as f:
        json.dump(training_samples, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Results:", results_path)
    print("Stats:  ", stats_path)
    print("Train:  ", train_path)
    print("\nSummary:")
    print(f"  replay-ok steps:  {n}")
    print(f"  baseline success: {stats['baseline_success_rate']:.2%}")
    print(f"  patched success:  {stats['patched_success_rate']:.2%}")
    print(f"  improved:         {stats['improvement_rate']:.2%}")
    print(f"  avg Δ reward:     {stats['avg_improvement']:.2f}")


if __name__ == "__main__":
    main()
