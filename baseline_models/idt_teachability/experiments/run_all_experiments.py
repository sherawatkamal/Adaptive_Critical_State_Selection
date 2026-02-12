#!/usr/bin/env python3
"""
Run full IDT experiment suite: EXP0-EXP7.

Usage:
  python -m baseline_models.idt_teachability.experiments.run_all_experiments --failure_data <path> --num_trajectories 500
  python -m baseline_models.idt_teachability.experiments.run_all_experiments --failure_data <path> --smoke
  python -m baseline_models.idt_teachability.experiments.run_all_experiments --failure_data <path> --fast --num_trajectories 100
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure baseline_models and repo root are on path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IDT_DIR = os.path.dirname(THIS_DIR)
BASELINE_DIR = os.path.dirname(IDT_DIR)
REPO_ROOT = os.path.dirname(BASELINE_DIR)
for p in [BASELINE_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from idt_teachability.idt_core import (
    load_failures,
    run_idt_experiment,
    set_all_seeds,
)
from eef_detailed_with_diagnosis import (
    setup_environment,
    setup_model,
    Agent,
    DiagnosisModelSelector,
)
from idt_teachability.step_selectors import (
    select_steps_last_n,
    select_steps_search_only,
    select_steps_random,
)
from idt_teachability.patchers import make_patcher
from idt_teachability.patch_simulator import PatchSimulator
from idt_teachability.idt_core import _is_action_valid_at_state, _classify_patch_type, derive_base_seed


def _default_model_path() -> str:
    return os.path.join(BASELINE_DIR, "ckpts", "web_click", "epoch_9", "model.pth")


def _default_diagnosis_path() -> str:
    return os.path.join(BASELINE_DIR, "simulation", "Qwen2.5", "qwen25_instruct_v1")


def _output_dir() -> str:
    return os.path.join(IDT_DIR, "outputs")


def _plots_dir() -> str:
    return os.path.join(_output_dir(), "plots")


def _save_exp_outputs(
    tag: str,
    records: List[Dict],
    stats: Dict,
    training_samples: List[Dict],
    output_dir: str,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"EXP_{tag}_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"patch_results_{prefix}.json")
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {results_path}")

    stats_path = os.path.join(output_dir, f"patch_stats_{prefix}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {stats_path}")

    train_path = os.path.join(output_dir, f"patch_training_samples_{prefix}.json")
    with open(train_path, "w") as f:
        json.dump(training_samples, f, indent=2)
    print(f"  Saved {train_path}")

    csv_path = os.path.join(output_dir, f"summary_{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "replay_ok_steps", "baseline_success_rate", "patched_success_rate", "rescue_rate", "break_rate", "avg_improvement"])
        w.writerow([
            tag,
            stats.get("replay_ok_steps", 0),
            stats.get("baseline_success_rate", 0),
            stats.get("patched_success_rate", 0),
            stats.get("rescue_rate", 0),
            stats.get("break_rate", 0),
            stats.get("avg_improvement", 0),
        ])
    print(f"  Saved {csv_path}")


def run_exp0_smoke(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Any]:
    """EXP0: Smoke test N=10."""
    print("\n" + "=" * 60)
    print("EXP0: Smoke test (N=10)")
    print("=" * 60)
    sub = failures[:10]
    records, stats, training = run_idt_experiment(
        failures=sub,
        env=env,
        agent=agent,
        strategy="diagnosis",
        M=2,
        patcher="agent_topk",
        patch_k=3,
        diagnosis_model=diagnosis_model,
        diagnosis_window=1,
        baseline_attempts=1,
        patch_attempts=1,
        paired_seeds=True,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=True,
    )
    _save_exp_outputs("EXP0_smoke", records, stats, training, args.output_dir)
    return {"records": records, "stats": stats, "training": training}


def run_exp1_main(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Any]:
    """EXP1: IDT best-of-candidates (main setting)."""
    print("\n" + "=" * 60)
    print("EXP1: IDT best-of-candidates")
    print("=" * 60)
    records, stats, training = run_idt_experiment(
        failures=failures,
        env=env,
        agent=agent,
        strategy="diagnosis",
        M=args.M,
        patcher="agent_topk",
        patch_k=5,
        diagnosis_model=diagnosis_model,
        diagnosis_window=1,
        baseline_attempts=3,
        patch_attempts=3,
        paired_seeds=True,
        max_steps=args.max_steps,
        seed=args.seed,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
    )
    _save_exp_outputs("EXP1_best_of", records, stats, training, args.output_dir)
    return {"records": records, "stats": stats, "training": training}


def run_exp2_compute_matched(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Any]:
    """EXP2: Compute-matched baseline (baseline gets A + A*C attempts to match IDT compute)."""
    print("\n" + "=" * 60)
    print("EXP2: Compute-matched baseline")
    print("=" * 60)
    patch_k = 5
    patch_attempts = 3
    # Compute-matched: baseline gets A*C = 3*5 = 15 attempts (same as patch side)
    baseline_only_attempts = patch_attempts * patch_k
    records, stats, training = run_idt_experiment(
        failures=failures,
        env=env,
        agent=agent,
        strategy="diagnosis",
        M=args.M,
        patcher="agent_topk",
        patch_k=patch_k,
        diagnosis_model=diagnosis_model,
        diagnosis_window=1,
        baseline_attempts=3,
        patch_attempts=patch_attempts,
        paired_seeds=True,
        max_steps=args.max_steps,
        seed=args.seed,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
        baseline_only_attempts=baseline_only_attempts,
    )
    _save_exp_outputs("EXP2_compute_matched", records, stats, training, args.output_dir)
    # EXP2 validation: print unique-first-action diversity (15-rollout baseline explores)
    cov = stats.get("coverage", {})
    div = cov.get("exp2_baseline_first_action_diversity")
    if div:
        print("\n  [EXP2 validation] Baseline first-action diversity (15 rollouts/step):")
        print(f"    mean unique first actions per step: {div.get('mean_unique_first_actions', 0):.2f}")
        print(f"    min / max unique first actions:      {div.get('min_unique_first_actions', 0)} / {div.get('max_unique_first_actions', 0)}")
        print(f"    mean attempts used (early stop):     {div.get('mean_attempts_used', 0):.2f}")
        if div.get("histogram_unique_first_actions"):
            print(f"    histogram (unique_count -> n_steps): {div['histogram_unique_first_actions']}")
    return {"records": records, "stats": stats, "training": training}


def run_exp3_random_patch(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Any]:
    """EXP3: Random patch (one random candidate) vs best-of."""
    print("\n" + "=" * 60)
    print("EXP3: Random patch ablation")
    print("=" * 60)
    records, stats, training = run_idt_experiment(
        failures=failures,
        env=env,
        agent=agent,
        strategy="diagnosis",
        M=args.M,
        patcher="agent_topk",
        patch_k=5,
        diagnosis_model=diagnosis_model,
        diagnosis_window=1,
        baseline_attempts=3,
        patch_attempts=3,
        paired_seeds=True,
        max_steps=args.max_steps,
        seed=args.seed,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
        use_random_patch=True,
    )
    _save_exp_outputs("EXP3_random_patch", records, stats, training, args.output_dir)
    return {"records": records, "stats": stats, "training": training}


def run_exp4_patch_k_scaling(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Dict]:
    """EXP4: patch_k in {1,2,5,10}."""
    print("\n" + "=" * 60)
    print("EXP4: Patch K scaling")
    print("=" * 60)
    results = {}
    for k in [1, 2, 5, 10]:
        print(f"\n--- patch_k={k} ---")
        records, stats, training = run_idt_experiment(
            failures=failures,
            env=env,
            agent=agent,
            strategy="diagnosis",
            M=args.M,
            patcher="agent_topk",
            patch_k=k,
            diagnosis_model=diagnosis_model,
            diagnosis_window=1,
            baseline_attempts=3,
            patch_attempts=3,
            paired_seeds=True,
            max_steps=args.max_steps,
            seed=args.seed,
            simulation_budget=args.simulation_budget,
            verbose=args.verbose,
        )
        _save_exp_outputs(f"EXP4_k{k}", records, stats, training, args.output_dir)
        results[f"k{k}"] = {"records": records, "stats": stats}
    return results


def run_exp5_step_selectors(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Dict]:
    """EXP5: Step selection ablations."""
    print("\n" + "=" * 60)
    print("EXP5: Step selector ablations")
    print("=" * 60)
    results = {}
    for strategy in ["diagnosis", "last_n", "search_steps", "random_steps", "baseline"]:
        M = 8 if strategy == "last_n" else (10 if strategy == "search_steps" else 3)
        print(f"\n--- strategy={strategy} M={M} ---")
        records, stats, training = run_idt_experiment(
            failures=failures,
            env=env,
            agent=agent,
            strategy=strategy,
            M=M,
            patcher="agent_topk",
            patch_k=5,
            diagnosis_model=diagnosis_model if strategy == "diagnosis" else None,
            diagnosis_window=1,
            baseline_attempts=3,
            patch_attempts=3,
            paired_seeds=True,
            max_steps=args.max_steps,
            seed=args.seed,
            simulation_budget=args.simulation_budget,
            verbose=args.verbose,
        )
        _save_exp_outputs(f"EXP5_{strategy}", records, stats, training, args.output_dir)
        results[strategy] = {"records": records, "stats": stats}
    return results


def _build_annotated_success_trajectory(
    full_traj: List[Dict],
    patch_step: int,
    original_action: Optional[str],
    patch_action: str,
    task_id: Any,
    goal: str,
    max_obs_len: int = 4000,
) -> Dict[str, Any]:
    """
    Build an annotated full trajectory from rollout full_traj, marking the step
    where the patch was applied and what changed (original_action -> patch_action).
    """
    # full_traj: [ {step:-1, observation, action_taken:None}, {step:0,...}, ..., {step:patch_step-1,...}, then sim {step:0,...}, {step:1,...} ]
    # So len(replay) = patch_step + 1 (indices 0..patch_step). Index 0 is initial obs; 1..patch_step are after actions 0..patch_step-1.
    # Then sim steps start at full_traj[patch_step+1] with step=0 (global = patch_step), etc.
    annotated_steps = []
    for i, rec in enumerate(full_traj):
        if i == 0:
            global_step = -1
            obs = rec.get("observation", "")
            action_taken = rec.get("action_taken")
        else:
            if i <= patch_step:
                global_step = i - 1  # replay steps 0..patch_step-1
            else:
                sim_idx = i - patch_step - 1
                global_step = patch_step + sim_idx
            obs = rec.get("observation", "")
            action_taken = rec.get("action_taken")

        if len(obs) > max_obs_len:
            obs = obs[:max_obs_len] + "\n... [truncated]"

        is_patch_step = global_step == patch_step
        step_entry = {
            "step_index": global_step,
            "observation": obs,
            "action_taken": action_taken,
            "reward": rec.get("reward", 0.0),
            "done": rec.get("done", False),
            "is_patch_step": is_patch_step,
        }
        if is_patch_step:
            step_entry["patch_info"] = {
                "original_action": original_action,
                "patch_action": patch_action,
                "description": f"At step {patch_step}: changed action from '{original_action}' to '{patch_action}' (this intervention made the trajectory succeed).",
            }
        annotated_steps.append(step_entry)

    return {
        "task_id": task_id,
        "goal": goal[:500] if isinstance(goal, str) else goal,
        "patch_step": patch_step,
        "original_action_at_patch_step": original_action,
        "patch_action_at_patch_step": patch_action,
        "summary": f"Step {patch_step} was changed from '{original_action}' to '{patch_action}'; trajectory then succeeded.",
        "full_trajectory": annotated_steps,
        "num_steps": len(annotated_steps),
    }


def run_exp6_trajectory_discovery(failures: List[Dict], env, agent, diagnosis_model, args) -> Dict[str, Any]:
    """EXP6: Trajectory-level teachable moment discovery."""
    print("\n" + "=" * 60)
    print("EXP6: Trajectory-level discovery")
    print("=" * 60)

    from idt_teachability.idt_core import _select_states

    patcher_obj = make_patcher("agent_topk", seed=args.seed)
    simulator = PatchSimulator(env, agent, max_steps=args.max_steps, debug=False)
    reliability_threshold = 2 / 3
    n_traj = min(len(failures), args.num_trajectories or 500)
    discovered = []
    t_star_dist = []
    patch_type_counts = {}

    for traj_idx, traj in enumerate(failures[:n_traj]):
        task_id = traj.get("task_id")
        goal = traj.get("goal", "")
        steps = traj.get("steps", [])

        diag_steps, _ = _select_states(
            traj, strategy="diagnosis", M=3, agent=agent,
            diagnosis_model=diagnosis_model, diagnosis_window=1, seed=args.seed
        )
        last_steps, _ = select_steps_last_n(traj, M=8)
        search_steps, _ = select_steps_search_only(traj, M=5)
        candidate_steps = sorted(set(diag_steps + last_steps + search_steps))
        candidate_steps = [s for s in candidate_steps if 0 <= s < len(steps) - 1]

        best_overall = None

        for step_idx in candidate_steps:
            orig_action = steps[step_idx].get("action_taken") or steps[step_idx].get("action")
            replay = simulator.replay_prefix(traj, step_idx)
            if not replay.ok:
                continue
            valid_actions = replay.info.get("valid", []) or []
            proposals = patcher_obj.propose(replay.obs, goal, valid_actions, original_action=orig_action, agent=agent, k=5)
            proposals = [p for p in proposals if _is_action_valid_at_state(p.action, valid_actions, allow_any_search=False)]
            if not proposals:
                continue

            base_seed = derive_base_seed(args.seed, task_id, traj_idx, step_idx)
            baseline_best = -1.0
            for pa in range(2):
                set_all_seeds(base_seed + pa)
                out = simulator.rollout_from_state(traj, step_idx, method="softmax")
                if out.reward > baseline_best:
                    baseline_best = out.reward

            best_reward = -1.0
            best_action = None
            reliable_count = 0
            for prop in proposals[:3]:
                success_count = 0
                for pa in range(2):
                    set_all_seeds(base_seed + 100 + pa)
                    out = simulator.rollout_from_state(traj, step_idx, method="softmax", forced_first_action=prop.action)
                    if out.success:
                        success_count += 1
                    if out.reward > best_reward:
                        best_reward = out.reward
                        best_action = prop.action
                if success_count / 2 >= reliability_threshold:
                    reliable_count += 1

            if best_action and best_reward > baseline_best:
                pt = _classify_patch_type(best_action)
                patch_type_counts[pt] = patch_type_counts.get(pt, 0) + 1
                if best_overall is None or best_reward > best_overall[2]:
                    best_overall = (step_idx, best_action, best_reward, replay.obs[:1000] if replay.ok else "")

        if best_overall:
            t_final, a_final, best_reward, obs_snip = best_overall
            t_star_dist.append(t_final)
            orig_action = steps[t_final].get("action_taken") or steps[t_final].get("action")
            discovered.append({
                "goal": goal[:300],
                "obs": obs_snip,
                "step": t_final,
                "patch_action": a_final,
                "patch_type": _classify_patch_type(a_final),
                "task_id": task_id,
            })

            # Run one successful rollout to capture full trajectory with patch applied
            full_success_trajectories = getattr(args, "save_full_success_trajectories", True)
            if full_success_trajectories:
                success_out = None
                seed_for_full = derive_base_seed(args.seed, task_id, traj_idx, t_final)
                for attempt in range(5):
                    set_all_seeds(seed_for_full + 1000 + attempt)
                    out = simulator.rollout_from_state(
                        traj, t_final, method="softmax", forced_first_action=a_final
                    )
                    if out.success and out.full_traj:
                        success_out = out
                        break
                if success_out and success_out.full_traj:
                    annotated = _build_annotated_success_trajectory(
                        success_out.full_traj,
                        patch_step=t_final,
                        original_action=orig_action,
                        patch_action=a_final,
                        task_id=task_id,
                        goal=goal,
                    )
                    discovered[-1]["full_success_trajectory"] = annotated

    teachable_rate = len(discovered) / max(n_traj, 1)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    jl_path = os.path.join(output_dir, "idt_discovered_patches.jsonl")
    with open(jl_path, "w") as f:
        for d in discovered:
            # Serialize without full_success_trajectory for the compact file
            d_compact = {k: v for k, v in d.items() if k != "full_success_trajectory"}
            f.write(json.dumps(d_compact) + "\n")
    print(f"  Saved {jl_path}")

    # Save full successful trajectories (with patch step and change clearly marked)
    success_traj_path = os.path.join(output_dir, "exp6_successful_trajectories.jsonl")
    count_saved = 0
    with open(success_traj_path, "w") as f:
        for d in discovered:
            if d.get("full_success_trajectory") is not None:
                f.write(json.dumps(d["full_success_trajectory"]) + "\n")
                count_saved += 1
    print(f"  Saved {success_traj_path} ({count_saved} full trajectories with patch step annotated)")

    exp6_stats = {
        "teachable_rate": teachable_rate,
        "n_trajectories": n_traj,
        "n_teachable": len(discovered),
        "t_star_distribution": t_star_dist,
        "patch_type_counts": patch_type_counts,
    }
    stats_path = os.path.join(output_dir, "EXP6_trajectory_discovery_stats.json")
    with open(stats_path, "w") as f:
        json.dump(exp6_stats, f, indent=2)
    return {"discovered": discovered, "stats": exp6_stats}


def main():
    parser = argparse.ArgumentParser(description="IDT Full Experiment Suite")
    parser.add_argument("--failure_data", type=str, required=True, help="Path to failures JSON/JSONL")
    parser.add_argument("--num_trajectories", type=int, default=None, help="Limit trajectories (default: all)")
    parser.add_argument("--output_dir", type=str, default=None, help=f"Output dir (default: {_output_dir()})")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--diagnosis_model_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulation_budget", type=int, default=999999)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", help="Run only EXP0 smoke test")
    parser.add_argument("--fast", action="store_true", help="Run EXP1-5 on N=100")
    parser.add_argument("--exp6_only", action="store_true", help="Run only EXP6 (trajectory-level discovery)")
    parser.add_argument("--no_save_full_success_trajectories", action="store_true",
                        help="Do not save full successful trajectories in EXP6 (saves disk/time)")
    args = parser.parse_args()

    args.save_full_success_trajectories = not getattr(args, "no_save_full_success_trajectories", False)
    args.output_dir = args.output_dir or _output_dir()
    args.model_path = args.model_path or _default_model_path()
    args.diagnosis_model_path = args.diagnosis_model_path or _default_diagnosis_path()

    failures = load_failures(args.failure_data)
    if args.num_trajectories is not None:
        failures = failures[: args.num_trajectories]
    if args.fast and not args.smoke and not args.exp6_only:
        failures = failures[: min(len(failures), 100)]
        print(f"FAST mode: using first 100 trajectories")
    if args.exp6_only:
        n = args.num_trajectories or 500
        failures = failures[: min(len(failures), n)]
        print(f"EXP6 only: using {len(failures)} trajectories")

    print("Setting up environment and models...")
    env = setup_environment()
    models = setup_model(args.model_path)
    agent = Agent(models)
    diagnosis_model = None
    if os.path.exists(args.diagnosis_model_path):
        diagnosis_model = DiagnosisModelSelector(args.diagnosis_model_path, "Qwen/Qwen2.5-3B-Instruct")
    else:
        print(f"Warning: Diagnosis model not found at {args.diagnosis_model_path}, diagnosis strategy will fail")

    if args.smoke:
        run_exp0_smoke(failures, env, agent, diagnosis_model, args)
        print("\nSmoke test complete.")
        return

    if args.exp6_only:
        run_exp6_trajectory_discovery(failures, env, agent, diagnosis_model, args)
    else:
        run_exp0_smoke(failures, env, agent, diagnosis_model, args)
        run_exp1_main(failures, env, agent, diagnosis_model, args)
        run_exp2_compute_matched(failures, env, agent, diagnosis_model, args)
        run_exp3_random_patch(failures, env, agent, diagnosis_model, args)
        run_exp4_patch_k_scaling(failures, env, agent, diagnosis_model, args)
        run_exp5_step_selectors(failures, env, agent, diagnosis_model, args)
        run_exp6_trajectory_discovery(failures, env, agent, diagnosis_model, args)

    print("\nGenerating plots and report...")
    try:
        from .make_plots import make_all_plots
        make_all_plots(args.output_dir)
    except ImportError:
        from make_plots import make_all_plots
        make_all_plots(args.output_dir)
    except Exception as e:
        print(f"Plots failed: {e}")
    try:
        from .write_report import write_report
        write_report(args.output_dir)
    except ImportError:
        from write_report import write_report
        write_report(args.output_dir)
    except Exception as e:
        print(f"Report failed: {e}")

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
