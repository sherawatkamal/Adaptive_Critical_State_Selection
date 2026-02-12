#!/usr/bin/env python3
"""
run_idt_patch.py

IDT-style "patch teachability" evaluation on top of your current EEF pipeline.

What this script does (high level):
  1) For each failed trajectory, select a few candidate steps (baseline/entropy/diagnosis).
  2) For each candidate step t:
       - Baseline: replay to step t and let the agent continue (stochastic rollouts).
       - Patched: replay to step t, FORCE a proposed alternative first action at t, then continue.
  3) Compare baseline vs patched under fixed rollout budgets.

Why this is useful:
  - It operationalizes "teachable moments" as: states where a *minimal intervention*
    (here: a single forced first action) has high leverage.

Important implementation details added in this version:
  - Deterministic seeding per (trajectory, step, attempt) for reproducibility.
  - Optional "paired seeds" (default): baseline and patched share seeds, reducing noise in Δ.
  - Patch validity checks: by default, skip patch actions that are invalid at the state
    (unless --allow_invalid_patch_actions).
  - Logs whether the forced action was actually executed (vs falling back to agent).

Expected repo layout:
  baseline_models/
    idt_teachability/   (this file)
    ckpts/...
    simulation/Qwen2.5/qwen25_instruct_v1

You can run from either repo root or baseline_models/:
  python baseline_models/idt_teachability/run_idt_patch.py --failure_data baseline_models/failures.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# NOTE ON PATHS
#
# Users may run this script from either the repo root or baseline_models/.
# We therefore add BOTH the repo root and baseline_models/ to sys.path.
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # baseline_models/
REPO_ROOT = os.path.abspath(os.path.join(BASELINE_DIR, ".."))  # repo root

sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from idt_teachability.patch_simulator import PatchSimulator
from idt_teachability.patchers import make_patcher, PatchProposal
from idt_teachability.idt_core import run_idt_experiment

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


# --------------------------
# Reproducibility helpers
# --------------------------

def _stable_int(x: Any) -> int:
    """Convert arbitrary object to a stable-ish non-negative int (for seed derivation)."""
    if x is None:
        return 0
    # Most task_ids are ints, but keep it robust.
    try:
        return int(x)
    except Exception:
        return abs(hash(str(x))) % 1_000_000_000


def set_all_seeds(seed: int) -> None:
    """Set python, numpy, and torch seeds for reproducible stochastic policies."""
    seed = int(seed) % 2_000_000_000  # keep in a safe range for torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_base_seed(global_seed: int, task_id: Any, traj_idx: int, step_idx: int) -> int:
    """Derive a deterministic base seed per (trajectory, step)."""
    # Large-ish multipliers to avoid collisions.
    base = (
        _stable_int(global_seed) * 1_000_003
        + _stable_int(task_id) * 9_173
        + int(traj_idx) * 1_003
        + int(step_idx) * 97
    )
    return int(base % 2_000_000_000)


# --------------------------
# Failure loading
# --------------------------

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


# --------------------------
# State selection wrapper
# --------------------------

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


def _is_action_valid_at_state(action: str, valid_actions: List[str], *, allow_any_search: bool) -> bool:
    if not isinstance(action, str) or not action:
        return False
    if action in valid_actions:
        return True
    if allow_any_search and action.startswith("search["):
        return True
    return False


def _forced_was_used(out) -> bool:
    """Detect whether PatchSimulator actually executed the forced action at sim step 0."""
    try:
        if not out.sim_traj:
            return False
        info = out.sim_traj[0].get("action_info", {}) or {}
        return info.get("type") == "forced"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="IDT Patch Teachability Evaluation (on top of EEF/WebShop)")
    parser.add_argument("--failure_data", type=str, required=True, help="Path to pre-collected failure trajectories (json/jsonl)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(THIS_DIR, "outputs"),
        help="Where to write outputs (default: baseline_models/idt_teachability/outputs)",
    )

    # State selection
    parser.add_argument("--strategy", type=str, default="diagnosis",
                        choices=["baseline", "entropy", "stratified_entropy", "diagnosis", "last_n", "search_steps", "random_steps"],
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

    # Rollout budgets (use same attempt count for fair baseline vs patched comparison)
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per rollout after the intervention step")
    parser.add_argument("--baseline_attempts", type=int, default=3, help="How many baseline rollouts per (traj, step)")
    parser.add_argument("--patch_attempts", type=int, default=None,
                        help="Rollouts per patch candidate (default: same as baseline_attempts for fair comparison)")
    parser.add_argument("--simulation_budget", type=int, default=999999, help="Total rollout budget across dataset")

    # Action sampling
    parser.add_argument("--greedy", action="store_true", default=False, help="Use greedy instead of softmax sampling")

    # Early stopping
    parser.add_argument("--no_stop_on_success", action="store_true", default=False,
                        help="Disable early stopping on success (default: stop early).")

    # Backward-compat: older versions accepted --stop_on_success (it was always-on).
    parser.add_argument("--stop_on_success", action="store_true", default=True, help=argparse.SUPPRESS)
    # Patcher
    parser.add_argument("--patcher", type=str, default="agent_topk",
                        choices=["agent_topk", "random", "diagnosis_text"],
                        help="How to propose alternative first actions")
    parser.add_argument("--patch_k", type=int, default=5, help="How many patch actions to evaluate per step")
    parser.add_argument("--allow_any_search", action="store_true", default=False,
                        help="Allow forced search[query] even if not listed in info['valid'].")

    # Validity / logging / reproducibility
    parser.add_argument("--allow_invalid_patch_actions", action="store_true", default=False,
                        help="Do not filter invalid patch proposals (not recommended).")
    parser.add_argument("--unpaired_seeds", action="store_true", default=False,
                        help="Do NOT reuse the same seeds between baseline and patch evaluations (more noise).")
    parser.add_argument("--seed", type=int, default=0, help="Global seed (used to derive per-(traj,step) seeds).")
    parser.add_argument("--debug_step_details", action="store_true", default=False,
                        help="Print original action, patch validity, and whether forced was used.")

    # Book-keeping
    parser.add_argument("--num_trajectories", type=int, default=None, help="Process only first N trajectories")
    parser.add_argument("--save_full_trajectories", action="store_true", default=False,
                        help="Include full replay+rollout trajectories in outputs (large files)")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()
    if args.patch_attempts is None:
        args.patch_attempts = args.baseline_attempts
    os.makedirs(args.output_dir, exist_ok=True)

    action_method = "greedy" if args.greedy else "softmax"
    attempt_mismatch = args.baseline_attempts != args.patch_attempts
    stop_on_success = bool(getattr(args, 'stop_on_success', True)) and (not args.no_stop_on_success)
    paired_seeds = (not args.unpaired_seeds)

    print("=" * 80)
    print("IDT PATCH TEACHABILITY EVALUATION")
    print("=" * 80)
    print(f"Failure data:      {args.failure_data}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Strategy:          {args.strategy} (M={args.M})")
    print(f"Student model:     {args.model_path}")
    print(f"Patcher:           {args.patcher} (k={args.patch_k})")
    print(f"Rollouts:          baseline_attempts={args.baseline_attempts}, patch_attempts={args.patch_attempts}")
    if attempt_mismatch:
        print(f"  WARNING: attempt count mismatch (baseline has more attempts → baseline advantage)")
    else:
        print(f"  Fair comparison: same attempt count per (traj, step)")
    print(f"Action method:     {action_method}")
    print(f"Stop on success:   {stop_on_success}")
    print(f"Paired seeds:      {paired_seeds} (baseline and patched share same seeds per attempt index)")
    print(f"Allow any search:  {args.allow_any_search}")
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

    # Load failures
    failures = load_failures(args.failure_data)
    if args.num_trajectories is not None:
        failures = failures[: args.num_trajectories]

    # Run via idt_core (includes coverage, compute counters, forced_skip_reason)
    all_records, stats, training_samples = run_idt_experiment(
        failures=failures,
        env=env,
        agent=agent,
        strategy=args.strategy,
        M=args.M,
        patcher=args.patcher,
        patch_k=args.patch_k,
        diagnosis_model=diagnosis_model,
        diagnosis_window=args.diagnosis_window,
        baseline_attempts=args.baseline_attempts,
        patch_attempts=args.patch_attempts,
        paired_seeds=paired_seeds,
        max_steps=args.max_steps,
        allow_any_search=args.allow_any_search,
        allow_invalid_patch_actions=args.allow_invalid_patch_actions,
        stop_on_success=stop_on_success,
        greedy=args.greedy,
        seed=args.seed,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
    )

    n = stats.get("replay_ok_steps", 0)
    rescue_rate = stats.get("rescue_rate", 0)
    rescue_count = stats.get("rescue_count", 0)
    n_baseline_failed = stats.get("n_baseline_failed", 0)
    break_rate = stats.get("break_rate", 0)
    break_count = stats.get("break_count", 0)
    n_baseline_succeeded = stats.get("n_baseline_succeeded", 0)
    n_with_patch = stats.get("n_with_patch", 0)
    patch_valid_count = stats.get("patch_valid_count", 0)
    patch_valid_rate = stats.get("patch_valid_rate", 0)
    forced_used_count = stats.get("forced_used_count", 0)
    forced_used_rate = stats.get("forced_used_rate", 0)
    attempt_mismatch = args.baseline_attempts != args.patch_attempts
    stats["attempt_mismatch"] = attempt_mismatch
    stats["action_method"] = action_method

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    print("\n--- Coverage / Denominators ---")
    cov = stats.get("coverage", {})
    for k in ["N_traj_total", "N_traj_processed", "N_step_candidates_total", "N_step_replay_ok", "N_step_with_patch_candidates"]:
        if k in cov:
            print(f"  {k}: {cov[k]}")
    skip = cov.get("skip_reasons", {})
    if skip:
        print("  Skip reasons:", skip)
    forced_skip = cov.get("forced_skip_reason_counts", {})
    if forced_skip:
        print("  Forced skip reason counts:", forced_skip)
    print("\n--- Compute ---")
    comp = stats.get("compute", {})
    for k in ["env_steps_baseline", "env_steps_patched", "model_calls_baseline", "model_calls_patched", "number_of_patch_candidates_evaluated"]:
        if k in comp:
            print(f"  {k}: {comp[k]}")
    print("\n--- Fairness ---")
    print(f"  Attempts: baseline={args.baseline_attempts}, patch={args.patch_attempts}" + (" (mismatch: baseline advantaged)" if attempt_mismatch else " (fair: same)"))
    print(f"  Paired seeds: {paired_seeds}")
    print("\n--- Patch validity & execution ---")
    print(f"  Steps with patch proposals: {n_with_patch}")
    print(f"  patch_valid (patch in valid_actions): {patch_valid_count}/{n_with_patch} ({patch_valid_rate:.1%})")
    print(f"  forced_used (simulator executed forced action): {forced_used_count}/{n_with_patch} ({forced_used_rate:.1%})")
    if n_with_patch and (patch_valid_rate < 0.99 or forced_used_rate < 0.99):
        print("  → Many invalid or fallback patches can make results noisy.")
    print("\n--- Success metrics (patched = best over patch candidates) ---")
    print(f"  replay-ok steps:   {n}")
    print(f"  baseline success: {stats['baseline_success_rate']:.2%}")
    print(f"  patched success: {stats['patched_success_rate']:.2%} (best-of-candidates)")
    print(f"  rescue rate:      {rescue_rate:.2%}  (baseline failed → patched succeeded: {rescue_count}/{n_baseline_failed})")
    print(f"  break rate:      {break_rate:.2%}  (baseline succeeded → patched failed: {break_count}/{n_baseline_succeeded})")
    print(f"  improved (Δ>0):  {stats['improvement_rate']:.2%}")
    print(f"  avg Δ reward:    {stats['avg_improvement']:.2f}")


if __name__ == "__main__":
    main()
