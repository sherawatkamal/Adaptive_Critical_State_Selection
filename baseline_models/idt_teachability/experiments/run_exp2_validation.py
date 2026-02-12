#!/usr/bin/env python3
"""Run only EXP2 and print/save diversity validation. Usage: python -m baseline_models.idt_teachability.experiments.run_exp2_validation --failure_data baseline_models/failures.json --num_trajectories 30"""

import os
import sys
import json
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IDT_DIR = os.path.dirname(THIS_DIR)
BASELINE_DIR = os.path.dirname(IDT_DIR)
REPO_ROOT = os.path.dirname(BASELINE_DIR)
for p in [BASELINE_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_data", type=str, required=True)
    parser.add_argument("--num_trajectories", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir or os.path.join(IDT_DIR, "outputs")

    from baseline_models.idt_teachability.idt_core import load_failures, run_idt_experiment
    from eef_detailed_with_diagnosis import setup_environment, setup_model, Agent, DiagnosisModelSelector

    failures = load_failures(args.failure_data)[: args.num_trajectories]
    print("Loaded", len(failures), "failures for EXP2 validation")

    print("Setting up env and models...")
    env = setup_environment()
    models = setup_model(os.path.join(BASELINE_DIR, "ckpts", "web_click", "epoch_9", "model.pth"))
    agent = Agent(models)
    diagnosis_model = None
    diag_path = os.path.join(BASELINE_DIR, "simulation", "Qwen2.5", "qwen25_instruct_v1")
    if os.path.exists(diag_path):
        diagnosis_model = DiagnosisModelSelector(diag_path, "Qwen/Qwen2.5-3B-Instruct")

    records, stats, training = run_idt_experiment(
        failures=failures,
        env=env,
        agent=agent,
        strategy="diagnosis",
        M=3,
        patcher="agent_topk",
        patch_k=5,
        diagnosis_model=diagnosis_model,
        diagnosis_window=1,
        baseline_attempts=3,
        patch_attempts=3,
        paired_seeds=True,
        max_steps=50,
        seed=0,
        simulation_budget=999999,
        verbose=True,
        baseline_only_attempts=15,
    )

    print("\n" + "=" * 60)
    print("EXP2 VALIDATION: Baseline first-action diversity (15 rollouts/step)")
    print("=" * 60)
    cov = stats.get("coverage", {})
    div = cov.get("exp2_baseline_first_action_diversity")
    if div:
        print("  mean unique first actions per step:", round(div.get("mean_unique_first_actions", 0), 2))
        print("  min / max unique first actions:   ", div.get("min_unique_first_actions", 0), "/", div.get("max_unique_first_actions", 0))
        print("  mean attempts used (early stop):  ", round(div.get("mean_attempts_used", 0), 2))
        print("  histogram (unique_count -> n_steps):", div.get("histogram_unique_first_actions", {}))
        print()
        mean_u = div.get("mean_unique_first_actions", 0)
        if mean_u > 1.5:
            print("  VERDICT: Baseline is exploring (multiple distinct first actions per step).")
        elif mean_u > 1.0:
            print("  VERDICT: Mild diversity; some steps have >1 first action.")
        else:
            print("  VERDICT: Low diversity; check for greedy policy or seed issues.")
    else:
        print("  (no diversity block in stats)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_stats = os.path.join(args.output_dir, "EXP2_validation_stats.json")
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nStats saved to", out_stats)
    # Save records without full baseline_first_actions to keep small (optional: full in separate file)
    out_records = os.path.join(args.output_dir, "EXP2_validation_records.json")
    slim = []
    for r in records:
        s = {k: v for k, v in r.items() if k != "baseline_first_actions"}
        s["baseline_n_first_actions"] = len(r.get("baseline_first_actions", []))
        slim.append(s)
    with open(out_records, "w") as f:
        json.dump(slim, f, indent=2)
    print("Records (slim) saved to", out_records)


if __name__ == "__main__":
    main()
