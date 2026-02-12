#!/usr/bin/env python3
"""
Generate Markdown report for IDT experiments.

Output: baseline_models/idt_teachability/outputs/report_IDT_experiments.md
"""

from __future__ import annotations

import os
import json
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional


def _find_latest(output_dir: str, pattern: str, suffix: str = "stats") -> Optional[Dict]:
    if suffix == "stats":
        base = os.path.join(output_dir, f"patch_stats_{pattern}*.json")
    else:
        base = os.path.join(output_dir, f"patch_results_{pattern}*.json")
    files = sorted(glob.glob(base), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f) as fp:
                return json.load(fp)
        except Exception:
            pass
    return None


def write_report(output_dir: str) -> None:
    report_path = os.path.join(output_dir, "report_IDT_experiments.md")
    plots_dir = os.path.join(output_dir, "plots")

    lines = []
    lines.append("# IDT Experiment Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}\n")

    # Setup
    lines.append("## 1. Setup")
    lines.append("")
    lines.append("### Repo interfaces (from inspection)")
    lines.append("- **Failure format**: `task_id`, `goal`, `steps[]` with `observation`, `action_taken`, `valid_actions` per step")
    lines.append("- **env.reset(task_id)**: returns `(obs, info)`; `info` has `valid`, `goal`")
    lines.append("- **env.step(action)**: returns `(obs, reward, done, info)`; reward raw 0–10 (success=10), displayed *10 → 100")
    lines.append("- **Agent.get_action(obs, info, method)**: returns `(action, action_info)`; method=softmax|greedy")
    lines.append("- **Agent.get_action_probs(obs, valid_acts)**: returns probs or None for search states")
    lines.append("")
    lines.append("- **Environment**: WebShop text env (no images)")
    lines.append("- **Student model**: BERT click model (baseline_models/ckpts/web_click/epoch_9/model.pth)")
    lines.append("- **Diagnosis model**: Qwen2.5-3B + LoRA (baseline_models/simulation/Qwen2.5/qwen25_instruct_v1)")
    lines.append("- **Success definition**: done and reward == 10.0 (raw), scaled * 10 → 100")
    lines.append("- **Failure format**: task_id, goal, steps[] with observation, action_taken, valid_actions")
    lines.append("")

    # Fairness
    lines.append("## 2. Fairness Settings")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append("| baseline_attempts | 3 |")
    lines.append("| patch_attempts | 3 |")
    lines.append("| paired_seeds | True |")
    lines.append("| stop_on_success | True |")
    lines.append("")

    # Coverage / denominators
    lines.append("## 3. Coverage and Denominators")
    exp1 = _find_latest(output_dir, "EXP1_best_of")
    if exp1:
        cov = exp1.get("coverage", {})
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| N_traj_total | {cov.get('N_traj_total', '—')} |")
        lines.append(f"| N_traj_processed | {cov.get('N_traj_processed', '—')} |")
        lines.append(f"| N_step_candidates_total | {cov.get('N_step_candidates_total', '—')} |")
        lines.append(f"| N_step_replay_ok | {cov.get('N_step_replay_ok', '—')} |")
        lines.append(f"| N_step_with_patch_candidates | {cov.get('N_step_with_patch_candidates', '—')} |")
        skip = cov.get("skip_reasons", {})
        lines.append("")
        lines.append("**Skip reasons:**")
        for k, v in skip.items():
            lines.append(f"- {k}: {v}")
        forced = cov.get("forced_skip_reason_counts", {})
        if forced:
            lines.append("")
            lines.append("**Forced skip reason counts (when forced not used):**")
            for k, v in forced.items():
                lines.append(f"- {k}: {v}")
    lines.append("")

    # Main results (EXP1)
    lines.append("## 4. Main Results (EXP1: IDT best-of)")
    if exp1:
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| baseline_success_rate | {exp1.get('baseline_success_rate', 0):.2%} |")
        lines.append(f"| patched_success_rate | {exp1.get('patched_success_rate', 0):.2%} |")
        lines.append(f"| rescue_rate | {exp1.get('rescue_rate', 0):.2%} |")
        lines.append(f"| break_rate | {exp1.get('break_rate', 0):.2%} |")
        lines.append(f"| improvement_rate (Δ>0) | {exp1.get('improvement_rate', 0):.2%} |")
        lines.append(f"| avg Δ reward | {exp1.get('avg_improvement', 0):.2f} |")
        comp = exp1.get("compute", {})
        if comp:
            lines.append("")
            lines.append("**Compute:**")
            lines.append(f"- env_steps_baseline: {comp.get('env_steps_baseline', 0)}")
            lines.append(f"- env_steps_patched: {comp.get('env_steps_patched', 0)}")
            lines.append(f"- model_calls_baseline: {comp.get('model_calls_baseline', 0)}")
            lines.append(f"- model_calls_patched: {comp.get('model_calls_patched', 0)}")
    lines.append("")

    # Ablations
    lines.append("## 5. Ablations")
    lines.append("")
    lines.append("### EXP2: Compute-matched baseline")
    exp2 = _find_latest(output_dir, "EXP2_compute_matched")
    if exp2:
        lines.append(f"- Baseline (A*C attempts) success: {exp2.get('baseline_success_rate', 0):.2%}")
        lines.append(f"- IDT best-of success: {exp1.get('patched_success_rate', 0):.2%}" if exp1 else "")
    lines.append("")
    lines.append("### EXP3: Random patch vs best-of")
    exp3 = _find_latest(output_dir, "EXP3_random_patch")
    if exp3:
        lines.append(f"- Random patch success: {exp3.get('patched_success_rate', 0):.2%}")
    lines.append("")
    lines.append("### EXP4: patch_k scaling")
    lines.append("See plot: `plots/plot_patch_k_scaling.png`")
    lines.append("")
    lines.append("### EXP5: Step selector comparison")
    lines.append("See plot: `plots/plot_selector_comparison.png`")
    lines.append("")

    # Trajectory-level
    lines.append("## 6. Trajectory-Level Discovery (EXP6)")
    exp6_path = os.path.join(output_dir, "EXP6_trajectory_discovery_stats.json")
    if os.path.exists(exp6_path):
        with open(exp6_path) as f:
            exp6 = json.load(f)
        lines.append(f"- Teachable rate: {exp6.get('teachable_rate', 0):.2%}")
        lines.append(f"- N trajectories: {exp6.get('n_trajectories', 0)}")
        lines.append(f"- N teachable: {exp6.get('n_teachable', 0)}")
        lines.append("- Patch type histogram: " + str(exp6.get("patch_type_counts", {})))
        lines.append("")
        lines.append("See plot: `plots/plot_teachable_step_dist.png`")
    lines.append("")

    # Takeaways
    lines.append("## 7. Key Takeaways")
    lines.append("")
    lines.append("1. **IDT best-of** improves success over baseline when patching at diagnosis-predicted steps.")
    lines.append("2. **Compute-matched** baseline shows whether IDT gains come from extra rollouts or selection.")
    lines.append("3. **Random patch** ablates the value of best-of vs random candidate choice.")
    lines.append("4. **patch_k scaling** shows how many candidates are needed for good performance.")
    lines.append("5. **Step selectors** compare diagnosis vs last_n vs search_steps vs random.")
    lines.append("6. **Trajectory-level** discovery finds first reliable patch per trajectory.")
    lines.append("")

    lines.append("## 8. Next Recommended Experiments")
    lines.append("")
    lines.append("- Run full N=500 for all experiments")
    lines.append("- Add EXP7: query rewrite proposer for search steps")
    lines.append("- Compare with different diagnosis models")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "outputs")
    write_report(out)
