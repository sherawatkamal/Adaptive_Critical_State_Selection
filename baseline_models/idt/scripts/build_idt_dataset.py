#!/usr/bin/env python3
"""
CLI: Build IDT dataset from failed trajectories.
  python -m idt.scripts.build_idt_dataset --failures_path ... --out_path ... --num_trajectories 200 --K 5 --threshold 0.6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idt.dataset import record_from_search_result, save_dataset
from idt.env_adapter import ToyEnvAdapter
from idt.agent_adapter import ToyAgentAdapter
from idt.propose import HeuristicPatchProposer
from idt.search import SearchConfig, search_minimal_patch
from idt.types import load_trajectories, Trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build IDT dataset from failures")
    ap.add_argument("--failures_path", type=str, default=None, help="Path to failures JSON/JSONL")
    ap.add_argument("--out_path", type=str, default="idt_dataset.jsonl", help="Output JSONL path")
    ap.add_argument("--num_trajectories", type=int, default=20, help="Max trajectories to process")
    ap.add_argument("--K", type=int, default=5, help="Attempt budget (e.g. 5)")
    ap.add_argument("--threshold", type=float, default=0.6, help="Recovery threshold")
    ap.add_argument("--toy", action="store_true", help="Use toy env (no WebShop)")
    ap.add_argument("--model_path", type=str, default="baseline_models/ckpts/web_click/epoch_9/model.pth",
                    help="Path to agent model (for real WebShop pipeline; relative to repo root)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load trajectories: from failures_path if set, else generate toy trajectories
    if args.failures_path:
        trajectories = load_trajectories(args.failures_path)
        trajectories = trajectories[: args.num_trajectories]
        logger.info("Loaded %d trajectories from %s", len(trajectories), args.failures_path)
    else:
        from idt.toy_env.toy_env import TOY_FAILURE_ACTIONS
        trajectories = []
        for i in range(args.num_trajectories):
            traj = Trajectory(
                traj_id=str(i),
                task_id=0,
                instruction="Reach state 3.",
                observations=[f"state={j} goal=3" for j in range(4)],
                actions=TOY_FAILURE_ACTIONS.copy(),
                rewards=[0.0] * 3,
                done=True,
                info={},
            )
            trajectories.append(traj)
        logger.info("Generated %d toy trajectories", len(trajectories))

    if not trajectories:
        logger.warning("No trajectories; writing empty dataset")
        save_dataset(args.out_path, [])
        return

    # Choose env/agent: try WebShop unless --toy; fallback to toy on failure
    use_toy_env = args.toy
    if not use_toy_env:
        try:
            from idt.webshop_setup import create_webshop_env_and_agent
            env, agent = create_webshop_env_and_agent(model_path=args.model_path)
            logger.info("Using real WebShop env and agent (model=%s)", args.model_path)
            max_rollout_steps = 50
        except Exception as e:
            import traceback
            logger.warning("WebShop env/agent setup failed (%s); falling back to toy env/agent", e)
            traceback.print_exc()
            use_toy_env = True
    if use_toy_env:
        env = ToyEnvAdapter(seed=args.seed)
        agent = ToyAgentAdapter()
        max_rollout_steps = 10
        if args.failures_path:
            logger.info("Using toy env/agent on real failure data (recovery stats not WebShop-meaningful)")

    proposer = HeuristicPatchProposer(agent=agent)
    config = SearchConfig(
        attempt_budget_schedule=[1, 3, args.K],
        threshold=args.threshold,
        max_rollout_steps=max_rollout_steps,
        base_seed=args.seed,
    )

    records = []
    for traj in trajectories:
        result = search_minimal_patch(env, agent, proposer, traj, config=config)
        rec = record_from_search_result(result)
        if traj.observations and result.best_step is not None:
            rec["instruction"] = traj.instruction
            rec["observation"] = traj.observation_at(result.best_step) if result.best_step < len(traj.observations) else ""
            rec["history_len"] = result.best_step
        records.append(rec)
        logger.info("Traj %s: found_patch=%s teachable=%s", traj.traj_id, result.found_patch, result.teachable_label)

    save_dataset(args.out_path, records)
    logger.info("Wrote %d records to %s", len(records), args.out_path)


if __name__ == "__main__":
    main()
