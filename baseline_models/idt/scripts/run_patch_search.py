#!/usr/bin/env python3
"""
CLI: Run minimal patch search on a single trajectory or failures file.
  python -m idt.scripts.run_patch_search --failures_path ... [--toy] [--num 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idt.env_adapter import ToyEnvAdapter
from idt.agent_adapter import ToyAgentAdapter
from idt.propose import HeuristicPatchProposer
from idt.search import SearchConfig, search_minimal_patch
from idt.types import load_trajectories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures_path", type=str, default=None)
    ap.add_argument("--toy", action="store_true", help="Use toy env")
    ap.add_argument("--num", type=int, default=1, help="Number of trajectories to run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.toy or not args.failures_path:
        from idt.toy_env.toy_env import TOY_FAILURE_ACTIONS
        from idt.types import Trajectory
        trajs = [
            Trajectory(
                traj_id="0",
                task_id=0,
                instruction="Reach state 3.",
                observations=[f"state={j} goal=3" for j in range(4)],
                actions=TOY_FAILURE_ACTIONS.copy(),
                rewards=[0.0] * 3,
                done=True,
                info={},
            )
        ]
    else:
        trajs = load_trajectories(args.failures_path)[: args.num]

    env = ToyEnvAdapter(seed=args.seed)
    agent = ToyAgentAdapter()
    proposer = HeuristicPatchProposer(agent=agent)
    config = SearchConfig(attempt_budget_schedule=[1, 3, 5], threshold=0.6, max_rollout_steps=10 if args.toy else 50, base_seed=args.seed)

    for traj in trajs:
        result = search_minimal_patch(env, agent, proposer, traj, config=config)
        print(json.dumps({
            "traj_id": result.traj_id,
            "found_patch": result.found_patch,
            "best_step": result.best_step,
            "patch_type": result.patch_type,
            "R1": result.R1,
            "R3": result.R3,
            "R5": result.R5,
            "teachable_label": result.teachable_label,
        }, indent=2))


if __name__ == "__main__":
    main()
