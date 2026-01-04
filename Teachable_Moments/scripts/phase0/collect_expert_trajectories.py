#!/usr/bin/env python3
"""
Collect expert (GPT-4o) trajectories for Setup A: Expert-Failure Pipeline.

This script:
1. Runs GPT-4o on WebShop tasks
2. Collects both success and failure trajectories
3. Partitions trajectories for downstream teachability analysis
4. Saves in format compatible with snapshot extraction

Usage:
    python scripts/phase0/collect_expert_trajectories.py \
        --n-tasks 1000 \
        --output-dir results/phase0/expert_trajectories \
        --teacher-model gpt-4o
"""

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.teacher.client import TeacherClient, TeacherConfig
from src.data.webshop_env import create_env, WebShopConfig
from src.utils.common import setup_logging, set_seed, ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class ExpertStep:
    """Single step in expert trajectory."""
    step_idx: int
    observation: str
    valid_actions: list[str]
    action_taken: str
    action_rationale: Optional[str]
    reward: float
    cumulative_reward: float
    done: bool
    
    # For snapshot extraction
    env_state_b64: Optional[str] = None
    action_probs: Optional[dict] = None  # Expert doesn't have probs, but student will


@dataclass
class ExpertTrajectory:
    """Complete expert trajectory on a task."""
    trajectory_id: str
    task_id: str
    task_description: str
    
    steps: list[ExpertStep] = field(default_factory=list)
    
    # Outcome
    success: bool = False
    final_reward: float = 0.0
    n_steps: int = 0
    
    # Metadata
    expert_model: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "steps": [asdict(s) for s in self.steps],
            "success": self.success,
            "final_reward": self.final_reward,
            "n_steps": self.n_steps,
            "expert_model": self.expert_model,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }


# Expert action prompt template
EXPERT_ACTION_PROMPT = """You are an expert web shopping agent. Your task is to find and purchase items that match the given criteria.

## Task
{task_description}

## Current Page
{observation}

## Available Actions
{valid_actions}

## Instructions
1. Analyze the current page and available actions
2. Choose the SINGLE BEST action to make progress toward the goal
3. Explain your reasoning briefly

## Response Format
Respond with JSON:
{{
    "action": "your chosen action exactly as listed above",
    "reasoning": "brief explanation of why this action"
}}

Choose wisely - you want to find the best matching product efficiently."""


class ExpertAgent:
    """GPT-4o expert agent for WebShop."""
    
    def __init__(self, client: TeacherClient, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.action_count = 0
    
    def get_action(
        self,
        task_description: str,
        observation: str,
        valid_actions: list[str],
    ) -> tuple[str, str]:
        """
        Get expert action for current state.
        
        Returns:
            Tuple of (action, reasoning)
        """
        # Format prompt
        actions_str = "\n".join(f"- {a}" for a in valid_actions)
        prompt = EXPERT_ACTION_PROMPT.format(
            task_description=task_description,
            observation=observation[:2000],  # Truncate long observations
            valid_actions=actions_str,
        )
        
        # Get response from teacher
        response = self.client.generate(
            prompt=prompt,
            system_prompt="You are an expert web shopping agent. Always respond with valid JSON.",
        )
        
        self.action_count += 1
        
        # Parse response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                action = data.get("action", "")
                reasoning = data.get("reasoning", "")
            else:
                # Fallback: try to find action in valid_actions
                action = self._extract_action_fallback(response, valid_actions)
                reasoning = response
        except json.JSONDecodeError:
            action = self._extract_action_fallback(response, valid_actions)
            reasoning = response
        
        # Validate action is in valid_actions
        if action not in valid_actions:
            # Try fuzzy matching
            action = self._fuzzy_match_action(action, valid_actions)
        
        return action, reasoning
    
    def _extract_action_fallback(self, response: str, valid_actions: list[str]) -> str:
        """Extract action from response when JSON parsing fails."""
        response_lower = response.lower()
        for action in valid_actions:
            if action.lower() in response_lower:
                return action
        return valid_actions[0] if valid_actions else ""
    
    def _fuzzy_match_action(self, action: str, valid_actions: list[str]) -> str:
        """Fuzzy match action to valid actions."""
        action_lower = action.lower().strip()
        
        # Exact match
        for va in valid_actions:
            if va.lower() == action_lower:
                return va
        
        # Substring match
        for va in valid_actions:
            if action_lower in va.lower() or va.lower() in action_lower:
                return va
        
        # Default to first action
        return valid_actions[0] if valid_actions else ""


def collect_single_trajectory(
    expert: ExpertAgent,
    env,
    task_id: Optional[str],
    trajectory_idx: int,
    save_env_state: bool = True,
) -> ExpertTrajectory:
    """
    Collect a single expert trajectory.
    
    Args:
        expert: Expert agent
        env: WebShop environment
        task_id: Specific task ID or None for random
        trajectory_idx: Index for trajectory ID generation
        save_env_state: Whether to save environment state bytes
    
    Returns:
        ExpertTrajectory object
    """
    start_time = time.time()
    
    # Reset environment
    obs = env.reset(task_id)
    actual_task_id = obs.get("task_id", task_id or f"task_{trajectory_idx}")
    task_description = obs.get("observation", "")[:500]  # First part is usually goal
    
    trajectory = ExpertTrajectory(
        trajectory_id=f"expert_{trajectory_idx:06d}",
        task_id=actual_task_id,
        task_description=task_description,
        expert_model=expert.model,
        timestamp=datetime.now().isoformat(),
    )
    
    done = False
    cumulative_reward = 0.0
    step_idx = 0
    
    while not done and step_idx < env.config.max_steps:
        observation = obs["observation"]
        valid_actions = obs.get("valid_actions", [])
        
        if not valid_actions:
            break
        
        # Save environment state before action
        env_state_b64 = None
        if save_env_state:
            env_state_bytes = env.get_state()
            if env_state_bytes:
                env_state_b64 = base64.b64encode(env_state_bytes).decode("ascii")
        
        # Get expert action
        action, reasoning = expert.get_action(
            task_description=task_description,
            observation=observation,
            valid_actions=valid_actions,
        )
        
        # Take action
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        
        # Record step
        step = ExpertStep(
            step_idx=step_idx,
            observation=observation,
            valid_actions=valid_actions,
            action_taken=action,
            action_rationale=reasoning,
            reward=reward,
            cumulative_reward=cumulative_reward,
            done=done,
            env_state_b64=env_state_b64,
        )
        trajectory.steps.append(step)
        
        step_idx += 1
    
    # Finalize trajectory
    trajectory.n_steps = len(trajectory.steps)
    trajectory.final_reward = cumulative_reward
    trajectory.success = env.is_success(cumulative_reward)
    trajectory.duration_seconds = time.time() - start_time
    
    return trajectory


def collect_expert_trajectories(
    n_tasks: int,
    output_dir: Path,
    teacher_config: TeacherConfig,
    env_config: WebShopConfig,
    task_ids: Optional[list[str]] = None,
    save_env_state: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Collect expert trajectories on multiple tasks.
    
    Args:
        n_tasks: Number of tasks to run
        output_dir: Output directory
        teacher_config: Teacher model configuration
        env_config: Environment configuration
        task_ids: Specific task IDs or None for random
        save_env_state: Whether to save environment state
        progress_callback: Progress callback
    
    Returns:
        Summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    client = TeacherClient(teacher_config)
    expert = ExpertAgent(client, model=teacher_config.model)
    env = create_env(env_config, mock=False)
    
    all_trajectories = []
    successes = []
    failures = []
    partial_failures = []  # Got some reward but didn't complete
    
    task_list = task_ids if task_ids else [None] * n_tasks
    
    for i, task_id in enumerate(task_list):
        try:
            trajectory = collect_single_trajectory(
                expert=expert,
                env=env,
                task_id=task_id,
                trajectory_idx=i,
                save_env_state=save_env_state,
            )
            
            all_trajectories.append(trajectory)
            
            # Categorize
            if trajectory.success:
                successes.append(trajectory)
            elif trajectory.final_reward > 0:
                partial_failures.append(trajectory)
            else:
                failures.append(trajectory)
            
            if progress_callback:
                progress_callback(i + 1, len(task_list))
                
        except Exception as e:
            logger.error(f"Error on task {i}: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # All trajectories
    all_path = output_dir / f"all_trajectories_{timestamp}.json"
    with open(all_path, "w") as f:
        json.dump([t.to_dict() for t in all_trajectories], f, indent=2, default=str)
    
    # Successes
    success_path = output_dir / f"success_trajectories_{timestamp}.json"
    with open(success_path, "w") as f:
        json.dump([t.to_dict() for t in successes], f, indent=2, default=str)
    
    # Failures (complete)
    failure_path = output_dir / f"complete_failure_trajectories_{timestamp}.json"
    with open(failure_path, "w") as f:
        json.dump([t.to_dict() for t in failures], f, indent=2, default=str)
    
    # Partial failures (for EEF-style analysis)
    partial_path = output_dir / f"partial_failure_trajectories_{timestamp}.json"
    with open(partial_path, "w") as f:
        json.dump([t.to_dict() for t in partial_failures], f, indent=2, default=str)
    
    # Summary
    summary = {
        "n_tasks": len(all_trajectories),
        "n_successes": len(successes),
        "n_complete_failures": len(failures),
        "n_partial_failures": len(partial_failures),
        "success_rate": len(successes) / len(all_trajectories) if all_trajectories else 0,
        "avg_steps": sum(t.n_steps for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0,
        "avg_reward": sum(t.final_reward for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0,
        "total_api_calls": expert.action_count,
        "output_files": {
            "all": str(all_path),
            "successes": str(success_path),
            "failures": str(failure_path),
            "partial_failures": str(partial_path),
        },
        "timestamp": timestamp,
    }
    
    summary_path = output_dir / f"collection_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Collection complete: {summary}")
    
    env.close()
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert trajectories for teachable moments research"
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=100,
        help="Number of tasks to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase0/expert_trajectories"),
        help="Output directory",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gpt-4o",
        help="Teacher model name",
    )
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=Path("configs/teacher.yaml"),
        help="Teacher configuration file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="Specific task IDs (optional)",
    )
    parser.add_argument(
        "--no-save-state",
        action="store_true",
        help="Don't save environment state (faster but can't do counterfactuals)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    # Load teacher config
    teacher_config = TeacherConfig.from_yaml(str(args.teacher_config))
    teacher_config.model = args.teacher_model
    
    # Environment config
    env_config = WebShopConfig(max_steps=args.max_steps)
    
    # Progress tracking
    tracker = ProgressTracker(args.n_tasks, "Collecting expert trajectories")
    
    def progress_callback(completed, total):
        tracker.update()
    
    # Collect trajectories
    logger.info(f"Starting collection of {args.n_tasks} expert trajectories")
    logger.info(f"Expert model: {args.teacher_model}")
    logger.info(f"Output directory: {args.output_dir}")
    
    summary = collect_expert_trajectories(
        n_tasks=args.n_tasks,
        output_dir=args.output_dir,
        teacher_config=teacher_config,
        env_config=env_config,
        task_ids=args.task_ids,
        save_env_state=not args.no_save_state,
        progress_callback=progress_callback,
    )
    
    tracker.finish()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Expert Trajectory Collection Summary")
    print("=" * 50)
    print(f"Total tasks: {summary['n_tasks']}")
    print(f"Successes: {summary['n_successes']} ({summary['success_rate']:.1%})")
    print(f"Complete failures: {summary['n_complete_failures']}")
    print(f"Partial failures: {summary['n_partial_failures']}")
    print(f"Average steps: {summary['avg_steps']:.1f}")
    print(f"Average reward: {summary['avg_reward']:.2f}")
    print(f"Total API calls: {summary['total_api_calls']}")
    print(f"\nOutput files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
