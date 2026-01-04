#!/usr/bin/env python3
"""
EEF (Exploring Expert Failures) Comparison Baseline.

This script runs the EEF baseline for comparison with teachable moments.
EEF focuses on learning from expert failure trajectories rather than
student failures.

Reference: "Exploring Expert Failures Improves LLM Agent Tuning"

Key differences from Teachable Moments:
- EEF: Uses expert (GPT-4o) failures, learns from expert mistakes
- Teachable Moments: Uses student failures, identifies teachable states

This comparison tests whether:
1. Student failures provide more teachable moments than expert failures
2. Quadrant-based selection outperforms EEF's trajectory-level selection

Usage:
    python scripts/baselines/run_eef_comparison.py \
        --expert-trajectories results/phase0/expert_trajectories/ \
        --n-samples 500 \
        --output results/baselines/eef_comparison/
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import random

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging, set_seed, ProgressTracker
from src.training.sft_trainer import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


@dataclass
class EEFConfig:
    """Configuration for EEF baseline."""
    
    # Data selection
    n_samples: int = 500
    use_partial_failures: bool = True  # Include trajectories with partial reward
    min_reward_threshold: float = 0.0  # Minimum reward to include
    max_reward_threshold: float = 0.9  # Maximum reward (exclude full successes)
    
    # EEF-specific settings
    extract_critical_steps: bool = True  # Focus on steps before failure
    steps_before_failure: int = 3  # How many steps to extract
    
    # Training
    base_model: str = "meta-llama/Llama-3-8B-Instruct"
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    
    # Output
    output_dir: str = "results/baselines/eef_comparison"
    seed: int = 42


def load_expert_trajectories(
    trajectories_dir: Path,
    config: EEFConfig,
) -> List[Dict[str, Any]]:
    """
    Load expert trajectories and filter for failures.
    """
    trajectories = []
    
    # Look for trajectory files
    trajectory_files = list(trajectories_dir.glob("*.json"))
    
    for traj_file in trajectory_files:
        with open(traj_file) as f:
            data = json.load(f)
        
        # Handle both single trajectory and list of trajectories
        if isinstance(data, list):
            trajectories.extend(data)
        else:
            trajectories.append(data)
    
    logger.info(f"Loaded {len(trajectories)} total trajectories")
    
    # Filter for failures (reward below threshold)
    failures = []
    for traj in trajectories:
        reward = traj.get("final_reward", traj.get("reward", 0))
        
        if config.min_reward_threshold <= reward < config.max_reward_threshold:
            failures.append(traj)
    
    logger.info(f"Found {len(failures)} failure trajectories "
               f"(reward in [{config.min_reward_threshold}, {config.max_reward_threshold}))")
    
    return failures


def extract_eef_training_examples(
    trajectories: List[Dict[str, Any]],
    config: EEFConfig,
) -> List[Dict[str, Any]]:
    """
    Extract training examples from expert failure trajectories.
    
    EEF approach: For each failure trajectory, extract steps leading
    up to the failure and create training examples that show:
    1. The observation at each step
    2. What the expert did (which led to failure)
    3. What should have been done instead (hindsight correction)
    """
    examples = []
    
    for traj in trajectories:
        steps = traj.get("steps", [])
        if not steps:
            continue
        
        task_description = traj.get("task_description", "")
        
        if config.extract_critical_steps:
            # Focus on steps before failure
            n_steps = len(steps)
            start_idx = max(0, n_steps - config.steps_before_failure)
            critical_steps = steps[start_idx:]
        else:
            # Use all steps
            critical_steps = steps
        
        for step in critical_steps:
            observation = step.get("observation", "")
            action_taken = step.get("action_taken", step.get("action", ""))
            valid_actions = step.get("valid_actions", [])
            
            # For EEF, we want to learn from what DIDN'T work
            # The training format shows the bad action and asks model to avoid it
            example = {
                "input": format_eef_input(
                    task_description=task_description,
                    observation=observation,
                    failed_action=action_taken,
                    valid_actions=valid_actions,
                ),
                "output": format_eef_output(valid_actions, action_taken),
                "metadata": {
                    "trajectory_id": traj.get("trajectory_id", ""),
                    "step_idx": step.get("step_idx", 0),
                    "failed_action": action_taken,
                    "trajectory_reward": traj.get("final_reward", 0),
                },
            }
            examples.append(example)
    
    return examples


def format_eef_input(
    task_description: str,
    observation: str,
    failed_action: str,
    valid_actions: List[str],
) -> str:
    """
    Format input for EEF-style training.
    
    EEF teaches the model to avoid failed actions by showing
    what didn't work and asking for alternatives.
    """
    actions_str = ", ".join(valid_actions[:15])
    
    return f"""Task: {task_description}

Current observation:
{observation[:1500]}

An expert tried action "{failed_action}" but this led to failure.

Valid actions: {actions_str}

Choose a DIFFERENT action that would be more likely to succeed:"""


def format_eef_output(valid_actions: List[str], failed_action: str) -> str:
    """
    Format output for EEF training.
    
    For EEF, we want to train the model to choose any action
    OTHER than the failed one. In practice, we'd want hindsight
    correction to know the "correct" action, but as a baseline
    we can use heuristics or random alternative selection.
    """
    # Filter out the failed action
    alternatives = [a for a in valid_actions if a != failed_action]
    
    if alternatives:
        # Return first alternative (in production, would use hindsight correction)
        return alternatives[0]
    else:
        return failed_action  # Fallback if no alternatives


def run_eef_experiment(
    trajectories_dir: Path,
    config: EEFConfig,
) -> Dict[str, Any]:
    """
    Run the EEF baseline experiment.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load expert failure trajectories
    trajectories = load_expert_trajectories(trajectories_dir, config)
    
    if not trajectories:
        logger.error("No failure trajectories found!")
        return {"error": "No failure trajectories"}
    
    # Extract training examples
    all_examples = extract_eef_training_examples(trajectories, config)
    
    logger.info(f"Extracted {len(all_examples)} training examples from {len(trajectories)} trajectories")
    
    # Sample if we have too many
    if len(all_examples) > config.n_samples:
        random.seed(config.seed)
        examples = random.sample(all_examples, config.n_samples)
    else:
        examples = all_examples
    
    logger.info(f"Using {len(examples)} training examples")
    
    # Analyze trajectory reward distribution
    rewards = [ex["metadata"]["trajectory_reward"] for ex in examples]
    reward_stats = {
        "mean": np.mean(rewards),
        "std": np.std(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
    }
    logger.info(f"Reward stats: {reward_stats}")
    
    # Save training data
    training_data_path = output_dir / "training_data.json"
    with open(training_data_path, "w") as f:
        json.dump(examples, f, indent=2)
    
    # Prepare for SFT trainer format
    sft_examples = [
        {"input": ex["input"], "output": ex["output"]}
        for ex in examples
    ]
    
    # Train model
    sft_config = SFTConfig(
        base_model=config.base_model,
        output_dir=str(output_dir / "checkpoints"),
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
    )
    
    trainer = SFTTrainer(sft_config)
    
    logger.info("Starting EEF training...")
    training_result = trainer.train(sft_examples)
    
    # Save results
    results = {
        "config": asdict(config),
        "n_trajectories": len(trajectories),
        "n_examples": len(examples),
        "reward_stats": reward_stats,
        "training_result": training_result,
    }
    
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"EEF experiment complete. Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run EEF comparison baseline"
    )
    parser.add_argument(
        "--expert-trajectories",
        type=Path,
        required=True,
        help="Directory containing expert trajectory files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of training examples",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Minimum reward threshold for failures",
    )
    parser.add_argument(
        "--max-reward",
        type=float,
        default=0.9,
        help="Maximum reward threshold (exclude successes)",
    )
    parser.add_argument(
        "--steps-before-failure",
        type=int,
        default=3,
        help="Number of steps before failure to extract",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/eef_comparison"),
        help="Output directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Base model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    config = EEFConfig(
        n_samples=args.n_samples,
        min_reward_threshold=args.min_reward,
        max_reward_threshold=args.max_reward,
        steps_before_failure=args.steps_before_failure,
        output_dir=str(args.output),
        base_model=args.base_model,
        seed=args.seed,
    )
    
    results = run_eef_experiment(args.expert_trajectories, config)
    
    print("\n" + "=" * 50)
    print("EEF Comparison Baseline Results")
    print("=" * 50)
    print(f"Trajectories used: {results.get('n_trajectories', 0)}")
    print(f"Training examples: {results.get('n_examples', 0)}")
    if "reward_stats" in results:
        print(f"Reward range: {results['reward_stats']['min']:.2f} - {results['reward_stats']['max']:.2f}")


if __name__ == "__main__":
    main()
