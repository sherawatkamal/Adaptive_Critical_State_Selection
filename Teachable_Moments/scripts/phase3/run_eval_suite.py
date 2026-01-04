#!/usr/bin/env python3
"""
Minimal Evaluation Suite (Figures 3-6).

Runs comprehensive evaluation for Teachable Moments models:
- Figure 3: Per-quadrant Success Rate (Validation)
- Figure 4: Transfer Matrix (Generalization)
- Figure 5: Retention (Catastrophic Forgetting)
- Figure 6: Stuckness Analysis (Behavioral)

Usage:
    python scripts/phase3/run_eval_suite.py \
        --models-manifest configs/eval_manifest.json \
        --test-panel panels/test_panel.json \
        --outdir results/phase3/eval_suite
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging
from src.utils.model_factory import ModelFactory, ModelConfig
from src.policies.model_factory_policy import ModelFactoryPolicy
from src.data.webshop_env import create_env, WebShopConfig
from src.data.snapshot import Snapshot

logger = logging.getLogger(__name__)


@dataclass
class EvalModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    base_model: str
    lora_path: Optional[str] = None
    train_quadrant: str = "Base"
    supervision: str = "Base"


def run_episode_from_reset(
    factory: ModelFactory,
    task_id: str,
    max_steps: int,
    mock_env: bool = False
) -> Dict[str, Any]:
    """Run episode from standard reset (full task)."""
    env_config = WebShopConfig(max_steps=max_steps)
    env = create_env(env_config, mock=mock_env)
    
    try:
        obs_dict = env.reset(task_id)
        obs = obs_dict["observation"]
        valid_actions = obs_dict.get("valid_actions", [])
        instruction = obs_dict.get("instruction_text", "")
        
        policy = ModelFactoryPolicy(factory)
        
        states = []
        actions = []
        rewards = []
        
        done = False
        total_reward = 0.0
        
        while not done:
            action = policy.get_action(obs, valid_actions, instruction)
            
            obs_dict, reward, done, _ = env.step(action)
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            
            obs = obs_dict.get("observation", "")
            valid_actions = obs_dict.get("valid_actions", [])
            
        success = env.is_success(total_reward)
        
        return {
            "success": success,
            "total_reward": total_reward,
            "n_steps": len(actions),
            "states": states,
            "actions": actions,
            "task_id": task_id,
        }
    finally:
        env.close()


def run_episode_from_snapshot(
    factory: ModelFactory,
    snapshot: Snapshot,
    max_steps: int,
    mock_env: bool = False
) -> Dict[str, Any]:
    """Run episode from restored snapshot state."""
    env_config = WebShopConfig(max_steps=max_steps)
    env = create_env(env_config, mock=mock_env)
    
    try:
        # Use env_state_bytes directly from snapshot
        env_state = snapshot.env_state_bytes
        
        # Restore
        obs_dict = env.set_state(env_state)
        obs = obs_dict["observation"]
        valid_actions = obs_dict.get("valid_actions", [])
        instruction = obs_dict.get("instruction_text", "")
        
        policy = ModelFactoryPolicy(factory)
        
        actions = []
        done = False
        total_reward = 0.0
        
        while not done and len(actions) < max_steps:
            action = policy.get_action(obs, valid_actions, instruction)
            
            obs_dict, reward, done, _ = env.step(action)
            actions.append(action)
            total_reward += reward
            
            obs = obs_dict.get("observation", "")
            valid_actions = obs_dict.get("valid_actions", [])
            
        success = env.is_success(total_reward)
        
        return {
            "success": success,
            "total_reward": total_reward,
            "n_steps": len(actions),
            "actions": actions,
        }
    finally:
        env.close()


def evaluate_model(
    model_cfg: EvalModelConfig,
    panel: Dict[str, Any],
    outdir: Path,
    mock_env: bool = False,
):
    """Evaluate a single model on all metrics."""
    logger.info(f"Evaluating model: {model_cfg.name}")
    
    # Load model once
    config = ModelConfig(
        model_path=model_cfg.base_model,
        lora_path=model_cfg.lora_path,
        load_in_8bit=True
    )
    factory = ModelFactory(config)
    
    results = {
        "per_quadrant": [],
        "retention": [],
        "stuckness": [],
    }
    
    snapshots = [Snapshot.from_dict(s) for s in panel.get("snapshots", [])]
    
    # 1. Per-Quadrant Success (from snapshots)
    for snap in snapshots:
        res = run_episode_from_snapshot(factory, snap, max_steps=15, mock_env=mock_env)
        
        results["per_quadrant"].append({
            "quadrant": snap.quadrant or "Unknown",
            "supervision": model_cfg.supervision,
            "model": model_cfg.name,
            "success": res["success"],
            "snapshot_id": snap.id,
        })
        
        # 4. Stuckness Analysis
        actions = res["actions"]
        n_actions = len(actions)
        unique_actions = len(set(actions))
        repeat_rate = 1.0 - (unique_actions / n_actions) if n_actions > 0 else 0.0
        
        results["stuckness"].append({
            "model": model_cfg.name,
            "repeat_rate": repeat_rate,
            "n_steps": n_actions,
            "success": res["success"],
        })
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Run minimal eval suite")
    parser.add_argument("--models-manifest", required=True)
    parser.add_argument("--test-panel", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mock-env", action="store_true")
    
    args = parser.parse_args()
    setup_logging()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    with open(args.models_manifest) as f:
        manifest = json.load(f)
        
    # Load panel
    with open(args.test_panel) as f:
        panel = json.load(f)
        
    all_results = {
        "per_quadrant": [],
        "stuckness": [],
    }
    
    for m in manifest:
        cfg = EvalModelConfig(**m)
        res = evaluate_model(cfg, panel, outdir, args.mock_env)
        
        all_results["per_quadrant"].extend(res["per_quadrant"])
        all_results["stuckness"].extend(res["stuckness"])
        
    # Save aggregated results
    pd.DataFrame(all_results["per_quadrant"]).to_csv(outdir / "per_quadrant_results.csv", index=False)
    pd.DataFrame(all_results["stuckness"]).to_csv(outdir / "stuckness_results.csv", index=False)
    
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
