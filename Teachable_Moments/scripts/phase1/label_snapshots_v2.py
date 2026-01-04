#!/usr/bin/env python3
"""
Phase 1: Full Teachability Labeling (V2)

1. Collects retention tasks (tasks the student already solves).
2. Computes Uncertainty, Leverage, AND CPT (with retention check).
3. Assigns Quadrants.
"""

import argparse
import logging
import json
from pathlib import Path
import random

import sys
import os

# Allow running as a script from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import (
    load_yaml,
    save_json,
    ensure_dir,
    setup_logging,
    set_seed,
    Timer,
)
from src.utils.model_factory import ModelFactory, ModelConfig
from src.data.webshop_env import create_env
from src.policies import ModelFactoryPolicy
from src.data.snapshot import Snapshot
from src.label import (
    compute_all_uncertainty,
    estimate_leverage,
    run_cpt,
    assign_quadrant,
    compute_thresholds,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Label Snapshots V2")
    parser.add_argument("--snapshots", type=str, required=True, help="Input snapshots (e.g. failure_snapshots.json)")
    parser.add_argument("--output", type=str, default="outputs/labeled_snapshots.json")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    
    # Retention
    parser.add_argument("--n-retention-tasks", type=int, default=10, help="Number of success tasks to find for retention Set")
    parser.add_argument("--max-retention-attempts", type=int, default=100)
    
    # Model (if not in config)
    parser.add_argument("--model-checkpoint", type=str, help="Path to student model checkpoint")
    
    return parser.parse_args()

def collect_retention_tasks(env_factory, policy, n_needed, max_attempts):
    """Find tasks that the student succeeds on."""
    retention_ids = []
    
    env = env_factory()
    attempts = 0
    
    logger.info(f"Mining {n_needed} retention tasks...")
    
    while len(retention_ids) < n_needed and attempts < max_attempts:
        task_id = env.sample_task() if hasattr(env, "sample_task") else None
        obs_dict = env.reset(task_id)
        current_task_id = obs_dict.get("task_id")
        
        # Rollout
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 30: # Hardcoded max steps for mining
            action = policy.get_action(
                obs_dict["observation"], 
                obs_dict["valid_actions"], 
                obs_dict.get("instruction_text", "")
            )
            obs_dict, reward, done, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            
        if env.is_success(total_reward):
            retention_ids.append(current_task_id)
            logger.info(f"Found retention task: {current_task_id}")
            
        attempts += 1
        
    if hasattr(env, "close"):
        env.close()
        
    return retention_ids

def main():
    args = parse_args()
    setup_logging()
    set_seed(42)
    
    # Config
    config = load_yaml(args.config) if Path(args.config).exists() else {}
    
    # Load Snapshots
    with open(args.snapshots) as f:
        data = json.load(f)
        snapshots = data if isinstance(data, list) else data.get("snapshots", [])
        
    logger.info(f"Loaded {len(snapshots)} snapshots to label.")
    
    # Setup Env & Policy
    env_name = config.get("environment", {}).get("name", "webshop")
    env_factory = lambda: create_env(config.get("webshop", {})) 
    # Note: Using config.get("webshop") assuming WebShopConfig structure
    
    # Policy Setup
    ckpt = args.model_checkpoint or config.get("model", {}).get("path")
    if not ckpt:
        raise ValueError("Must provide --model-checkpoint or define model.path in config")
        
    logger.info(f"Loading policy from {ckpt}...")
    mf = ModelFactory(ModelConfig.from_checkpoint(ckpt))
    policy = ModelFactoryPolicy(mf)
    
    # 1. Collect Retention Tasks
    retention_ids = collect_retention_tasks(
        env_factory, 
        policy, 
        n_needed=args.n_retention_tasks, 
        max_attempts=args.max_retention_attempts
    )
    logger.info(f"Final Retention Set ({len(retention_ids)}): {retention_ids}")
    
    if len(retention_ids) < args.n_retention_tasks:
        logger.warning("Could not find enough retention tasks! CPT retention metric will be noisy or empty.")

    # 2. Label Snapshots
    labeled_snapshots = []
    
    for i, snap in enumerate(snapshots):
        if i % 10 == 0:
            logger.info(f"Labeling snapshot {i+1}/{len(snapshots)}...")
            
        # Ensure snapshot has env_state_b64 (critical)
        if "env_state_b64" not in snap:
            # Fallback for mining script variance
            if "env_state" in snap: 
                snap["env_state_b64"] = snap["env_state"]
            else:
                logger.warning(f"Snapshot {snap.get('id')} missing env_state_b64, skipping CPT/Leverage.")
                continue

        # A) Uncertainty (if not present)
        # Assuming action_probs available? Or compute via policy?
        # If action_probs missing, we might skip or re-compute if policy supports it.
        # Here we assume mine_failure_snapshots populated action_probs or we skip.
        if "uncertainty" not in snap and "action_probs" in snap:
             snap["uncertainty"] = compute_all_uncertainty(snap["action_probs"])
             
        # B) Leverage
        if "leverage" not in snap:
            snap["leverage"] = estimate_leverage(
                policy=policy,
                expert=None, # Expert not strictly needed if we rely on teacher hints for 'oracle'? No, leverage needs oracle.
                # If we don't have expert object, estimate_leverage might fail or fallback to teacher hint simulator?
                # Actually estimate_leverage usually needs an expert policy to estimate L_upper.
                # For now, we skip if expert not provided, or assume user handles it. 
                # Let's assume we can compute L_local at least.
                env_factory=env_factory,
                snapshot=snap,
            )
            
        # C) CPT
        if "cpt" not in snap:
            # Get teacher hint
            hint = snap.get("teacher_hint")
            if not hint:
                logger.warning(f"Snapshot {snap.get('id')} missing teacher_hint, skipping CPT.")
                continue
                
            # Convert dict to Snapshot object for run_cpt
            snap_obj = Snapshot.from_dict(snap)
            
            cpt_results = run_cpt(
                snapshot=snap_obj,
                env=env_factory(), # run_cpt takes 'env' instance
                student_policy=policy,
                teacher_hint=snap_obj.teacher_hint, # Use the object's hint
                config=None, # Use default CPTConfig
                retention_task_ids=retention_ids,
                retention_env_factory=env_factory
            )
            snap["cpt"] = cpt_results.to_dict()
            
        labeled_snapshots.append(snap)
        
    # 3. Assign Quadrants
    # Compute thresholds dynamically or use fixed? 
    # Let's use compute_thresholds default
    thresholds = compute_thresholds(labeled_snapshots)
    for snap in labeled_snapshots:
        snap["quadrant"] = assign_quadrant(snap, thresholds)
        
    # Save
    ensure_dir(Path(args.output).parent)
    save_json({"snapshots": labeled_snapshots, "thresholds": thresholds}, args.output)
    logger.info(f"Saved {len(labeled_snapshots)} labeled snapshots to {args.output}")

if __name__ == "__main__":
    main()
