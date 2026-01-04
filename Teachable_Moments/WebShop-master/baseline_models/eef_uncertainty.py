#!/usr/bin/env python3
"""
EEF Pipeline - With Decisional Uncertainty Calculation

Changes from original:
1. Runs N rollouts (default 10) from each selected state to compute decisional uncertainty
2. Calculates U(s) = p̂ × (1 - p̂) where p̂ = successes/N
3. Saves detailed state analysis JSON with all rollout results
4. Includes per-state uncertainty metrics and rollout details

Default: Uses softmax exploration (use --greedy to disable)
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# ENVIRONMENT AND MODEL SETUP
# ============================================================================

def setup_environment(split='test'):
    """Setup WebShop environment - NO IMAGES"""
    print("Setting up WebShop environment...")
    
    # Temporarily clear sys.argv to prevent train_rl from parsing our args
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    
    env_args = webenv_args()[0]
    
    # Restore original argv
    sys.argv = original_argv
    
    env_args.get_image = 0
    env_args.human_goals = 0
    env_args.extra_search_path = ""
    
    env = WebEnv(env_args, split=split)
    print("✓ Environment loaded (no images)")
    return env


def setup_model(model_path="./ckpts/web_click/epoch_9/model.pth"):
    """Setup the IL model - NO BART, NO IMAGES"""
    from train_choice_il import tokenizer, data_collator, process, process_goal
    from models.bert import BertModelForWebshop, BertConfigForWebshop
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}...")
    config = BertConfigForWebshop(image=False)
    model = BertModelForWebshop(config)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device} (no image features)")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'data_collator': data_collator,
        'process': process,
        'process_goal': process_goal,
        'device': device,
    }


class Agent:
    """Agent wrapper - NO BART"""
    
    def __init__(self, models_dict):
        self.model = models_dict['model']
        self.tokenizer = models_dict['tokenizer']
        self.data_collator = models_dict['data_collator']
        self.process = models_dict['process']
        self.process_goal = models_dict['process_goal']
        self.device = models_dict['device']
    
    def get_action(self, obs: str, info: dict, method='softmax') -> Tuple[str, dict]:
        """Get action from the model. Default is softmax for exploration."""
        valid_acts = info.get('valid', [])
        
        if not valid_acts:
            return 'click[back to search]', {'type': 'fallback'}
        
        # Handle search page - NO BART
        if valid_acts[0].startswith('search['):
            action = valid_acts[-1] if valid_acts else 'search[query]'
            return action, {'type': 'search', 'selected': 'valid_acts[-1]'}
        
        # Encode state and actions
        state_encodings = self.tokenizer(
            self.process(obs), max_length=512, truncation=True, padding='max_length'
        )
        action_encodings = self.tokenizer(
            list(map(self.process, valid_acts)), max_length=512, truncation=True, padding='max_length'
        )
        
        batch = {
            'state_input_ids': state_encodings['input_ids'],
            'state_attention_mask': state_encodings['attention_mask'],
            'action_input_ids': action_encodings['input_ids'],
            'action_attention_mask': action_encodings['attention_mask'],
            'sizes': len(valid_acts),
            'images': [0.0] * 512,
            'labels': 0
        }
        batch = self.data_collator([batch])
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)
            
            if method == 'greedy':
                idx = logits.argmax(0).item()
            else:  # softmax (default)
                idx = torch.multinomial(probs, 1)[0].item()
        
        action = valid_acts[idx] if idx < len(valid_acts) else valid_acts[0]
        return action, {
            'type': 'choice',
            'chosen_idx': idx,
            'num_valid': len(valid_acts),
            'confidence': probs[idx].item(),
            'action_probs': probs.cpu().tolist(),
        }


# ============================================================================
# EEF SIMULATOR
# ============================================================================

class EEFSimulator:
    """EEF Simulator"""
    
    def __init__(self, env, agent, max_steps=50, debug=False):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.debug = debug
        self.stats = {
            'total_simulations': 0,
            'successful_replays': 0,
            'replay_failures': 0,
            'recoveries': 0,
        }
    
    def simulate_from_state(self, task_id: int, target_step: int, 
                           trajectory: Dict, method='softmax') -> Tuple[bool, float, List[Dict]]:
        """Simulate from a specific step in the trajectory."""
        self.stats['total_simulations'] += 1
        
        steps = trajectory.get('steps', [])
        if target_step >= len(steps):
            return False, 0.0, []
        
        obs, info = self.env.reset(task_id)
        simulation_traj = []
        
        # PHASE 1: Replay actions to reach target state
        for step_idx in range(target_step):
            if step_idx >= len(steps):
                break
            
            action = steps[step_idx].get('action_taken', '')
            if not action:
                continue
            
            simulation_traj.append({
                'step': step_idx,
                'observation': obs,
                'action_taken': action,
                'is_replay': True,
                'valid_actions': info.get('valid', [])
            })
            
            obs, reward, done, info = self.env.step(action)
            
            if done:
                return reward == 10.0, reward * 10, simulation_traj
        
        self.stats['successful_replays'] += 1
        
        # PHASE 2: Agent-driven simulation from target state
        for sim_step in range(self.max_steps):
            valid_acts = info.get('valid', [])
            if not valid_acts:
                break
            
            action, action_info = self.agent.get_action(obs, info, method=method)
            
            simulation_traj.append({
                'step': target_step + sim_step,
                'observation': obs,
                'action_taken': action,
                'action_info': action_info,
                'is_replay': False,
                'valid_actions': valid_acts,
            })
            
            obs, reward, done, info = self.env.step(action)
            
            simulation_traj[-1]['reward'] = reward * 10
            simulation_traj[-1]['done'] = done
            
            if done:
                success = reward == 10.0
                if success:
                    self.stats['recoveries'] += 1
                return success, reward * 10, simulation_traj
        
        return False, 0.0, simulation_traj


# ============================================================================
# STATE SELECTION (ACSS)
# ============================================================================

def select_critical_states_baseline(trajectory: Dict, M: int = 5) -> List[int]:
    """Baseline: Equal-interval skip-length selection"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return []
    l = max(1, T // (M + 1))
    return [m * l for m in range(1, M + 1) if m * l < T]


def select_critical_states_entropy(trajectory: Dict, M: int = 5) -> List[int]:
    """ACSS: Entropy-approximation based selection"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return []
    
    scores = []
    for i, step in enumerate(steps[:-1]):
        n_actions = len(step.get('valid_actions', []))
        entropy_score = np.log(n_actions + 1) / np.log(100)
        position = i / max(T - 1, 1)
        position_score = np.exp(-0.5 * ((position - 0.4) / 0.3) ** 2)
        score = 0.6 * entropy_score + 0.4 * position_score
        scores.append((i, score, n_actions))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return sorted([idx for idx, _, _ in scores[:M]])


def get_entropy_score(trajectory: Dict, state_idx: int) -> Tuple[float, float, float]:
    """Get the entropy and position scores for a specific state"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if state_idx >= len(steps):
        return 0.0, 0.0, 0.0
    
    n_actions = len(steps[state_idx].get('valid_actions', []))
    entropy_score = np.log(n_actions + 1) / np.log(100)
    position = state_idx / max(T - 1, 1)
    position_score = np.exp(-0.5 * ((position - 0.4) / 0.3) ** 2)
    combined_score = 0.6 * entropy_score + 0.4 * position_score
    
    return entropy_score, position_score, combined_score


# ============================================================================
# DECISIONAL UNCERTAINTY CALCULATION
# ============================================================================

def compute_decisional_uncertainty(successes: int, N: int) -> Dict:
    """
    Compute decisional uncertainty using variance formula.
    U(s) = p̂ × (1 - p̂), peaks at 0.25 when p̂ = 0.5
    
    Args:
        successes: Number of successful rollouts
        N: Total number of rollouts
    
    Returns:
        Dictionary with p_hat and uncertainty values
    """
    p_hat = successes / N if N > 0 else 0
    uncertainty = p_hat * (1 - p_hat)
    return {
        'p_hat': p_hat,
        'uncertainty': uncertainty,
        'successes': successes,
        'total_rollouts': N,
    }


# ============================================================================
# EEF PIPELINE WITH DECISIONAL UNCERTAINTY
# ============================================================================

def run_eef_detailed(env, agent, failures: List[Dict], 
                     M: int = 5, N: int = 10, strategy: str = 'baseline',
                     simulation_budget: int = 10000, verbose: bool = True,
                     greedy: bool = False):
    """
    Run EEF pipeline with detailed segment saving and decisional uncertainty.
    
    For each selected state, runs N rollouts to compute:
    - p̂ = successes / N
    - U(s) = p̂ × (1 - p̂)
    
    Args:
        M: Number of states to select per trajectory
        N: Number of rollouts per state for uncertainty estimation (default 10)
    
    Returns:
        - full_success_segments: States that achieved reward=100
        - improvement_segments: States that improved but didn't reach 100
        - all_beneficial_states: Detailed state info for entropy analysis
        - detailed_analysis: Per-task breakdown with all rollout info
        - training_samples: All training data
        - stats: Pipeline statistics
    """
    simulator = EEFSimulator(env, agent, debug=verbose)
    select_states = select_critical_states_entropy if strategy == 'entropy' else select_critical_states_baseline
    
    # Default is softmax, use greedy only if explicitly set
    action_method = 'greedy' if greedy else 'softmax'
    
    print(f"\n{'='*70}")
    print(f"RUNNING EEF WITH DECISIONAL UNCERTAINTY")
    print(f"{'='*70}")
    print(f"  Strategy: {strategy}")
    print(f"  M (states per trajectory): {M}")
    print(f"  N (rollouts per state): {N}")
    print(f"  Failures: {len(failures)}")
    print(f"  Simulation budget: {simulation_budget}")
    print(f"  Action method: {action_method}")
    print(f"{'='*70}\n")
    
    # Separate collections
    full_success_segments = []      # reward = 100
    improvement_segments = []        # reward > original but < 100
    all_beneficial_states = []       # Detailed state info for ALL beneficial states
    all_simulated_states = []        # ALL states that were simulated (for analysis)
    
    # Detailed per-task analysis
    detailed_task_analysis = []
    
    simulations_run = 0
    total_states_selected = 0
    
    for traj_idx, trajectory in enumerate(failures):
        task_id = trajectory['task_id']
        original_reward = trajectory.get('reward', 0)
        goal = trajectory.get('goal', '')
        traj_length = len(trajectory.get('steps', []))
        critical_states = select_states(trajectory, M)
        
        if not critical_states:
            continue
        
        total_states_selected += len(critical_states)
        
        # Initialize task analysis
        task_analysis = {
            'task_id': task_id,
            'goal': goal[:500],
            'trajectory_length': traj_length,
            'original_reward': original_reward,
            'strategy': strategy,
            'selected_states': critical_states,
            'num_states_selected': len(critical_states),
            'state_details': [],
            'task_summary': {
                'total_rollouts': 0,
                'total_successes': 0,
                'total_improvements': 0,
                'any_full_success': False,
                'any_improvement': False,
            }
        }
        
        if verbose:
            print(f"\n  Task {task_id} ({traj_idx+1}/{len(failures)}):")
            print(f"    Trajectory length: {traj_length} steps")
            print(f"    Original reward: {original_reward:.0f}")
            print(f"    Selected states: {critical_states}")
        
        for state_idx in critical_states:
            if simulations_run >= simulation_budget:
                break
            
            # Get the state info from trajectory
            steps = trajectory.get('steps', [])
            state_obs = steps[state_idx].get('observation', '') if state_idx < len(steps) else ''
            state_valid_actions = steps[state_idx].get('valid_actions', []) if state_idx < len(steps) else []
            
            # Get entropy scores for this state
            entropy_score, position_score, combined_score = get_entropy_score(trajectory, state_idx)
            
            # Run N rollouts for this state
            rollout_results = []
            successes = 0
            improvements = 0
            best_reward = -1
            best_traj = None
            best_success = False
            all_rewards = []
            
            for rollout_idx in range(N):
                if simulations_run >= simulation_budget:
                    break
                
                success, reward, sim_traj = simulator.simulate_from_state(
                    task_id, state_idx, trajectory, method=action_method
                )
                simulations_run += 1
                
                # Record this rollout
                rollout_info = {
                    'rollout_idx': rollout_idx,
                    'success': success,
                    'reward': reward,
                    'is_improvement': reward > original_reward,
                    'num_steps': len([s for s in sim_traj if not s.get('is_replay', False)]),
                }
                rollout_results.append(rollout_info)
                all_rewards.append(reward)
                
                if success:
                    successes += 1
                if reward > original_reward:
                    improvements += 1
                
                # Track best result
                if reward > best_reward:
                    best_reward = reward
                    best_traj = sim_traj
                    best_success = success
            
            # Compute decisional uncertainty
            actual_rollouts = len(rollout_results)
            uncertainty_info = compute_decisional_uncertainty(successes, actual_rollouts)
            
            # State-level detail
            state_detail = {
                'state_idx': state_idx,
                'num_valid_actions': len(state_valid_actions),
                'valid_actions': state_valid_actions[:20],  # Truncate for storage
                
                # Selection scores (for entropy strategy)
                'entropy_score': entropy_score,
                'position_score': position_score,
                'combined_selection_score': combined_score,
                
                # Rollout statistics
                'num_rollouts': actual_rollouts,
                'num_successes': successes,
                'num_improvements': improvements,
                'success_rate': successes / actual_rollouts if actual_rollouts > 0 else 0,
                'improvement_rate': improvements / actual_rollouts if actual_rollouts > 0 else 0,
                
                # Reward statistics
                'rewards': all_rewards,
                'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0,
                'std_reward': float(np.std(all_rewards)) if all_rewards else 0,
                'min_reward': float(min(all_rewards)) if all_rewards else 0,
                'max_reward': float(max(all_rewards)) if all_rewards else 0,
                'best_reward': best_reward,
                
                # Decisional uncertainty
                'p_hat': uncertainty_info['p_hat'],
                'decisional_uncertainty': uncertainty_info['uncertainty'],
                
                # Best result info
                'best_is_full_success': best_success and best_reward >= 100,
                'best_is_improvement': best_reward > original_reward + 10,
                
                # All rollout details
                'rollouts': rollout_results,
            }
            task_analysis['state_details'].append(state_detail)
            
            # Update task summary
            task_analysis['task_summary']['total_rollouts'] += actual_rollouts
            task_analysis['task_summary']['total_successes'] += successes
            task_analysis['task_summary']['total_improvements'] += improvements
            
            if verbose:
                print(f"    State {state_idx}: {successes}/{actual_rollouts} success, "
                      f"p̂={uncertainty_info['p_hat']:.2f}, U={uncertainty_info['uncertainty']:.3f}, "
                      f"best={best_reward:.0f}, valid_actions={len(state_valid_actions)}")
            
            # Record ALL simulated states for analysis
            simulated_state_info = {
                'task_id': task_id,
                'recovery_step': state_idx,
                'trajectory_length': traj_length,
                'original_reward': original_reward,
                'final_reward': best_reward,
                'is_success': best_success,
                'is_improvement': best_reward > original_reward + 10,
                'state': state_obs[:2000],
                'valid_actions': state_valid_actions,
                'num_valid_actions': len(state_valid_actions),
                'num_rollouts': actual_rollouts,
                'num_successes': successes,
                'p_hat': uncertainty_info['p_hat'],
                'decisional_uncertainty': uncertainty_info['uncertainty'],
                'strategy': strategy,
            }
            all_simulated_states.append(simulated_state_info)
            
            # Check for improvement
            is_full_success = best_success and best_reward >= 100
            is_improvement = best_reward > original_reward + 10
            
            if is_full_success:
                task_analysis['task_summary']['any_full_success'] = True
            if is_improvement:
                task_analysis['task_summary']['any_improvement'] = True
            
            if is_full_success or is_improvement:
                status = "SUCCESS" if is_full_success else "IMPROVED"
                if verbose:
                    print(f"      → ✓ {status}! {original_reward:.0f} → {best_reward:.0f}")
                
                # Extract beneficial actions
                new_actions = [s for s in best_traj if not s.get('is_replay', False)]
                
                segment_data = {
                    'task_id': task_id,
                    'goal': goal,
                    'recovery_step': state_idx,
                    'trajectory_length': traj_length,
                    'original_reward': original_reward,
                    'final_reward': best_reward,
                    'is_full_success': is_full_success,
                    'num_recovery_steps': len(new_actions),
                    'state_observation': state_obs[:2000],
                    'state_valid_actions': state_valid_actions,
                    'p_hat': uncertainty_info['p_hat'],
                    'decisional_uncertainty': uncertainty_info['uncertainty'],
                    'steps': new_actions,
                }
                
                if is_full_success:
                    full_success_segments.append(segment_data)
                else:
                    improvement_segments.append(segment_data)
                
                # Detailed state info for entropy analysis
                all_beneficial_states.append({
                    'task_id': task_id,
                    'recovery_step': state_idx,
                    'trajectory_length': traj_length,
                    'original_reward': original_reward,
                    'final_reward': best_reward,
                    'is_full_success': is_full_success,
                    'state': state_obs[:2000],
                    'valid_actions': state_valid_actions,
                    'num_valid_actions': len(state_valid_actions),
                    'goal': goal[:500],
                    'strategy': strategy,
                    'p_hat': uncertainty_info['p_hat'],
                    'decisional_uncertainty': uncertainty_info['uncertainty'],
                    'num_rollouts': actual_rollouts,
                    'num_successes': successes,
                })
        
        # Compute task-level uncertainty statistics
        if task_analysis['state_details']:
            uncertainties = [s['decisional_uncertainty'] for s in task_analysis['state_details']]
            p_hats = [s['p_hat'] for s in task_analysis['state_details']]
            task_analysis['task_summary']['mean_uncertainty'] = float(np.mean(uncertainties))
            task_analysis['task_summary']['max_uncertainty'] = float(max(uncertainties))
            task_analysis['task_summary']['mean_p_hat'] = float(np.mean(p_hats))
        
        detailed_task_analysis.append(task_analysis)
        
        if simulations_run >= simulation_budget:
            print(f"\n  Budget exhausted after {traj_idx+1} trajectories")
            break
    
    # Create training samples WITH detailed info
    training_samples_success = []
    training_samples_improvement = []
    
    for seg in full_success_segments:
        for step in seg['steps']:
            if step.get('observation') and step.get('action_taken'):
                training_samples_success.append({
                    'state': step['observation'],
                    'goal': seg['goal'],
                    'action': step['action_taken'],
                    'valid_actions': step.get('valid_actions', []),
                    'action_info': step.get('action_info', {}),
                    'task_id': seg['task_id'],
                    'recovery_step': seg['recovery_step'],
                    'final_reward': seg['final_reward'],
                    'source': 'full_success',
                })
    
    for seg in improvement_segments:
        for step in seg['steps']:
            if step.get('observation') and step.get('action_taken'):
                training_samples_improvement.append({
                    'state': step['observation'],
                    'goal': seg['goal'],
                    'action': step['action_taken'],
                    'valid_actions': step.get('valid_actions', []),
                    'action_info': step.get('action_info', {}),
                    'task_id': seg['task_id'],
                    'recovery_step': seg['recovery_step'],
                    'final_reward': seg['final_reward'],
                    'source': 'improvement',
                })
    
    # Combined training samples
    all_training_samples = training_samples_success + training_samples_improvement
    
    # Compute global uncertainty statistics
    all_uncertainties = [s['decisional_uncertainty'] for s in all_simulated_states]
    all_p_hats = [s['p_hat'] for s in all_simulated_states]
    
    stats = {
        'strategy': strategy,
        'action_method': action_method,
        'M': M,
        'N': N,
        'failures_processed': len(detailed_task_analysis),
        'total_states_selected': total_states_selected,
        'simulations_run': simulations_run,
        'successful_replays': simulator.stats['successful_replays'],
        'full_success_count': len(full_success_segments),
        'improvement_count': len(improvement_segments),
        'total_beneficial': len(full_success_segments) + len(improvement_segments),
        'recovery_rate': simulator.stats['recoveries'] / max(simulations_run, 1),
        'training_samples_success': len(training_samples_success),
        'training_samples_improvement': len(training_samples_improvement),
        'training_samples_total': len(all_training_samples),
        
        # Uncertainty statistics
        'uncertainty_stats': {
            'mean': float(np.mean(all_uncertainties)) if all_uncertainties else 0,
            'std': float(np.std(all_uncertainties)) if all_uncertainties else 0,
            'min': float(min(all_uncertainties)) if all_uncertainties else 0,
            'max': float(max(all_uncertainties)) if all_uncertainties else 0,
            'mean_p_hat': float(np.mean(all_p_hats)) if all_p_hats else 0,
        }
    }
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EEF PIPELINE STATISTICS")
    print(f"{'='*70}")
    print(f"  Strategy:                 {strategy}")
    print(f"  Action method:            {action_method}")
    print(f"  M (states/trajectory):    {M}")
    print(f"  N (rollouts/state):       {N}")
    print(f"  Failures processed:       {stats['failures_processed']}")
    print(f"  States selected:          {stats['total_states_selected']}")
    print(f"  Simulations run:          {stats['simulations_run']}")
    print(f"  ---")
    print(f"  Full Success (r=100):     {stats['full_success_count']}")
    print(f"  Improvements (r>orig):    {stats['improvement_count']}")
    print(f"  Total Beneficial:         {stats['total_beneficial']}")
    print(f"  Recovery rate:            {stats['recovery_rate']:.2%}")
    print(f"  ---")
    print(f"  Mean Uncertainty:         {stats['uncertainty_stats']['mean']:.4f}")
    print(f"  Max Uncertainty:          {stats['uncertainty_stats']['max']:.4f}")
    print(f"  Mean p̂:                   {stats['uncertainty_stats']['mean_p_hat']:.4f}")
    print(f"  ---")
    print(f"  Training samples (success):     {stats['training_samples_success']}")
    print(f"  Training samples (improvement): {stats['training_samples_improvement']}")
    print(f"  Training samples (total):       {stats['training_samples_total']}")
    print(f"{'='*70}\n")
    
    return {
        'full_success_segments': full_success_segments,
        'improvement_segments': improvement_segments,
        'all_beneficial_states': all_beneficial_states,
        'all_simulated_states': all_simulated_states,
        'detailed_task_analysis': detailed_task_analysis,
        'training_samples_success': training_samples_success,
        'training_samples_improvement': training_samples_improvement,
        'all_training_samples': all_training_samples,
        'stats': stats,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EEF Pipeline with Decisional Uncertainty")
    parser.add_argument("--failure_data", type=str, required=True,
                       help="Path to pre-collected failure trajectories")
    parser.add_argument("--strategy", type=str, default="baseline", 
                       choices=['baseline', 'entropy'])
    parser.add_argument("--M", type=int, default=5,
                       help="Number of states to select per trajectory")
    parser.add_argument("--N", type=int, default=10,
                       help="Number of rollouts per state for uncertainty estimation")
    parser.add_argument("--simulation_budget", type=int, default=10000)
    parser.add_argument("--greedy", action='store_true', default=False,
                       help="Use greedy instead of softmax (default: softmax)")
    parser.add_argument("--num_trajectories", type=int, default=None,
                       help="Limit number of trajectories to process (default: all)")
    parser.add_argument("--model_path", type=str, 
                       default="./ckpts/web_click/epoch_9/model.pth")
    parser.add_argument("--output_dir", type=str, default="./eef_output_uncertainty")
    parser.add_argument("--verbose", action='store_true', default=True)
    args = parser.parse_args()
    
    action_method = 'greedy' if args.greedy else 'softmax'
    
    print("="*70)
    print("EEF PIPELINE - DECISIONAL UNCERTAINTY")
    print("="*70)
    print(f"  Model: {args.model_path}")
    print(f"  Strategy: {args.strategy}")
    print(f"  M (states/trajectory): {args.M}")
    print(f"  N (rollouts/state): {args.N}")
    print(f"  Budget: {args.simulation_budget}")
    print(f"  Action method: {action_method}")
    print(f"  Num trajectories: {args.num_trajectories if args.num_trajectories else 'all'}")
    print("="*70)
    
    # Setup
    env = setup_environment()
    models = setup_model(args.model_path)
    agent = Agent(models)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load failures
    print(f"\nLoading failures from {args.failure_data}...")
    with open(args.failure_data, 'r') as f:
        failures = json.load(f)
    print(f"  Loaded {len(failures)} failures")
    
    # Limit trajectories if specified
    if args.num_trajectories is not None:
        failures = failures[:args.num_trajectories]
        print(f"  Limited to {len(failures)} trajectories")
    
    # Run EEF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_eef_detailed(
        env, agent, failures,
        M=args.M, N=args.N, strategy=args.strategy,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
        greedy=args.greedy
    )
    
    # Save all outputs
    prefix = f"{args.strategy}_M{args.M}_N{args.N}_{timestamp}"
    
    # 1. MAIN OUTPUT: Detailed task analysis with all rollout info
    path = os.path.join(args.output_dir, f"detailed_analysis_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['detailed_task_analysis'], f, indent=2)
    print(f"✓ Saved detailed analysis for {len(results['detailed_task_analysis'])} tasks to {path}")
    
    # 2. Full success segments
    path = os.path.join(args.output_dir, f"full_success_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['full_success_segments'], f, indent=2)
    print(f"✓ Saved {len(results['full_success_segments'])} full success segments to {path}")
    
    # 3. Improvement segments
    path = os.path.join(args.output_dir, f"improvement_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['improvement_segments'], f, indent=2)
    print(f"✓ Saved {len(results['improvement_segments'])} improvement segments to {path}")
    
    # 4. All beneficial states (for entropy analysis)
    path = os.path.join(args.output_dir, f"beneficial_states_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_beneficial_states'], f, indent=2)
    print(f"✓ Saved {len(results['all_beneficial_states'])} beneficial states to {path}")
    
    # 5. ALL simulated states (for comprehensive analysis)
    path = os.path.join(args.output_dir, f"all_simulated_states_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_simulated_states'], f, indent=2)
    print(f"✓ Saved {len(results['all_simulated_states'])} simulated states to {path}")
    
    # 6. Training samples - success only
    path = os.path.join(args.output_dir, f"training_success_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['training_samples_success'], f, indent=2)
    print(f"✓ Saved {len(results['training_samples_success'])} success training samples to {path}")
    
    # 7. Training samples - improvement only
    path = os.path.join(args.output_dir, f"training_improvement_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['training_samples_improvement'], f, indent=2)
    print(f"✓ Saved {len(results['training_samples_improvement'])} improvement training samples to {path}")
    
    # 8. Training samples - all combined
    path = os.path.join(args.output_dir, f"training_all_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_training_samples'], f, indent=2)
    print(f"✓ Saved {len(results['all_training_samples'])} total training samples to {path}")
    
    # 9. Statistics
    path = os.path.join(args.output_dir, f"stats_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['stats'], f, indent=2)
    print(f"✓ Saved statistics to {path}")
    
    print("\n" + "="*70)
    print("✓ EEF pipeline completed successfully!")
    print("="*70)
    print(f"\nOutput files in {args.output_dir}/:")
    print(f"  - detailed_analysis_{prefix}.json     (MAIN: per-task breakdown with all rollouts)")
    print(f"  - full_success_segments_{prefix}.json (reward=100)")
    print(f"  - improvement_segments_{prefix}.json  (reward > original)")
    print(f"  - beneficial_states_{prefix}.json     (for entropy analysis)")
    print(f"  - all_simulated_states_{prefix}.json  (ALL simulated states)")
    print(f"  - training_success_{prefix}.json      (training data from successes)")
    print(f"  - training_improvement_{prefix}.json  (training data from improvements)")
    print(f"  - training_all_{prefix}.json          (combined training data)")
    print(f"  - stats_{prefix}.json                 (statistics)")
    
    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS JSON STRUCTURE:")
    print(f"{'='*70}")
    print("""
Each task in detailed_analysis_{prefix}.json contains:
{
  "task_id": 123,
  "goal": "Find a blue shirt...",
  "trajectory_length": 8,
  "original_reward": 30,
  "strategy": "baseline|entropy",
  "selected_states": [1, 3, 5],
  "state_details": [
    {
      "state_idx": 1,
      "num_valid_actions": 15,
      "entropy_score": 0.58,
      "position_score": 0.72,
      "num_rollouts": 10,
      "num_successes": 3,
      "num_improvements": 5,
      "p_hat": 0.3,
      "decisional_uncertainty": 0.21,
      "rewards": [0, 100, 30, 100, 50, ...],
      "mean_reward": 45.0,
      "rollouts": [
        {"rollout_idx": 0, "success": false, "reward": 0},
        {"rollout_idx": 1, "success": true, "reward": 100},
        ...
      ]
    },
    ...
  ],
  "task_summary": {
    "total_rollouts": 30,
    "total_successes": 7,
    "mean_uncertainty": 0.18,
    "any_full_success": true
  }
}
""")


if __name__ == "__main__":
    main()