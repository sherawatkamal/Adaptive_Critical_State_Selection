#!/usr/bin/env python3
"""
EEF Pipeline - With Detailed Segment Saving + TRUE Policy Entropy

FIXED: State selection now uses TRUE policy entropy H(π|s) = -Σ π(a|s) log π(a|s)
instead of action count log(|A|).

Changes from original:
1. Added compute_true_entropy() to Agent class
2. Fixed select_critical_states_entropy() to use true entropy from model logprobs
3. Saves full success segments (reward=100) separately
4. Saves improvement segments (reward > original) separately  
5. Saves detailed state info for entropy analysis
6. Includes recovery_step, final_reward, valid_actions in all outputs

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
    """Agent wrapper - NO BART, with TRUE entropy computation"""
    
    def __init__(self, models_dict):
        self.model = models_dict['model']
        self.tokenizer = models_dict['tokenizer']
        self.data_collator = models_dict['data_collator']
        self.process = models_dict['process']
        self.process_goal = models_dict['process_goal']
        self.device = models_dict['device']
    
    def get_action_probs(self, obs: str, valid_acts: List[str]) -> Optional[torch.Tensor]:
        """
        Get action probability distribution from model.
        
        Returns:
            Tensor of probabilities over valid_acts, or None if can't compute
        """
        if not valid_acts:
            return None
        
        # Skip search states - can't compute entropy for generative actions
        if valid_acts[0].startswith('search['):
            return None
        
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
        
        return probs
    
    def compute_true_entropy(self, obs: str, valid_acts: List[str]) -> Tuple[float, float, float]:
        """
        Compute TRUE policy entropy: H(π|s) = -Σ π(a|s) log π(a|s)
        
        This is the CORRECT way to measure model uncertainty.
        NOT log(|valid_actions|) which just counts buttons!
        
        Args:
            obs: Observation string
            valid_acts: List of valid action strings
            
        Returns:
            (entropy, normalized_entropy, max_prob)
            - entropy: Raw entropy in nats
            - normalized_entropy: H / log(|A|), scaled to [0,1]
            - max_prob: Confidence in top action (1 - this = uncertainty)
        """
        probs = self.get_action_probs(obs, valid_acts)
        
        if probs is None:
            # Search state or error - return 0
            return 0.0, 0.0, 1.0
        
        # Compute entropy: H = -Σ p log p
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * torch.log(probs_clamped)).sum().item()
        
        # Normalized entropy: H / H_max where H_max = log(|A|)
        n_actions = len(valid_acts)
        max_entropy = np.log(n_actions) if n_actions > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Max probability (confidence)
        max_prob = probs.max().item()
        
        return entropy, normalized_entropy, max_prob
    
    def get_action(self, obs: str, info: dict, method='softmax') -> Tuple[str, dict]:
        """Get action from the model. Default is softmax for exploration."""
        valid_acts = info.get('valid', [])
        
        if not valid_acts:
            return 'click[back to search]', {'type': 'fallback'}
        
        # Handle search page - NO BART
        if valid_acts[0].startswith('search['):
            action = valid_acts[-1] if valid_acts else 'search[query]'
            return action, {
                'type': 'search', 
                'selected': 'valid_acts[-1]',
                'entropy': 0.0,
                'normalized_entropy': 0.0,
            }
        
        # Get probabilities
        probs = self.get_action_probs(obs, valid_acts)
        
        if probs is None:
            return valid_acts[0], {'type': 'error'}
        
        # Compute entropy
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * torch.log(probs_clamped)).sum().item()
        n_actions = len(valid_acts)
        max_entropy = np.log(n_actions) if n_actions > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Select action
        if method == 'greedy':
            idx = probs.argmax().item()
        else:  # softmax (default)
            idx = torch.multinomial(probs, 1)[0].item()
        
        action = valid_acts[idx] if idx < len(valid_acts) else valid_acts[0]
        return action, {
            'type': 'choice',
            'chosen_idx': idx,
            'num_valid': len(valid_acts),
            'confidence': probs[idx].item(),
            'action_probs': probs.cpu().tolist(),
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
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
# STATE SELECTION (ACSS) - FIXED WITH TRUE ENTROPY
# ============================================================================

def select_critical_states_baseline(trajectory: Dict, M: int = 5, agent=None) -> Tuple[List[int], List[Dict]]:
    """Baseline: Equal-interval skip-length selection"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return [], []
    l = max(1, T // (M + 1))
    indices = [m * l for m in range(1, M + 1) if m * l < T]
    
    # Return empty entropy info for baseline
    entropy_info = [{'state_idx': idx, 'true_entropy': 0.0, 'method': 'baseline'} for idx in indices]
    return indices, entropy_info


def select_critical_states_entropy(trajectory: Dict, M: int = 5, agent=None) -> Tuple[List[int], List[Dict]]:
    """
    ACSS: TRUE policy entropy based selection (FIXED)
    
    Uses H(π|s) = -Σ π(a|s) log π(a|s) from model logprobs.
    NOT log(|A|) which just counts actions!
    
    Args:
        trajectory: Trajectory dict with 'steps'
        M: Number of states to select
        agent: Agent instance (required for true entropy)
        
    Returns:
        (selected_indices, entropy_info)
    """
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return [], []
    
    # If no agent provided, fall back to baseline
    if agent is None:
        print("Warning: No agent provided for entropy calculation. Using baseline.")
        return select_critical_states_baseline(trajectory, M, agent)
    
    scores = []
    for i, step in enumerate(steps[:-1]):  # Exclude last state
        obs = step.get('observation', '')
        valid_acts = step.get('valid_actions', [])
        n_actions = len(valid_acts)
        
        # Position score (slight preference for earlier-middle states)
        position = i / max(T - 1, 1)
        position_score = np.exp(-0.5 * ((position - 0.4) / 0.3) ** 2)
        
        # Check for search state
        is_search = valid_acts and valid_acts[0].startswith('search[')
        
        if is_search or not valid_acts:
            # For search states, use position only (can't compute entropy)
            scores.append({
                'state_idx': i,
                'true_entropy': 0.0,
                'normalized_entropy': 0.0,
                'action_count_score': np.log(n_actions + 1) / np.log(100) if n_actions > 0 else 0,
                'position_score': position_score,
                'combined_score': 0.2 * position_score,  # Low score for search states
                'n_actions': n_actions,
                'max_prob': 1.0,
                'is_search_state': True,
                'method': 'true_entropy',
            })
            continue
        
        # Compute TRUE policy entropy from model
        try:
            entropy, normalized_entropy, max_prob = agent.compute_true_entropy(obs, valid_acts)
        except Exception as e:
            print(f"  Warning: Entropy computation failed for state {i}: {e}")
            entropy, normalized_entropy, max_prob = 0.0, 0.0, 1.0
        
        # Also compute action count score for comparison/logging
        action_count_score = np.log(n_actions + 1) / np.log(100)
        
        # Combined score: primarily entropy, with position tiebreaker
        # Using normalized entropy so it's comparable across different action counts
        combined_score = 0.8 * normalized_entropy + 0.2 * position_score
        
        scores.append({
            'state_idx': i,
            'true_entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'action_count_score': action_count_score,
            'position_score': position_score,
            'combined_score': combined_score,
            'n_actions': n_actions,
            'max_prob': max_prob,
            'is_search_state': False,
            'method': 'true_entropy',
        })
    
    # Filter out search states, then sort by combined score
    choice_states = [s for s in scores if not s.get('is_search_state', False)]
    choice_states.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Select top M
    selected = choice_states[:M]
    indices = sorted([s['state_idx'] for s in selected])
    
    return indices, selected


# ============================================================================
# EEF PIPELINE WITH DETAILED SAVING
# ============================================================================

def run_eef_detailed(env, agent, failures: List[Dict], 
                     M: int = 5, strategy: str = 'baseline',
                     simulation_budget: int = 10000, verbose: bool = True,
                     greedy: bool = False, num_attempts: int = 1):
    """
    Run EEF pipeline with detailed segment saving.
    
    Returns:
        - full_success_segments: States that achieved reward=100
        - improvement_segments: States that improved but didn't reach 100
        - all_beneficial_states: Detailed state info for entropy analysis
        - training_samples: All training data
        - stats: Pipeline statistics
    """
    simulator = EEFSimulator(env, agent, debug=verbose)
    
    # Select state selection function
    if strategy == 'entropy':
        select_states = lambda traj, m: select_critical_states_entropy(traj, m, agent)
    else:
        select_states = lambda traj, m: select_critical_states_baseline(traj, m, agent)
    
    # Default is softmax, use greedy only if explicitly set
    action_method = 'greedy' if greedy else 'softmax'
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: RUNNING EEF (DETAILED) - TRUE ENTROPY")
    print(f"{'='*70}")
    print(f"  Strategy: {strategy}")
    if strategy == 'entropy':
        print(f"  Selection: H(π|s) = -Σ π(a|s) log π(a|s)  [TRUE POLICY ENTROPY]")
    else:
        print(f"  Selection: Equal intervals")
    print(f"  M (states per trajectory): {M}")
    print(f"  Failures: {len(failures)}")
    print(f"  Simulation budget: {simulation_budget}")
    print(f"  Action method: {action_method} (attempts per state: {num_attempts})")
    print(f"{'='*70}\n")
    
    # Separate collections
    full_success_segments = []      # reward = 100
    improvement_segments = []        # reward > original but < 100
    all_beneficial_states = []       # Detailed state info for ALL beneficial states
    all_simulated_states = []        # ALL states that were simulated (for analysis)
    
    simulations_run = 0
    total_states_selected = 0
    
    # Track entropy statistics
    all_selection_info = []
    
    for traj_idx, trajectory in enumerate(failures):
        task_id = trajectory['task_id']
        original_reward = trajectory.get('reward', 0)
        goal = trajectory.get('goal', '')
        traj_length = len(trajectory.get('steps', []))
        
        # Select states using chosen strategy
        critical_states, selection_info = select_states(trajectory, M)
        
        if not critical_states:
            continue
        
        total_states_selected += len(critical_states)
        all_selection_info.extend(selection_info)
        
        if verbose:
            print(f"\n  Task {task_id} ({traj_idx+1}/{len(failures)}):")
            print(f"    Trajectory length: {traj_length} steps")
            print(f"    Original reward: {original_reward:.0f}")
            print(f"    Critical states: {critical_states}")
            if strategy == 'entropy' and selection_info:
                entropies = [f"{s.get('true_entropy', 0):.2f}" for s in selection_info[:3]]
                print(f"    True entropies: {entropies}")
        
        for state_idx in critical_states:
            if simulations_run >= simulation_budget:
                break
            
            # Get the state info from trajectory for entropy analysis
            steps = trajectory.get('steps', [])
            state_obs = steps[state_idx].get('observation', '') if state_idx < len(steps) else ''
            state_valid_actions = steps[state_idx].get('valid_actions', []) if state_idx < len(steps) else []
            
            # Get entropy info for this state
            state_entropy_info = next(
                (s for s in selection_info if s['state_idx'] == state_idx), 
                {'true_entropy': 0.0, 'normalized_entropy': 0.0, 'action_count_score': 0.0}
            )
            
            best_reward = -1
            best_traj = None
            best_success = False
            attempts_made = 0
            
            for attempt in range(num_attempts):
                if simulations_run >= simulation_budget:
                    break
                
                success, reward, sim_traj = simulator.simulate_from_state(
                    task_id, state_idx, trajectory, method=action_method
                )
                simulations_run += 1
                attempts_made += 1
                
                if reward > best_reward:
                    best_reward = reward
                    best_traj = sim_traj
                    best_success = success
                
                if success:
                    break
            
            # Record ALL simulated states for analysis
            simulated_state_info = {
                'task_id': task_id,
                'recovery_step': state_idx,
                'trajectory_length': traj_length,
                'original_reward': original_reward,
                'final_reward': best_reward,
                'is_success': best_success,
                'is_improvement': best_reward > original_reward + 10,
                'state': state_obs[:2000],  # Truncate for storage
                'valid_actions': state_valid_actions,
                'num_valid_actions': len(state_valid_actions),
                'attempts_made': attempts_made,
                'strategy': strategy,
                # Entropy info
                'true_entropy': state_entropy_info.get('true_entropy', 0.0),
                'normalized_entropy': state_entropy_info.get('normalized_entropy', 0.0),
                'action_count_score': state_entropy_info.get('action_count_score', 0.0),
            }
            all_simulated_states.append(simulated_state_info)
            
            # Check for improvement
            is_full_success = best_success and best_reward >= 100
            is_improvement = best_reward > original_reward + 10
            
            if is_full_success or is_improvement:
                status = "SUCCESS" if is_full_success else "IMPROVED"
                if verbose:
                    print(f"    Step {state_idx}: ✓ {status}! {original_reward:.0f} → {best_reward:.0f} "
                          f"(H={state_entropy_info.get('true_entropy', 0):.2f})")
                
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
                    'true_entropy': state_entropy_info.get('true_entropy', 0.0),
                    'normalized_entropy': state_entropy_info.get('normalized_entropy', 0.0),
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
                    'true_entropy': state_entropy_info.get('true_entropy', 0.0),
                    'normalized_entropy': state_entropy_info.get('normalized_entropy', 0.0),
                    'action_count_score': state_entropy_info.get('action_count_score', 0.0),
                })
            else:
                if verbose:
                    print(f"    Step {state_idx}: ✗ failed ({best_reward:.0f})")
        
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
    
    # Compute entropy statistics
    all_true_entropies = [s.get('true_entropy', 0) for s in all_selection_info if not s.get('is_search_state', False)]
    all_action_counts = [s.get('action_count_score', 0) for s in all_selection_info if not s.get('is_search_state', False)]
    
    stats = {
        'failures_processed': len(failures),
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
        'strategy': strategy,
        'action_method': action_method,
        # Entropy statistics
        'entropy_stats': {
            'mean_true_entropy': float(np.mean(all_true_entropies)) if all_true_entropies else 0,
            'max_true_entropy': float(np.max(all_true_entropies)) if all_true_entropies else 0,
            'mean_action_count_score': float(np.mean(all_action_counts)) if all_action_counts else 0,
        }
    }
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EEF PIPELINE STATISTICS (DETAILED - TRUE ENTROPY)")
    print(f"{'='*70}")
    print(f"  Strategy:                 {strategy}")
    print(f"  Action method:            {action_method}")
    print(f"  Failures processed:       {stats['failures_processed']}")
    print(f"  States selected:          {stats['total_states_selected']}")
    print(f"  Simulations run:          {stats['simulations_run']}")
    print(f"  ---")
    print(f"  Full Success (r=100):     {stats['full_success_count']}")
    print(f"  Improvements (r>orig):    {stats['improvement_count']}")
    print(f"  Total Beneficial:         {stats['total_beneficial']}")
    print(f"  Recovery rate:            {stats['recovery_rate']:.2%}")
    print(f"  ---")
    if strategy == 'entropy':
        print(f"  Mean True Entropy:        {stats['entropy_stats']['mean_true_entropy']:.4f}")
        print(f"  Max True Entropy:         {stats['entropy_stats']['max_true_entropy']:.4f}")
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
        'training_samples_success': training_samples_success,
        'training_samples_improvement': training_samples_improvement,
        'all_training_samples': all_training_samples,
        'stats': stats,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EEF Pipeline - Detailed Saving (TRUE ENTROPY)")
    parser.add_argument("--failure_data", type=str, required=True,
                       help="Path to pre-collected failure trajectories")
    parser.add_argument("--strategy", type=str, default="baseline", 
                       choices=['baseline', 'entropy'])
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--simulation_budget", type=int, default=10000)
    parser.add_argument("--greedy", action='store_true', default=False,
                       help="Use greedy instead of softmax (default: softmax)")
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--num_trajectories", type=int, default=None,
                       help="Limit number of trajectories to process (default: all)")
    parser.add_argument("--model_path", type=str, 
                       default="./ckpts/web_click/epoch_9/model.pth")
    parser.add_argument("--output_dir", type=str, default="./eef_output_detailed")
    parser.add_argument("--verbose", action='store_true', default=True)
    args = parser.parse_args()
    
    action_method = 'greedy' if args.greedy else 'softmax'
    
    print("="*70)
    print("EEF PIPELINE - DETAILED SAVING (TRUE ENTROPY)")
    print("="*70)
    print(f"  Model: {args.model_path}")
    print(f"  Strategy: {args.strategy}")
    if args.strategy == 'entropy':
        print(f"  Selection: H(π|s) = -Σ π(a|s) log π(a|s)  [TRUE POLICY ENTROPY]")
    print(f"  M: {args.M}")
    print(f"  Budget: {args.simulation_budget}")
    print(f"  Action method: {action_method} (attempts: {args.num_attempts})")
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
        M=args.M, strategy=args.strategy,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
        greedy=args.greedy,
        num_attempts=args.num_attempts
    )
    
    # Save all outputs
    prefix = f"{args.strategy}_{timestamp}"
    
    # 1. Full success segments
    path = os.path.join(args.output_dir, f"full_success_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['full_success_segments'], f, indent=2)
    print(f"✓ Saved {len(results['full_success_segments'])} full success segments to {path}")
    
    # 2. Improvement segments
    path = os.path.join(args.output_dir, f"improvement_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['improvement_segments'], f, indent=2)
    print(f"✓ Saved {len(results['improvement_segments'])} improvement segments to {path}")
    
    # 3. All beneficial states (for entropy analysis)
    path = os.path.join(args.output_dir, f"beneficial_states_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_beneficial_states'], f, indent=2)
    print(f"✓ Saved {len(results['all_beneficial_states'])} beneficial states to {path}")
    
    # 4. ALL simulated states (for comprehensive analysis)
    path = os.path.join(args.output_dir, f"all_simulated_states_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_simulated_states'], f, indent=2)
    print(f"✓ Saved {len(results['all_simulated_states'])} simulated states to {path}")
    
    # 5. Training samples - success only
    path = os.path.join(args.output_dir, f"training_success_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['training_samples_success'], f, indent=2)
    print(f"✓ Saved {len(results['training_samples_success'])} success training samples to {path}")
    
    # 6. Training samples - improvement only
    path = os.path.join(args.output_dir, f"training_improvement_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['training_samples_improvement'], f, indent=2)
    print(f"✓ Saved {len(results['training_samples_improvement'])} improvement training samples to {path}")
    
    # 7. Training samples - all combined
    path = os.path.join(args.output_dir, f"training_all_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_training_samples'], f, indent=2)
    print(f"✓ Saved {len(results['all_training_samples'])} total training samples to {path}")
    
    # 8. Statistics
    path = os.path.join(args.output_dir, f"stats_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['stats'], f, indent=2)
    print(f"✓ Saved statistics to {path}")
    
    print("\n" + "="*70)
    print("✓ EEF pipeline completed successfully!")
    print("="*70)
    print(f"\nOutput files in {args.output_dir}/:")
    print(f"  - full_success_segments_{prefix}.json  (reward=100)")
    print(f"  - improvement_segments_{prefix}.json   (reward > original)")
    print(f"  - beneficial_states_{prefix}.json      (for entropy analysis)")
    print(f"  - all_simulated_states_{prefix}.json   (ALL simulated states)")
    print(f"  - training_success_{prefix}.json       (training data from successes)")
    print(f"  - training_improvement_{prefix}.json   (training data from improvements)")
    print(f"  - training_all_{prefix}.json           (combined training data)")
    print(f"  - stats_{prefix}.json                  (statistics)")


if __name__ == "__main__":
    main()