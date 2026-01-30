#!/usr/bin/env python3
"""
EEF Pipeline - With Diagnosis Model Integration (PARALLEL VERSION)

PARALLEL VERSION: Uses multiprocessing to speed up simulations with 3 workers

Strategies:
1. baseline: Equal interval sampling
2. entropy: Top-M by true policy entropy
3. stratified_entropy: Balanced sampling across LOW/MEDIUM/HIGH entropy bins
4. diagnosis: Use trained failure diagnosis model to predict mistake step

Usage:
    # Baseline with 3 workers
    python eef_detailed_with_diagnosis_parallel.py --failure_data ./failures.json --strategy baseline --M 5 --num_workers 3

    # Diagnosis model with 3 workers (NEW)
    python eef_detailed_with_diagnosis_parallel.py --failure_data ./failures.json --strategy diagnosis \
        --diagnosis_model_path ./Qwen2.5/qwen25_instruct_v1 \
        --diagnosis_base_model Qwen/Qwen2.5-3B-Instruct \
        --M 3 --diagnosis_window 1 --num_workers 3
"""

import os
import sys
import json
import argparse
import random
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# DIAGNOSIS MODEL SELECTOR (NEW)
# ============================================================================

class DiagnosisModelSelector:
    """Use trained failure diagnosis model to select simulation states."""
    
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Load the trained diagnosis model.
        
        Args:
            model_path: Path to fine-tuned LoRA adapter
            base_model: Base model name (must match training)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print(f"Loading diagnosis model...")
        print(f"  Base model: {base_model}")
        print(f"  Adapter path: {model_path}")
        
        self.base_model_name = base_model
        self.model_path = model_path
        
        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        print(f"✓ Diagnosis model loaded on {self.device}")
        
        # Stats tracking
        self.stats = {
            'predictions_made': 0,
            'parse_failures': 0,
            'step_distribution': {},
        }
    
    def format_trajectory_for_diagnosis(self, trajectory: Dict) -> str:
        """
        Format trajectory for diagnosis model input.
        
        Args:
            trajectory: Dict with 'goal', 'steps' keys
            
        Returns:
            Formatted string for model input
        """
        goal = trajectory.get('goal', '')
        steps = trajectory.get('steps', [])
        
        lines = [f"Goal: {goal}\n"]
        
        for i, step in enumerate(steps):
            action = step.get('action_taken', '')
            obs = step.get('observation', '')
            
            # Truncate observation to avoid exceeding context length
            if len(obs) > 500:
                obs = obs[:500] + "..."
            
            lines.append(f"Step {i} | action: {action}")
            lines.append(f"Observation: {obs}\n")
        
        return "\n".join(lines)
    
    def predict_mistake_step(self, trajectory: Dict) -> Tuple[int, str]:
        """
        Predict which step contains the critical mistake.
        
        Args:
            trajectory: Failed trajectory dict
            
        Returns:
            (predicted_step, model_response)
        """
        self.stats['predictions_made'] += 1
        
        traj_text = self.format_trajectory_for_diagnosis(trajectory)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a Failure Diagnosis Model. Given a failed trajectory, identify the single critical mistake step. Optionally explain which action should have been taken instead."
            },
            {
                "role": "user", 
                "content": f"Failed trajectory:\n{traj_text}\n\nIdentify the single critical mistake step."
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract step number from response
        match = re.search(r'step\s*(\d+)', response.lower())
        if match:
            predicted_step = int(match.group(1))
            # Track distribution
            self.stats['step_distribution'][predicted_step] = \
                self.stats['step_distribution'].get(predicted_step, 0) + 1
            return predicted_step, response
        else:
            self.stats['parse_failures'] += 1
            return 0, response  # Default to step 0 if parsing fails
    
    def get_stats(self) -> Dict:
        """Return prediction statistics."""
        return self.stats


def select_critical_states_diagnosis(
    trajectory: Dict, 
    M: int = 3,
    diagnosis_model: DiagnosisModelSelector = None,
    window: int = 1,
    agent = None,  # For compatibility with other selectors
) -> Tuple[List[int], List[Dict]]:
    """
    Use diagnosis model to select states for simulation.
    
    Instead of computing entropy at every step, we:
    1. Ask the diagnosis model: "Which step was the mistake?"
    2. Simulate steps around that prediction (predicted ± window)
    
    Args:
        trajectory: Failed trajectory dict
        M: Max states to select (usually 3 for diagnosis)
        diagnosis_model: Trained DiagnosisModelSelector instance
        window: How many steps around prediction to include (±window)
        agent: Unused, for API compatibility
        
    Returns:
        (selected_indices, selection_info)
    """
    steps = trajectory.get('steps', [])
    T = len(steps)
    
    if T <= 1:
        return [], []
    
    if diagnosis_model is None:
        print("Warning: No diagnosis model provided. Falling back to baseline.")
        return select_critical_states_baseline(trajectory, M, agent)
    
    # Get model's prediction
    predicted_step, model_response = diagnosis_model.predict_mistake_step(trajectory)
    
    # Clamp to valid range (exclude last step, which is the failure)
    max_valid_step = T - 2  # -1 for 0-indexing, -1 to exclude last step
    predicted_step = max(0, min(predicted_step, max_valid_step))
    
    # Select steps within window around prediction
    candidates = []
    for offset in range(-window, window + 1):
        step_idx = predicted_step + offset
        if 0 <= step_idx <= max_valid_step:
            candidates.append(step_idx)
    
    # Remove duplicates and sort
    candidates = sorted(set(candidates))
    
    # Limit to M states
    if len(candidates) > M:
        # Prioritize: predicted step first, then closest to prediction
        candidates_with_dist = [(c, abs(c - predicted_step)) for c in candidates]
        candidates_with_dist.sort(key=lambda x: x[1])
        candidates = sorted([c for c, _ in candidates_with_dist[:M]])
    
    # Build selection info
    selection_info = []
    for idx in candidates:
        selection_info.append({
            'state_idx': idx,
            'predicted_mistake_step': predicted_step,
            'offset_from_prediction': idx - predicted_step,
            'is_predicted_step': idx == predicted_step,
            'model_response': model_response[:200] if idx == predicted_step else '',
            'method': 'diagnosis_model',
            # Placeholders for compatibility
            'true_entropy': 0.0,
            'normalized_entropy': 0.0,
            'action_count_score': 0.0,
            'combined_score': 1.0 if idx == predicted_step else 0.5,
        })
    
    return candidates, selection_info


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
    env_args.human_goals = 1
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
        """Get action probability distribution from model."""
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
        """Compute TRUE policy entropy: H(π|s) = -Σ π(a|s) log π(a|s)"""
        probs = self.get_action_probs(obs, valid_acts)
        
        if probs is None:
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
    """EEF Simulator - FIXED VERSION (no goal override)"""
    
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
            'goal_mismatches': 0,
        }
    
    def simulate_from_state(self, target_step: int, 
                           trajectory: Dict, method='softmax') -> Tuple[bool, float, List[Dict], List[Dict]]:
        """Simulate from a specific step in the trajectory."""
        self.stats['total_simulations'] += 1
        
        steps = trajectory.get('steps', [])
        task_id = trajectory.get('task_id')
        original_goal = trajectory.get('goal', '')
        
        if task_id is None:
            print(f"⚠️  WARNING: Trajectory missing task_id, cannot simulate")
            return False, 0.0, [], []
        
        if target_step >= len(steps):
            return False, 0.0, [], []
        
        # Reset to the CORRECT task (from trajectory)
        obs, info = self.env.reset(task_id)
        
        if self.debug:
            env_goal = info.get('goal', '')
            if env_goal != original_goal:
                print(f"⚠️  Goal mismatch detected!")
                self.stats['goal_mismatches'] += 1
        
        simulation_traj = []
        full_trajectory = []
        
        # PHASE 1: Replay actions to reach target state
        for step_idx in range(target_step):
            if step_idx >= len(steps):
                break
            
            action = steps[step_idx].get('action_taken', '')
            if not action:
                continue
            
            step_record = {
                'step': step_idx,
                'observation': obs,
                'action_taken': action,
                'is_replay': True,
                'valid_actions': info.get('valid', [])
            }
            
            simulation_traj.append(step_record)
            full_trajectory.append(step_record)
            
            obs, reward, done, info = self.env.step(action)
            
            if done:
                return reward == 10.0, reward * 10, simulation_traj, full_trajectory
        
        self.stats['successful_replays'] += 1
        
        # PHASE 2: Agent-driven simulation from target state
        for sim_step in range(self.max_steps):
            valid_acts = info.get('valid', [])
            if not valid_acts:
                break
            
            action, action_info = self.agent.get_action(obs, info, method=method)
            
            step_record = {
                'step': target_step + sim_step,
                'observation': obs,
                'action_taken': action,
                'action_info': action_info,
                'is_replay': False,
                'valid_actions': valid_acts,
            }
            
            obs, reward, done, info = self.env.step(action)
            
            step_record['reward'] = reward * 10
            step_record['done'] = done
            
            simulation_traj.append(step_record)
            full_trajectory.append(step_record)
            
            if done:
                success = reward == 10.0
                if success:
                    self.stats['recoveries'] += 1
                return success, reward * 10, simulation_traj, full_trajectory
        
        return False, 0.0, simulation_traj, full_trajectory


# ============================================================================
# STATE SELECTION STRATEGIES
# ============================================================================

def select_critical_states_baseline(trajectory: Dict, M: int = 5, agent=None) -> Tuple[List[int], List[Dict]]:
    """Baseline: Equal-interval skip-length selection"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return [], []
    l = max(1, T // (M + 1))
    indices = [m * l for m in range(1, M + 1) if m * l < T]
    
    entropy_info = [{'state_idx': idx, 'true_entropy': 0.0, 'method': 'baseline'} for idx in indices]
    return indices, entropy_info


def select_critical_states_entropy(trajectory: Dict, M: int = 5, agent=None) -> Tuple[List[int], List[Dict]]:
    """ACSS: TRUE policy entropy based selection"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return [], []
    
    if agent is None:
        print("Warning: No agent provided for entropy calculation. Using baseline.")
        return select_critical_states_baseline(trajectory, M, agent)
    
    scores = []
    for i, step in enumerate(steps[:-1]):
        obs = step.get('observation', '')
        valid_acts = step.get('valid_actions', [])
        n_actions = len(valid_acts)
        
        position = i / max(T - 1, 1)
        position_score = np.exp(-0.5 * ((position - 0.4) / 0.3) ** 2)
        
        is_search = valid_acts and valid_acts[0].startswith('search[')
        
        if is_search or not valid_acts:
            scores.append({
                'state_idx': i,
                'true_entropy': 0.0,
                'normalized_entropy': 0.0,
                'action_count_score': np.log(n_actions + 1) / np.log(100) if n_actions > 0 else 0,
                'position_score': position_score,
                'combined_score': 0.2 * position_score,
                'n_actions': n_actions,
                'max_prob': 1.0,
                'is_search_state': True,
                'method': 'true_entropy',
            })
            continue
        
        try:
            entropy, normalized_entropy, max_prob = agent.compute_true_entropy(obs, valid_acts)
        except Exception as e:
            print(f"  Warning: Entropy computation failed for state {i}: {e}")
            entropy, normalized_entropy, max_prob = 0.0, 0.0, 1.0
        
        action_count_score = np.log(n_actions + 1) / np.log(100)
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
    
    choice_states = [s for s in scores if not s.get('is_search_state', False)]
    choice_states.sort(key=lambda x: x['combined_score'], reverse=True)
    
    selected = choice_states[:M]
    indices = sorted([s['state_idx'] for s in selected])
    
    return indices, selected


def select_critical_states_stratified_entropy(trajectory: Dict, M: int = 5, agent=None) -> Tuple[List[int], List[Dict]]:
    """Stratified ACSS: Balanced sampling across entropy levels"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return [], []
    
    if agent is None:
        print("Warning: No agent provided for entropy calculation. Using baseline.")
        return select_critical_states_baseline(trajectory, M, agent)
    
    scores = []
    for i, step in enumerate(steps[:-1]):
        obs = step.get('observation', '')
        valid_acts = step.get('valid_actions', [])
        n_actions = len(valid_acts)
        
        position = i / max(T - 1, 1)
        position_score = np.exp(-0.5 * ((position - 0.4) / 0.3) ** 2)
        
        is_search = valid_acts and valid_acts[0].startswith('search[')
        
        if is_search or not valid_acts:
            continue
        
        try:
            entropy, normalized_entropy, max_prob = agent.compute_true_entropy(obs, valid_acts)
        except Exception as e:
            print(f"  Warning: Entropy computation failed for state {i}: {e}")
            entropy, normalized_entropy, max_prob = 0.0, 0.0, 1.0
        
        action_count_score = np.log(n_actions + 1) / np.log(100)
        
        scores.append({
            'state_idx': i,
            'true_entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'action_count_score': action_count_score,
            'position_score': position_score,
            'n_actions': n_actions,
            'max_prob': max_prob,
            'is_search_state': False,
            'method': 'stratified_entropy',
        })
    
    if not scores:
        return [], []
    
    all_entropies = [s['normalized_entropy'] for s in scores]
    p33 = np.percentile(all_entropies, 33)
    p67 = np.percentile(all_entropies, 67)
    
    for s in scores:
        norm_ent = s['normalized_entropy']
        if norm_ent < p33:
            s['entropy_bin'] = 'LOW'
        elif norm_ent < p67:
            s['entropy_bin'] = 'MEDIUM'
        else:
            s['entropy_bin'] = 'HIGH'
    
    low_states = [s for s in scores if s['entropy_bin'] == 'LOW']
    medium_states = [s for s in scores if s['entropy_bin'] == 'MEDIUM']
    high_states = [s for s in scores if s['entropy_bin'] == 'HIGH']
    
    low_states.sort(key=lambda x: x['position_score'], reverse=True)
    medium_states.sort(key=lambda x: x['position_score'], reverse=True)
    high_states.sort(key=lambda x: x['position_score'], reverse=True)
    
    m_per_bin = M // 3
    m_remainder = M % 3
    
    selected = []
    selected.extend(low_states[:m_per_bin])
    n_medium = m_per_bin + (1 if m_remainder >= 1 else 0)
    selected.extend(medium_states[:n_medium])
    n_high = m_per_bin + (1 if m_remainder >= 2 else 0)
    selected.extend(high_states[:n_high])
    
    for s in selected:
        s['combined_score'] = s['normalized_entropy']
    
    indices = sorted([s['state_idx'] for s in selected])
    
    bin_counts = {
        'LOW': len([s for s in selected if s['entropy_bin'] == 'LOW']),
        'MEDIUM': len([s for s in selected if s['entropy_bin'] == 'MEDIUM']),
        'HIGH': len([s for s in selected if s['entropy_bin'] == 'HIGH']),
    }
    
    if len(selected) > 0:
        print(f"    Stratified selection: LOW={bin_counts['LOW']}, MED={bin_counts['MEDIUM']}, HIGH={bin_counts['HIGH']}")
    
    return indices, selected


# ============================================================================
# PARALLEL WORKER FUNCTION
# ============================================================================

def worker_simulate_trajectory(args):
    """
    Worker function for parallel simulation.
    
    Args:
        args: Tuple of (traj_idx, trajectory, critical_states, selection_info, 
                       model_path, action_method, num_attempts, strategy,
                       verbose, shared_stats_dict)
    
    Returns:
        Dict with results for this trajectory
    """
    (traj_idx, trajectory, critical_states, selection_info, 
     model_path, action_method, num_attempts, strategy, 
     verbose, shared_stats_dict) = args
    
    # Each worker needs its own environment and agent
    env = setup_environment()
    models = setup_model(model_path)
    agent = Agent(models)
    simulator = EEFSimulator(env, agent, debug=verbose)
    
    task_id = trajectory['task_id']
    original_reward = trajectory.get('reward', 0)
    goal = trajectory.get('goal', '')
    traj_length = len(trajectory.get('steps', []))
    
    # Results for this trajectory
    traj_results = {
        'task_id': task_id,
        'full_success_segments': [],
        'improvement_segments': [],
        'failure_segments': [],
        'simulated_states': [],
        'simulation_stats': [],
        'simulations_run': 0,
    }
    
    if verbose:
        print(f"\n  [Worker] Task {task_id} ({traj_idx}):")
        print(f"    Trajectory length: {traj_length} steps")
        print(f"    Original reward: {original_reward:.0f}")
        print(f"    Critical states: {critical_states}")
    
    for state_idx in critical_states:
        steps = trajectory.get('steps', [])
        state_obs = steps[state_idx].get('observation', '') if state_idx < len(steps) else ''
        state_valid_actions = steps[state_idx].get('valid_actions', []) if state_idx < len(steps) else []
        
        state_entropy_info = next(
            (s for s in selection_info if s['state_idx'] == state_idx), 
            {'true_entropy': 0.0, 'normalized_entropy': 0.0, 'action_count_score': 0.0}
        )
        
        best_reward = -1
        best_traj = None
        best_full_traj = None
        best_success = False
        attempts_made = 0
        
        for attempt in range(num_attempts):
            success, reward, sim_traj, full_traj = simulator.simulate_from_state(
                state_idx, trajectory, method=action_method
            )
            traj_results['simulations_run'] += 1
            attempts_made += 1
            
            if reward > best_reward:
                best_reward = reward
                best_traj = sim_traj
                best_full_traj = full_traj
                best_success = success
            
            if success:
                break
        
        # Record simulated state
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
            'attempts_made': attempts_made,
            'strategy': strategy,
            'true_entropy': state_entropy_info.get('true_entropy', 0.0),
            'normalized_entropy': state_entropy_info.get('normalized_entropy', 0.0),
            'action_count_score': state_entropy_info.get('action_count_score', 0.0),
        }
        
        if strategy == 'diagnosis':
            simulated_state_info['predicted_mistake_step'] = state_entropy_info.get('predicted_mistake_step', -1)
            simulated_state_info['offset_from_prediction'] = state_entropy_info.get('offset_from_prediction', 0)
            simulated_state_info['is_predicted_step'] = state_entropy_info.get('is_predicted_step', False)
        
        traj_results['simulated_states'].append(simulated_state_info)
        
        # Simulation stats
        traj_results['simulation_stats'].append({
            'task_id': task_id,
            'state_idx': state_idx,
            'true_entropy': state_entropy_info.get('true_entropy', 0.0),
            'normalized_entropy': state_entropy_info.get('normalized_entropy', 0.0),
            'success': best_success,
            'final_reward': best_reward,
            'original_reward': original_reward,
            'improvement': best_reward - original_reward,
            'is_improvement': best_reward > original_reward + 10,
            'strategy': strategy,
        })
        
        # Check for improvement
        is_full_success = best_success and best_reward >= 100
        is_improvement = best_reward > original_reward + 10
        
        if is_full_success or is_improvement:
            status = "SUCCESS" if is_full_success else "IMPROVED"
            if verbose:
                extra = ""
                if strategy == 'diagnosis':
                    extra = f" (pred={state_entropy_info.get('predicted_mistake_step', '?')})"
                print(f"    Step {state_idx}: ✓ {status}! {original_reward:.0f} → {best_reward:.0f}{extra}")
            
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
                'full_trajectory': best_full_traj,
                'steps': new_actions,
                'strategy': strategy,
            }
            
            if strategy == 'diagnosis':
                segment_data['predicted_mistake_step'] = state_entropy_info.get('predicted_mistake_step', -1)
            
            if is_full_success:
                traj_results['full_success_segments'].append(segment_data)
            else:
                traj_results['improvement_segments'].append(segment_data)
        else:
            new_actions = [s for s in best_traj if not s.get('is_replay', False)]
            
            failure_segment = {
                'task_id': task_id,
                'goal': goal,
                'recovery_step': state_idx,
                'trajectory_length': traj_length,
                'original_reward': original_reward,
                'final_reward': best_reward,
                'is_full_success': False,
                'num_recovery_steps': len(new_actions),
                'state_observation': state_obs[:2000],
                'state_valid_actions': state_valid_actions,
                'true_entropy': state_entropy_info.get('true_entropy', 0.0),
                'normalized_entropy': state_entropy_info.get('normalized_entropy', 0.0),
                'full_trajectory': best_full_traj,
                'steps': new_actions,
                'strategy': strategy,
            }
            
            if strategy == 'diagnosis':
                failure_segment['predicted_mistake_step'] = state_entropy_info.get('predicted_mistake_step', -1)
            
            traj_results['failure_segments'].append(failure_segment)
            
            if verbose:
                print(f"    Step {state_idx}: ✗ failed ({best_reward:.0f})")
    
    # Update shared stats
    if shared_stats_dict is not None:
        shared_stats_dict['simulations_run'] += traj_results['simulations_run']
        shared_stats_dict['recoveries'] += sum(1 for s in traj_results['simulation_stats'] if s['success'])
    
    return traj_results


# ============================================================================
# PARALLEL EEF PIPELINE
# ============================================================================

def run_eef_parallel(env, agent, failures: List[Dict], 
                    M: int = 5, strategy: str = 'baseline',
                    simulation_budget: int = 10000, verbose: bool = True,
                    greedy: bool = False, num_attempts: int = 1,
                    diagnosis_model: DiagnosisModelSelector = None,
                    diagnosis_window: int = 1,
                    num_workers: int = 3,
                    model_path: str = "./ckpts/web_click/epoch_9/model.pth"):
    """
    Run EEF pipeline with PARALLEL workers.
    
    Args:
        num_workers: Number of parallel workers (default: 3)
        model_path: Path to agent model (each worker will load its own copy)
    """
    
    # Select state selection function based on strategy
    if strategy == 'diagnosis':
        if diagnosis_model is None:
            raise ValueError("diagnosis_model required for strategy='diagnosis'")
        select_states = lambda traj, m: select_critical_states_diagnosis(
            traj, m, diagnosis_model, window=diagnosis_window, agent=agent
        )
        print(f"\n  Using DIAGNOSIS MODEL for state selection")
        print(f"  Model: {diagnosis_model.model_path}")
        print(f"  Window: ±{diagnosis_window} steps around prediction")
    elif strategy == 'entropy':
        select_states = lambda traj, m: select_critical_states_entropy(traj, m, agent)
    elif strategy == 'stratified_entropy':
        select_states = lambda traj, m: select_critical_states_stratified_entropy(traj, m, agent)
    else:  # baseline
        select_states = lambda traj, m: select_critical_states_baseline(traj, m, agent)
    
    action_method = 'greedy' if greedy else 'softmax'
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: RUNNING EEF (PARALLEL - {num_workers} workers)")
    print(f"{'='*70}")
    print(f"  Strategy: {strategy}")
    print(f"  M (states per trajectory): {M}")
    print(f"  Action method: {action_method}")
    print(f"  Workers: {num_workers}")
    
    # PHASE 1: State selection (serial - fast)
    print(f"\n  PHASE 1: Selecting critical states...")
    trajectory_jobs = []
    total_states_selected = 0
    all_selection_info = []
    diagnosis_predictions = []
    
    for traj_idx, trajectory in enumerate(failures):
        task_id = trajectory['task_id']
        traj_length = len(trajectory.get('steps', []))
        
        critical_states, selection_info = select_states(trajectory, M)
        
        if not critical_states:
            continue
        
        total_states_selected += len(critical_states)
        all_selection_info.extend(selection_info)
        
        # Track diagnosis predictions
        if strategy == 'diagnosis' and selection_info:
            pred_step = selection_info[0].get('predicted_mistake_step', -1)
            diagnosis_predictions.append({
                'task_id': task_id,
                'predicted_step': pred_step,
                'selected_states': critical_states,
                'traj_length': traj_length,
            })
        
        trajectory_jobs.append((
            traj_idx, trajectory, critical_states, selection_info,
            model_path, action_method, num_attempts, strategy,
            verbose, None  # shared_stats_dict placeholder
        ))
    
    print(f"  ✓ Selected {total_states_selected} states across {len(trajectory_jobs)} trajectories")
    
    # PHASE 2: Parallel simulation
    print(f"\n  PHASE 2: Running simulations with {num_workers} workers...")
    
    # Shared stats across workers
    manager = Manager()
    shared_stats = manager.dict()
    shared_stats['simulations_run'] = 0
    shared_stats['recoveries'] = 0
    
    # Update jobs with shared stats
    trajectory_jobs = [(traj_idx, traj, states, info, mp, am, na, strat, verb, shared_stats)
                       for traj_idx, traj, states, info, mp, am, na, strat, verb, _ in trajectory_jobs]
    
    # Run parallel simulations
    all_results = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(worker_simulate_trajectory, trajectory_jobs):
            all_results.append(result)
            
            # Progress update
            completed = len(all_results)
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{len(trajectory_jobs)} trajectories completed")
    
    print(f"  ✓ All simulations completed!")
    
    # PHASE 3: Aggregate results
    print(f"\n  PHASE 3: Aggregating results...")
    
    full_success_segments = []
    improvement_segments = []
    failure_segments = []
    all_simulated_states = []
    simulation_stats = []
    
    for result in all_results:
        full_success_segments.extend(result['full_success_segments'])
        improvement_segments.extend(result['improvement_segments'])
        failure_segments.extend(result['failure_segments'])
        all_simulated_states.extend(result['simulated_states'])
        simulation_stats.extend(result['simulation_stats'])
    
    # Create training samples
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
    
    all_training_samples = training_samples_success + training_samples_improvement
    
    # Compute statistics
    all_true_entropies = [s.get('true_entropy', 0) for s in all_selection_info if not s.get('is_search_state', False)]
    simulations_run = shared_stats['simulations_run']
    recoveries = shared_stats['recoveries']
    
    stats = {
        'failures_processed': len(failures),
        'total_states_selected': total_states_selected,
        'simulations_run': simulations_run,
        'full_success_count': len(full_success_segments),
        'improvement_count': len(improvement_segments),
        'failure_count': len(failure_segments),
        'total_beneficial': len(full_success_segments) + len(improvement_segments),
        'recovery_rate': recoveries / max(simulations_run, 1),
        'training_samples_success': len(training_samples_success),
        'training_samples_improvement': len(training_samples_improvement),
        'training_samples_total': len(all_training_samples),
        'strategy': strategy,
        'action_method': action_method,
        'num_workers': num_workers,
        'entropy_stats': {
            'mean_true_entropy': float(np.mean(all_true_entropies)) if all_true_entropies else 0,
            'max_true_entropy': float(np.max(all_true_entropies)) if all_true_entropies else 0,
        }
    }
    
    # Add diagnosis-specific stats
    if strategy == 'diagnosis' and diagnosis_model:
        stats['diagnosis_stats'] = diagnosis_model.get_stats()
        stats['diagnosis_predictions'] = diagnosis_predictions
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EEF PIPELINE STATISTICS (PARALLEL)")
    print(f"{'='*70}")
    print(f"  Strategy:                 {strategy}")
    print(f"  Action method:            {action_method}")
    print(f"  Workers:                  {num_workers}")
    print(f"  Failures processed:       {stats['failures_processed']}")
    print(f"  States selected:          {stats['total_states_selected']}")
    print(f"  Simulations run:          {stats['simulations_run']}")
    print(f"  ---")
    print(f"  Full Success (r=100):     {stats['full_success_count']}")
    print(f"  Improvements (r>orig):    {stats['improvement_count']}")
    print(f"  Failures (no improvement): {stats['failure_count']}")
    print(f"  Total Beneficial:         {stats['total_beneficial']}")
    print(f"  Recovery rate:            {stats['recovery_rate']:.2%}")
    print(f"  ---")
    if strategy == 'diagnosis' and diagnosis_model:
        diag_stats = diagnosis_model.get_stats()
        print(f"  Diagnosis predictions:    {diag_stats['predictions_made']}")
        print(f"  Parse failures:           {diag_stats['parse_failures']}")
        if diag_stats['step_distribution']:
            top_preds = sorted(diag_stats['step_distribution'].items(), key=lambda x: -x[1])[:5]
            print(f"  Top predicted steps:      {top_preds}")
    print(f"  ---")
    print(f"  Training samples (success):     {stats['training_samples_success']}")
    print(f"  Training samples (improvement): {stats['training_samples_improvement']}")
    print(f"  Training samples (total):       {stats['training_samples_total']}")
    
    # Compute efficiency metrics
    if total_states_selected > 0:
        efficiency = stats['total_beneficial'] / total_states_selected
        print(f"  ---")
        print(f"  EFFICIENCY: {efficiency:.2%} beneficial / state selected")
    
    print(f"{'='*70}\n")
    
    # Placeholder for beneficial states (not computed in parallel version for simplicity)
    all_beneficial_states = []
    for seg in full_success_segments + improvement_segments:
        all_beneficial_states.append({
            'task_id': seg['task_id'],
            'recovery_step': seg['recovery_step'],
            'trajectory_length': seg['trajectory_length'],
            'original_reward': seg['original_reward'],
            'final_reward': seg['final_reward'],
            'is_full_success': seg['is_full_success'],
            'state': seg['state_observation'],
            'valid_actions': seg['state_valid_actions'],
            'num_valid_actions': len(seg['state_valid_actions']),
            'goal': seg['goal'][:500],
            'strategy': strategy,
            'true_entropy': seg.get('true_entropy', 0.0),
            'normalized_entropy': seg.get('normalized_entropy', 0.0),
        })
    
    return {
        'full_success_segments': full_success_segments,
        'improvement_segments': improvement_segments,
        'failure_segments': failure_segments,
        'all_beneficial_states': all_beneficial_states,
        'all_simulated_states': all_simulated_states,
        'simulation_stats': simulation_stats,
        'training_samples_success': training_samples_success,
        'training_samples_improvement': training_samples_improvement,
        'all_training_samples': all_training_samples,
        'stats': stats,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EEF Pipeline with Diagnosis Model (PARALLEL)")
    parser.add_argument("--failure_data", type=str, required=True,
                       help="Path to pre-collected failure trajectories")
    parser.add_argument("--strategy", type=str, default="baseline", 
                        choices=['baseline', 'entropy', 'stratified_entropy', 'diagnosis'],
                        help="State selection strategy")
    parser.add_argument("--M", type=int, default=5,
                       help="Number of states to select per trajectory")
    parser.add_argument("--simulation_budget", type=int, default=999999)
    parser.add_argument("--greedy", action='store_true', default=False,
                       help="Use greedy instead of softmax (default: softmax)")
    parser.add_argument("--num_attempts", type=int, default=6)
    parser.add_argument("--num_trajectories", type=int, default=None,
                       help="Limit number of trajectories to process")
    parser.add_argument("--model_path", type=str, 
                       default="./ckpts/web_click/epoch_9/model.pth",
                       help="Path to agent model")
    parser.add_argument("--output_dir", type=str, default="./eef_output")
    parser.add_argument("--verbose", action='store_true', default=False,
                       help="Verbose output (disabled by default for parallel)")
    
    # Parallel execution
    parser.add_argument("--num_workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    
    # Diagnosis model arguments
    parser.add_argument("--diagnosis_model_path", type=str, default=None,
                       help="Path to trained diagnosis model (required for strategy=diagnosis)")
    parser.add_argument("--diagnosis_base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="Base model for diagnosis model")
    parser.add_argument("--diagnosis_window", type=int, default=1,
                       help="Window around predicted step: simulate [pred-window, pred+window]")
    
    args = parser.parse_args()
    
    action_method = 'greedy' if args.greedy else 'softmax'
    
    print("="*70)
    print("EEF PIPELINE - PARALLEL VERSION WITH DIAGNOSIS MODEL SUPPORT")
    print("="*70)
    print(f"  Agent Model: {args.model_path}")
    print(f"  Strategy: {args.strategy}")
    if args.strategy == 'diagnosis':
        print(f"  Diagnosis Model: {args.diagnosis_model_path}")
        print(f"  Diagnosis Base: {args.diagnosis_base_model}")
        print(f"  Diagnosis Window: ±{args.diagnosis_window}")
    print(f"  M: {args.M}")
    print(f"  Budget: {args.simulation_budget}")
    print(f"  Action method: {action_method}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Num trajectories: {args.num_trajectories if args.num_trajectories else 'all'}")
    print("="*70)
    
    # Setup environment and agent (for state selection only)
    env = setup_environment()
    models = setup_model(args.model_path)
    agent = Agent(models)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load diagnosis model if needed
    diagnosis_model = None
    if args.strategy == 'diagnosis':
        if args.diagnosis_model_path is None:
            raise ValueError("--diagnosis_model_path required for strategy=diagnosis")
        diagnosis_model = DiagnosisModelSelector(
            args.diagnosis_model_path,
            args.diagnosis_base_model
        )
    
    # Load failures
    print(f"\nLoading failures from {args.failure_data}...")
    with open(args.failure_data, 'r') as f:
        failures = json.load(f)
    print(f"  Loaded {len(failures)} failures")
    
    if args.num_trajectories is not None:
        failures = failures[:args.num_trajectories]
        print(f"  Limited to {len(failures)} trajectories")
    
    # Run EEF (PARALLEL)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_eef_parallel(
        env, agent, failures,
        M=args.M, strategy=args.strategy,
        simulation_budget=args.simulation_budget,
        verbose=args.verbose,
        greedy=args.greedy,
        num_attempts=args.num_attempts,
        diagnosis_model=diagnosis_model,
        diagnosis_window=args.diagnosis_window,
        num_workers=args.num_workers,
        model_path=args.model_path,
    )
    
    # Save outputs
    prefix = f"{args.strategy}_{timestamp}"
    
    # 1. Full success segments
    path = os.path.join(args.output_dir, f"full_success_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['full_success_segments'], f, indent=2)
    print(f"✓ Saved {len(results['full_success_segments'])} full success segments")
    
    # 2. Improvement segments
    path = os.path.join(args.output_dir, f"improvement_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['improvement_segments'], f, indent=2)
    print(f"✓ Saved {len(results['improvement_segments'])} improvement segments")
    
    # 3. Failure segments
    path = os.path.join(args.output_dir, f"failure_segments_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['failure_segments'], f, indent=2)
    print(f"✓ Saved {len(results['failure_segments'])} failure segments")
    
    # 4. Simulation statistics
    path = os.path.join(args.output_dir, f"simulation_stats_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['simulation_stats'], f, indent=2)
    print(f"✓ Saved {len(results['simulation_stats'])} simulation statistics")
    
    # 5. All simulated states
    path = os.path.join(args.output_dir, f"all_simulated_states_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_simulated_states'], f, indent=2)
    print(f"✓ Saved {len(results['all_simulated_states'])} simulated states")
    
    # 6. Training samples
    path = os.path.join(args.output_dir, f"training_all_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['all_training_samples'], f, indent=2)
    print(f"✓ Saved {len(results['all_training_samples'])} training samples")
    
    # 7. Statistics
    path = os.path.join(args.output_dir, f"stats_{prefix}.json")
    with open(path, 'w') as f:
        json.dump(results['stats'], f, indent=2)
    print(f"✓ Saved statistics")
    
    print("\n" + "="*70)
    print("✓ EEF pipeline completed!")
    print("="*70)
    
    # Print comparison hint
    if args.strategy == 'diagnosis':
        print(f"\nTo compare with baseline, run:")
        print(f"  python {sys.argv[0]} --failure_data {args.failure_data} --strategy baseline --M 5 --num_workers 3")
        print(f"  python {sys.argv[0]} --failure_data {args.failure_data} --strategy entropy --M 5 --num_workers 3")


if __name__ == "__main__":
    main()