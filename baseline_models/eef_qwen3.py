#!/usr/bin/env python3
"""
EEF Pipeline - Using Qwen3-8B (Same as trajectory collection)

This version uses the Qwen3-8B model directly (same as collect_trajectories_qwen8B.py)
instead of the IL-trained BERT model.

Usage:
    python eef_qwen3.py --failure_data failures.json --strategy baseline --workers 1
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import re

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment(split='test'):
    """Setup WebShop environment"""
    print("Setting up WebShop environment...")
    
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    
    env_args = webenv_args()[0]
    sys.argv = original_argv
    
    env_args.get_image = 0
    env_args.human_goals = 0
    env_args.extra_search_path = ""
    
    env = WebEnv(env_args, split=split)
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    
    print("✓ Environment loaded")
    return env


# ============================================================================
# QWEN3-8B AGENT (Same as trajectory collection)
# ============================================================================

SYSTEM_PROMPT = """You are an expert online shopping assistant. You help users find and purchase products that match their requirements.

You are interacting with a WebShop environment. At each step, you'll see:
1. The user's shopping goal/instruction
2. The current webpage observation
3. Available actions you can take

Your task is to select the BEST action to help complete the shopping task.

Rules:
- For search pages: Use search[query] to search for products
- For product listings: Click on a product ID (e.g., click[B07XYZ123]) to view details
- For product pages: Click on options (size, color) or click[Buy Now] to purchase
- Use click[Next >] to see more products if current ones don't match
- Use click[Back to Search] to try a different search
- Use click[< Prev] to go back to previous page

Respond with ONLY the action in the exact format: action[argument]"""


def parse_action(response_text, valid_actions):
    """Extract action from model response."""
    text = response_text.strip()
    
    # Try to find action pattern
    patterns = [
        r'(search\[.+?\])',
        r'(click\[.+?\])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            action = match.group(1)
            if action.lower().startswith('click'):
                action = 'click' + action[5:]
            return action
    
    # If no pattern found, try to match with valid actions
    text_lower = text.lower()
    for valid in valid_actions:
        if valid.lower() in text_lower:
            return valid
    
    # Default: return first valid action
    return valid_actions[0] if valid_actions else text


def format_prompt(obs, goal, valid_actions, tokenizer, no_think=True):
    """Format prompt for Qwen3-8B."""
    no_think_prefix = "/no_think\n" if no_think else ""
    
    user_content = f"""{no_think_prefix}SHOPPING GOAL: {goal}

CURRENT PAGE:
{obs[:2000]}

AVAILABLE ACTIONS:
{chr(10).join(f'- {a}' for a in valid_actions[:20])}

Select the best action to complete the shopping goal. Respond with ONLY the action."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant:"
    
    return prompt


class Qwen3Agent:
    """Qwen3-8B Agent for EEF simulation"""
    
    def __init__(self, model_name="Qwen/Qwen3-8B", device_id=0, no_think=True):
        self.model_name = model_name
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.no_think = no_think
        
        print(f"Loading Qwen3-8B model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()
        
        print(f"✓ Qwen3-8B loaded on {self.device}")
    
    @torch.no_grad()
    def get_action_probs(self, obs: str, valid_acts: List[str], goal: str) -> Optional[torch.Tensor]:
        """
        Get action probabilities by scoring each valid action.
        
        For Qwen3-8B, we generate the action and compute perplexity for each option.
        """
        if not valid_acts:
            return None
        
        # Skip search states (can't score generative actions reliably)
        if valid_acts[0].startswith('search['):
            return None
        
        # Score each action by computing log probability
        scores = []
        for action in valid_acts:
            prompt = format_prompt(obs, goal, [action], self.tokenizer, no_think=self.no_think)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Add the action as target
            action_tokens = self.tokenizer(action, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
            # Compute log probability of this action
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits
            
            # Simple scoring: just use a constant for now (uniform)
            # Proper scoring would require computing actual token-level probabilities
            scores.append(1.0)
        
        # Convert to probabilities (uniform for now - true scoring is complex for generative models)
        probs = torch.tensor(scores, dtype=torch.float32)
        probs = probs / probs.sum()
        
        return probs
    
    def compute_true_entropy(self, obs: str, valid_acts: List[str], goal: str) -> Tuple[float, float, float]:
        """
        Compute entropy over valid actions.
        
        For Qwen3-8B, this is approximate since we can't get true action distribution easily.
        """
        probs = self.get_action_probs(obs, valid_acts, goal)
        
        if probs is None:
            return 0.0, 0.0, 1.0
        
        # Compute entropy
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * torch.log(probs_clamped)).sum().item()
        
        # Normalized entropy
        n_actions = len(valid_acts)
        max_entropy = np.log(n_actions) if n_actions > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Max probability
        max_prob = probs.max().item()
        
        return entropy, normalized_entropy, max_prob
    
    @torch.no_grad()
    def get_action(self, obs: str, info: dict, goal: str, method='softmax') -> Tuple[str, dict]:
        """Get action from Qwen3-8B model."""
        valid_acts = info.get('valid', [])
        
        if not valid_acts:
            return 'click[back to search]', {'type': 'fallback'}
        
        # Format prompt
        prompt = format_prompt(obs, goal, valid_acts, self.tokenizer, no_think=self.no_think)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs['input_ids'].shape[1]
        
        # Generate
        temperature = 0.5 if method == 'softmax' else 0.0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only new tokens
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Parse action
        action = parse_action(response, valid_acts)
        
        return action, {
            'type': 'generation',
            'raw_response': response[:200],
            'entropy': 0.0,  # Would need proper computation
            'normalized_entropy': 0.0,
        }


# ============================================================================
# REST OF EEF PIPELINE (Same as enhanced version)
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
            'recoveries': 0,
        }
    
    def simulate_from_state(self, task_id: int, target_step: int, 
                           trajectory: Dict, goal: str, method='softmax') -> Tuple[bool, float, List[Dict]]:
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
            
            action = steps[step_idx].get('action_taken', steps[step_idx].get('action', ''))
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
            
            action, action_info = self.agent.get_action(obs, info, goal, method=method)
            
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


def select_critical_states_baseline(trajectory: Dict, M: int = 5) -> Tuple[List[int], List[Dict]]:
    """Baseline: Equal-interval skip-length selection"""
    steps = trajectory.get('steps', [])
    T = len(steps)
    if T <= 1:
        return [], []
    l = max(1, T // (M + 1))
    indices = [m * l for m in range(1, M + 1) if m * l < T]
    
    entropy_info = [{'state_idx': idx, 'true_entropy': 0.0, 'method': 'baseline'} for idx in indices]
    return indices, entropy_info


def process_trajectory_simple(traj_data, config):
    """Process single trajectory (simplified for Qwen3-8B)."""
    trajectory = traj_data['trajectory']
    traj_idx = traj_data['idx']
    
    # Setup for this trajectory
    env = setup_environment(split=config['split'])
    agent = Qwen3Agent(model_name=config['model_name'], device_id=config['device_id'], no_think=config['no_think'])
    simulator = EEFSimulator(env, agent, debug=config['verbose'])
    
    # Handle both formats
    task_id = trajectory.get('task_id', trajectory.get('idx', 0))
    original_reward = trajectory.get('reward', trajectory.get('final_reward', 0))
    goal = trajectory.get('goal', '')
    traj_length = len(trajectory.get('steps', []))
    
    # Select states (baseline only for Qwen3-8B)
    critical_states, selection_info = select_critical_states_baseline(trajectory, config['M'])
    
    if not critical_states:
        return {'traj_idx': traj_idx, 'segments': [], 'states': []}
    
    segments = []
    states = []
    
    for state_idx in critical_states:
        steps = trajectory.get('steps', [])
        state_obs = steps[state_idx].get('observation', '') if state_idx < len(steps) else ''
        state_valid_actions = steps[state_idx].get('valid_actions', []) if state_idx < len(steps) else []
        
        best_reward = -1
        best_traj = None
        best_success = False
        
        for attempt in range(config['num_attempts']):
            success, reward, sim_traj = simulator.simulate_from_state(
                task_id, state_idx, trajectory, goal, method=config['action_method']
            )
            
            if reward > best_reward:
                best_reward = reward
                best_traj = sim_traj
                best_success = success
            
            if success:
                break
        
        # Compute recoverability
        if original_reward < 100:
            recoverability_score = (best_reward - original_reward) / (100 - original_reward)
        else:
            recoverability_score = 0.0
        recoverability_score = max(0.0, min(1.0, recoverability_score))
        
        state_info = {
            'task_id': task_id,
            'recovery_step': state_idx,
            'original_reward': original_reward,
            'final_reward': best_reward,
            'recoverability_score': recoverability_score,
            'recovery_type': 'full' if best_success else ('partial' if best_reward > original_reward + 10 else 'none'),
            'normalized_entropy': 0.0,  # Not computed for Qwen3-8B
        }
        states.append(state_info)
        
        if best_success or (best_reward > original_reward + 10):
            new_actions = [s for s in best_traj if not s.get('is_replay', False)]
            segments.append({
                'task_id': task_id,
                'goal': goal,
                'recovery_step': state_idx,
                'final_reward': best_reward,
                'steps': new_actions,
            })
    
    return {'traj_idx': traj_idx, 'segments': segments, 'states': states}


def main():
    parser = argparse.ArgumentParser(description="EEF Pipeline with Qwen3-8B")
    parser.add_argument("--failure_data", type=str, required=True)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--num_trajectories", type=int, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./eef_qwen3_output")
    parser.add_argument("--no_think", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print("="*70)
    print("EEF PIPELINE - QWEN3-8B")
    print("="*70)
    print(f"  Model: {args.model_name}")
    print(f"  M: {args.M}")
    print(f"  Num trajectories: {args.num_trajectories or 'all'}")
    print("="*70)
    
    # Load failures
    print(f"\nLoading failures from {args.failure_data}...")
    with open(args.failure_data, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'trajectories' in data:
            failures = data['trajectories']
        elif 'failed_trajectories' in data:
            failures = data['failed_trajectories']
        else:
            failures = [data]
    elif isinstance(data, list):
        failures = data
    
    print(f"  Loaded {len(failures)} failures")
    
    if args.num_trajectories:
        failures = failures[:args.num_trajectories]
        print(f"  Limited to {len(failures)} trajectories")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process trajectories
    config = {
        'M': args.M,
        'num_attempts': args.num_attempts,
        'model_name': args.model_name,
        'split': args.split,
        'device_id': 0,
        'no_think': args.no_think,
        'action_method': 'softmax',
        'verbose': args.verbose,
    }
    
    all_segments = []
    all_states = []
    
    for idx, traj in enumerate(failures):
        print(f"\nProcessing trajectory {idx+1}/{len(failures)}...")
        result = process_trajectory_simple({'trajectory': traj, 'idx': idx}, config)
        all_segments.extend(result['segments'])
        all_states.extend(result['states'])
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    path = os.path.join(args.output_dir, f"all_simulated_states_{timestamp}.json")
    with open(path, 'w') as f:
        json.dump(all_states, f, indent=2)
    print(f"\n✓ Saved {len(all_states)} states to {path}")
    
    path = os.path.join(args.output_dir, f"segments_{timestamp}.json")
    with open(path, 'w') as f:
        json.dump(all_segments, f, indent=2)
    print(f"✓ Saved {len(all_segments)} segments to {path}")
    
    print("\n" + "="*70)
    print("✓ EEF pipeline completed!")
    print("="*70)


if __name__ == "__main__":
    main()