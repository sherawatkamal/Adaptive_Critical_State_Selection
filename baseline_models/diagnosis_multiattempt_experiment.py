#!/usr/bin/env python3
"""
Diagnosis Model + Multi-Attempt Experiment

Pipeline:
1. Load failed trajectories
2. Use trained diagnosis model to predict mistake step
3. Run N simulation attempts from predicted step (± window)
4. Measure success rate and compare to baselines

Run from baseline_models/ directory:
    python diagnosis_multiattempt_experiment.py \
        --failures ./simulation/failures.json \
        --diagnosis_model ./simulation/Qwen2.5/qwen25_instruct_v1 \
        --attempts 1,3,5,9 \
        --window 1
"""

import os
import sys
import json
import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# ENVIRONMENT AND MODEL SETUP (from multi_attempt_experiment.py)
# ============================================================================

def setup_environment(split='test'):
    """Setup WebShop environment - NO IMAGES"""
    print("Setting up WebShop environment...")
    
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    
    env_args = webenv_args()[0]
    sys.argv = original_argv
    
    env_args.get_image = 0
    env_args.human_goals = 1
    env_args.extra_search_path = ""
    
    env = WebEnv(env_args, split=split)
    print(f"✓ Environment ready (split={split})")
    return env


def setup_model(model_path="./ckpts/web_click/epoch_9/model.pth"):
    """Setup the IL model"""
    from train_choice_il import tokenizer, data_collator, process, process_goal
    from models.bert import BertModelForWebshop, BertConfigForWebshop
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}...")
    config = BertConfigForWebshop(image=False)
    model = BertModelForWebshop(config)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'data_collator': data_collator,
        'process': process,
        'process_goal': process_goal,
        'device': device,
    }


class Agent:
    """Agent wrapper with softmax exploration"""
    
    def __init__(self, models_dict):
        self.model = models_dict['model']
        self.tokenizer = models_dict['tokenizer']
        self.data_collator = models_dict['data_collator']
        self.process = models_dict['process']
        self.process_goal = models_dict['process_goal']
        self.device = models_dict['device']
    
    def get_action_probs(self, obs, valid_acts):
        if not valid_acts:
            return None
        
        if valid_acts[0].startswith('search['):
            return None
        
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
    
    def get_action(self, obs, info, method='softmax'):
        valid_acts = info.get('valid', [])
        
        if not valid_acts:
            return 'click[back to search]', {'type': 'fallback'}
        
        if valid_acts[0].startswith('search['):
            action = valid_acts[-1] if valid_acts else 'search[query]'
            return action, {'type': 'search'}
        
        probs = self.get_action_probs(obs, valid_acts)
        
        if probs is None:
            return valid_acts[0], {'type': 'error'}
        
        if method == 'greedy':
            idx = probs.argmax().item()
        else:  # softmax
            idx = torch.multinomial(probs, 1)[0].item()
        
        action = valid_acts[idx] if idx < len(valid_acts) else valid_acts[0]
        return action, {'type': 'choice', 'confidence': probs[idx].item()}


# ============================================================================
# DIAGNOSIS MODEL
# ============================================================================

def load_diagnosis_model(model_path: str, device: str = 'cuda'):
    """Load the trained diagnosis model."""
    print(f"Loading diagnosis model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    print(f"✓ Diagnosis model loaded on {device}")
    return model, tokenizer


def format_trajectory_for_diagnosis(trajectory: dict) -> str:
    """Format trajectory for the diagnosis model (same as training)."""
    goal = trajectory.get('goal', '')
    steps = trajectory.get('steps', [])
    
    lines = [f"Goal: {goal}", "", "Trajectory:"]
    for i, step in enumerate(steps):
        action = step.get('action_taken', step.get('action', ''))
        lines.append(f"Step {i}: {action}")
    
    return "\n".join(lines)


def predict_mistake_step(model, tokenizer, trajectory: dict, device: str = 'cuda') -> int:
    """Use diagnosis model to predict the mistake step."""
    traj_text = format_trajectory_for_diagnosis(trajectory)
    
    messages = [
        {"role": "system", "content": "You are an expert at analyzing failed web shopping trajectories. Given a trajectory, identify the step where the critical mistake occurred."},
        {"role": "user", "content": f"Analyze this failed trajectory and identify the step number where the mistake occurred:\n\n{traj_text}\n\nRespond with only the step number."}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    numbers = re.findall(r'\d+', response.strip())
    if numbers:
        return int(numbers[0])
    else:
        return 1  # Default to step 1 if parsing fails


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation_from_step(
    env,
    agent: Agent,
    trajectory: dict,
    start_step: int,
    max_steps: int = 50
) -> dict:
    """
    Run a simulation starting from a specific step in the trajectory.
    Uses softmax exploration (same as multi_attempt_experiment.py).
    """
    task_id = trajectory.get('task_id', '')
    steps = trajectory.get('steps', [])
    
    # Reset environment
    try:
        obs, info = env.reset(task_id)
    except Exception as e:
        return {'success': False, 'reward': 0, 'error': str(e)}
    
    # Replay trajectory up to start_step
    for i in range(min(start_step, len(steps))):
        action = steps[i].get('action_taken', steps[i].get('action', ''))
        if not action:
            return {'success': False, 'reward': 0, 'error': 'empty_action_in_replay'}
        
        try:
            obs, reward, done, info = env.step(action)
            if done:
                return {
                    'success': reward == 10.0,
                    'reward': reward * 10 if reward <= 1 else reward,
                    'steps_taken': i + 1,
                    'early_done': True
                }
        except Exception as e:
            return {'success': False, 'reward': 0, 'error': str(e)}
    
    # Simulate from start_step with softmax exploration
    sim_reward = 0
    sim_success = False
    
    for sim_step in range(max_steps):
        valid_acts = info.get('valid', [])
        if not valid_acts:
            break
        
        action, action_info = agent.get_action(obs, info, method='softmax')
        obs, reward, done, info = env.step(action)
        
        if done:
            sim_reward = reward * 10 if reward <= 1 else reward
            sim_success = (reward == 10.0) or (sim_reward >= 100)
            break
    
    return {
        'success': sim_success,
        'reward': sim_reward,
        'steps_taken': start_step + sim_step + 1,
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(
    failures: list,
    diagnosis_model,
    diagnosis_tokenizer,
    agent: Agent,
    env,
    num_attempts: list = [1, 3, 5, 9],
    window: int = 0,
    max_steps: int = 50,
    device: str = 'cuda'
) -> dict:
    """Run the full experiment: diagnosis + multi-attempt simulation."""
    
    max_attempts = max(num_attempts)
    
    results = {
        'config': {
            'num_failures': len(failures),
            'attempt_counts': num_attempts,
            'window': window,
            'max_steps': max_steps,
        },
        'per_trajectory': [],
        'summary': {}
    }
    
    # Track success at each attempt count
    success_by_attempts = {n: 0 for n in num_attempts}
    total_valid = 0
    
    # Track diagnosis predictions
    diagnosis_stats = {
        'total': 0,
        'predictions': defaultdict(int),
        'parse_failures': 0
    }
    
    for traj_idx, traj in enumerate(tqdm(failures, desc="Processing trajectories")):
        steps = traj.get('steps', [])
        if len(steps) < 2:
            continue
        
        total_valid += 1
        
        # Get diagnosis prediction
        try:
            predicted_step = predict_mistake_step(
                diagnosis_model, diagnosis_tokenizer, traj, device
            )
        except Exception as e:
            print(f"Warning: diagnosis failed for trajectory {traj_idx}: {e}")
            predicted_step = 1
            diagnosis_stats['parse_failures'] += 1
        
        # Clamp to valid range
        predicted_step = max(0, min(predicted_step, len(steps) - 1))
        diagnosis_stats['total'] += 1
        diagnosis_stats['predictions'][predicted_step] += 1
        
        # Determine steps to try (predicted ± window)
        steps_to_try = []
        for offset in range(-window, window + 1):
            step = predicted_step + offset
            if 0 <= step < len(steps):
                steps_to_try.append(step)
        steps_to_try = sorted(set(steps_to_try))
        
        # Track results for this trajectory
        traj_results = {
            'task_id': traj.get('task_id', traj_idx),
            'predicted_step': predicted_step,
            'steps_tried': steps_to_try,
            'trajectory_length': len(steps),
            'attempts': [],
            'first_success_attempt': None,
            'success_at': {n: False for n in num_attempts}
        }
        
        succeeded = False
        for attempt in range(max_attempts):
            if succeeded:
                traj_results['attempts'].append({
                    'attempt': attempt + 1,
                    'success': True,
                    'skipped': True
                })
                continue
            
            # Try each step in the window
            best_result = None
            for step in steps_to_try:
                result = run_simulation_from_step(
                    env, agent, traj, step,
                    max_steps=max_steps
                )
                
                if result['success']:
                    best_result = result
                    best_result['step_tried'] = step
                    break
                elif best_result is None or result.get('reward', 0) > best_result.get('reward', 0):
                    best_result = result
                    best_result['step_tried'] = step
            
            if best_result is None:
                best_result = {'success': False, 'reward': 0, 'step_tried': predicted_step}
            
            traj_results['attempts'].append({
                'attempt': attempt + 1,
                'success': best_result['success'],
                'reward': best_result.get('reward', 0),
                'step_tried': best_result.get('step_tried', predicted_step),
                'skipped': False
            })
            
            if best_result['success']:
                succeeded = True
                traj_results['first_success_attempt'] = attempt + 1
        
        # Update success counts
        for n in num_attempts:
            if any(a.get('success', False) for a in traj_results['attempts'][:n]):
                traj_results['success_at'][n] = True
                success_by_attempts[n] += 1
        
        results['per_trajectory'].append(traj_results)
        
        # Progress update
        if (traj_idx + 1) % 50 == 0:
            print(f"\n  Progress: {traj_idx + 1}/{len(failures)}")
            for n in num_attempts:
                rate = success_by_attempts[n] / total_valid * 100 if total_valid > 0 else 0
                print(f"    {n} attempts: {success_by_attempts[n]}/{total_valid} = {rate:.1f}%")
    
    # Summary
    results['summary'] = {
        'total_trajectories': total_valid,
        'success_rates': {
            n: {
                'count': success_by_attempts[n],
                'rate': success_by_attempts[n] / total_valid * 100 if total_valid > 0 else 0
            }
            for n in num_attempts
        },
        'diagnosis_stats': {
            'total_predictions': diagnosis_stats['total'],
            'parse_failures': diagnosis_stats['parse_failures'],
            'top_predictions': sorted(
                diagnosis_stats['predictions'].items(),
                key=lambda x: -x[1]
            )[:10]
        }
    }
    
    return results


def print_results(results: dict):
    """Pretty print results."""
    print("\n" + "="*70)
    print("DIAGNOSIS MODEL + MULTI-ATTEMPT EXPERIMENT RESULTS")
    print("="*70)
    
    config = results['config']
    summary = results['summary']
    
    print(f"\nConfiguration:")
    print(f"  Trajectories: {summary['total_trajectories']}")
    print(f"  Attempt counts tested: {config['attempt_counts']}")
    print(f"  Window: ±{config['window']} steps")
    
    print(f"\n" + "-"*70)
    print("SUCCESS RATES BY NUMBER OF ATTEMPTS")
    print("-"*70)
    
    for n in config['attempt_counts']:
        rate_info = summary['success_rates'][n]
        bar = "█" * int(rate_info['rate'] / 2)
        print(f"  {n} attempt(s): {rate_info['count']:3d}/{summary['total_trajectories']} = "
              f"{rate_info['rate']:5.1f}% {bar}")
    
    print(f"\n" + "-"*70)
    print("DIAGNOSIS MODEL PREDICTIONS")
    print("-"*70)
    print(f"  Total predictions: {summary['diagnosis_stats']['total_predictions']}")
    print(f"  Parse failures: {summary['diagnosis_stats']['parse_failures']}")
    print(f"  Top predicted steps: {summary['diagnosis_stats']['top_predictions'][:5]}")
    
    print(f"\n" + "-"*70)
    print("MARGINAL GAINS")
    print("-"*70)
    prev_rate = 0
    for n in config['attempt_counts']:
        rate = summary['success_rates'][n]['rate']
        gain = rate - prev_rate
        print(f"  {n} attempts: {rate:5.1f}% (+{gain:5.1f}% marginal)")
        prev_rate = rate
    
    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    
    attempt_counts = config['attempt_counts']
    rate_1 = summary['success_rates'][1]['rate']
    rate_max = summary['success_rates'][max(attempt_counts)]['rate']
    gain = rate_max - rate_1
    
    if gain > 15:
        print(f"  ✓ Multiple attempts help significantly! (+{gain:.1f}%)")
        print(f"  → Softmax exploration is valuable")
    elif gain > 5:
        print(f"  ~ Moderate benefit from multiple attempts (+{gain:.1f}%)")
    else:
        print(f"  ✗ Multiple attempts don't help much (+{gain:.1f}%)")
        print(f"  → Focus on improving diagnosis accuracy")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Diagnosis + Multi-Attempt Experiment')
    parser.add_argument('--failures', type=str, required=True,
                        help='Path to failed trajectories JSON')
    parser.add_argument('--diagnosis_model', type=str, required=True,
                        help='Path to trained diagnosis model')
    parser.add_argument('--agent_model', type=str, 
                        default='./ckpts/web_click/epoch_9/model.pth',
                        help='Path to BERT agent model')
    parser.add_argument('--attempts', type=str, default='1,3,5,9',
                        help='Comma-separated attempt counts to test')
    parser.add_argument('--window', type=int, default=1,
                        help='Window around predicted step (0=only predicted)')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Max steps per simulation')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Max trajectories to process (for testing)')
    parser.add_argument('--output_dir', type=str, default='./simulation/diagnosis_multiattempt_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    attempt_counts = [int(x) for x in args.attempts.split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load failures
    print(f"Loading failures from {args.failures}...")
    with open(args.failures) as f:
        failures = json.load(f)
    print(f"  Loaded {len(failures)} failed trajectories")
    
    if args.max_trajectories:
        failures = failures[:args.max_trajectories]
        print(f"  Limited to {len(failures)} trajectories")
    
    # Load diagnosis model
    diagnosis_model, diagnosis_tokenizer = load_diagnosis_model(
        args.diagnosis_model, args.device
    )
    
    # Setup environment and BERT agent
    env = setup_environment(split='test')
    models_dict = setup_model(args.agent_model)
    agent = Agent(models_dict)
    
    # Run experiment
    print(f"\nRunning experiment with {len(failures)} trajectories...")
    print(f"  Attempts: {attempt_counts}")
    print(f"  Window: ±{args.window}")
    
    results = run_experiment(
        failures=failures,
        diagnosis_model=diagnosis_model,
        diagnosis_tokenizer=diagnosis_tokenizer,
        agent=agent,
        env=env,
        num_attempts=attempt_counts,
        window=args.window,
        max_steps=args.max_steps,
        device=args.device
    )
    
    print_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"diagnosis_multiattempt_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()