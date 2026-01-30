#!/usr/bin/env python3
"""
Multi-Attempt Experiment on LABELED Mistake Steps

Uses recovery_step from success_segments as ground truth.
Runs multiple simulations at that step to measure success rate vs attempts.

Run from baseline_models/ directory:
    python multi_attempt_experiment.py \
        --failures ./simulation/failures.json \
        --success_segments ./simulation/Qwen2.5/full_success_segments_stratified_entropy_20260114_224102.json \
        --attempts 1,3,5,9
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# ENVIRONMENT AND MODEL SETUP (copied from your working script)
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
    print("✓ Environment loaded")
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
# DATA LOADING
# ============================================================================

def create_labeled_dataset(failures_path, success_segments_path):
    """
    Create labeled dataset by matching failures with successful recovery steps.
    recovery_step = the step where intervention led to success = mistake step
    """
    print("Creating labeled dataset...")
    
    with open(failures_path) as f:
        failures = json.load(f)
    print(f"  Loaded {len(failures)} failures")
    
    with open(success_segments_path) as f:
        successes = json.load(f)
    print(f"  Loaded {len(successes)} success segments")
    
    # Map task_id -> failure trajectory
    failure_by_task = {f['task_id']: f for f in failures}
    
    # Create labeled data
    labeled_data = []
    
    for seg in successes:
        task_id = seg['task_id']
        
        if task_id not in failure_by_task:
            continue
        
        labeled_data.append({
            'task_id': task_id,
            'mistake_step': seg['recovery_step'],
            'trajectory': failure_by_task[task_id],
            'original_reward': seg['original_reward'],
            'successful_final_reward': seg['final_reward'],
            'entropy_at_step': seg.get('true_entropy', 0),
        })
    
    print(f"  Created {len(labeled_data)} labeled examples")
    
    # Show distribution
    step_dist = Counter(d['mistake_step'] for d in labeled_data)
    print(f"  Mistake step distribution: {step_dist.most_common(10)}")
    
    return labeled_data


# ============================================================================
# MULTI-ATTEMPT EXPERIMENT
# ============================================================================

def run_multi_attempt_experiment(env, agent, labeled_data, 
                                  attempt_counts=[1, 3, 5, 9],
                                  max_steps=50, verbose=True):
    """
    Run multiple simulation attempts at each labeled mistake step.
    """
    
    max_attempts = max(attempt_counts)
    
    results = {
        'config': {
            'attempt_counts': attempt_counts,
            'max_steps': max_steps,
            'total_examples': len(labeled_data),
        },
        'per_example': [],
        'summary': {},
    }
    
    successes_by_attempts = {n: 0 for n in attempt_counts}
    total_simulations = 0
    
    print("\n" + "="*70)
    print("MULTI-ATTEMPT EXPERIMENT ON LABELED MISTAKE STEPS")
    print("="*70)
    print(f"  Examples: {len(labeled_data)}")
    print(f"  Attempt counts: {attempt_counts}")
    print(f"  Max attempts per example: {max_attempts}")
    print("="*70)
    
    for idx, item in enumerate(labeled_data):
        task_id = item['task_id']
        mistake_step = item['mistake_step']
        trajectory = item['trajectory']
        steps = trajectory['steps']
        original_reward = item['original_reward']
        
        if verbose and idx % 100 == 0:
            print(f"\nProcessing {idx+1}/{len(labeled_data)}...")
        
        # Run max_attempts simulations
        attempt_results = []
        first_success_attempt = None
        
        for attempt in range(max_attempts):
            # Reset environment
            obs, info = env.reset(task_id)
            
            # Replay actions up to mistake_step
            replay_ok = True
            for step_idx in range(mistake_step):
                if step_idx >= len(steps):
                    replay_ok = False
                    break
                
                action = steps[step_idx].get('action_taken', '')
                if not action:
                    replay_ok = False
                    break
                
                obs, reward, done, info = env.step(action)
                if done:
                    replay_ok = False
                    break
            
            if not replay_ok:
                attempt_results.append({
                    'success': False, 
                    'reward': 0, 
                    'reason': 'replay_failed'
                })
                total_simulations += 1
                continue
            
            # Simulate from mistake_step with softmax exploration
            sim_success = False
            sim_reward = 0
            
            for sim_step in range(max_steps):
                valid_acts = info.get('valid', [])
                if not valid_acts:
                    break
                
                action, action_info = agent.get_action(obs, info, method='softmax')
                obs, reward, done, info = env.step(action)
                
                if done:
                    sim_reward = reward * 10
                    sim_success = (reward == 10.0)
                    break
            
            attempt_results.append({
                'success': sim_success,
                'reward': sim_reward,
            })
            total_simulations += 1
            
            if sim_success and first_success_attempt is None:
                first_success_attempt = attempt + 1
        
        # Record results
        example_result = {
            'task_id': task_id,
            'mistake_step': mistake_step,
            'original_reward': original_reward,
            'entropy': item.get('entropy_at_step', 0),
            'first_success_attempt': first_success_attempt,
            'all_rewards': [r['reward'] for r in attempt_results],
            'best_reward': max((r['reward'] for r in attempt_results), default=0),
            'success_by_attempts': {},
        }
        
        for n in attempt_counts:
            succeeded = any(
                attempt_results[i]['success'] 
                for i in range(min(n, len(attempt_results)))
            )
            example_result['success_by_attempts'][n] = succeeded
            if succeeded:
                successes_by_attempts[n] += 1
        
        results['per_example'].append(example_result)
        
        if verbose and first_success_attempt and idx < 50:
            print(f"  Task {task_id} step {mistake_step}: ✓ Success on attempt {first_success_attempt}")
    
    # Compute summary
    total = len(results['per_example'])
    
    print("\n" + "="*70)
    print("RESULTS: Success Rate by Number of Attempts")
    print("="*70)
    
    for n in attempt_counts:
        rate = successes_by_attempts[n] / total * 100 if total > 0 else 0
        results['summary'][f'success_rate_{n}_attempts'] = rate
        bar = "█" * int(rate / 2)
        print(f"  {n} attempt(s): {successes_by_attempts[n]:3d}/{total} = {rate:5.1f}% {bar}")
    
    # First success distribution
    print("\n" + "-"*70)
    print("First Success Attempt Distribution:")
    
    first_success_dist = Counter(
        e['first_success_attempt'] 
        for e in results['per_example'] 
        if e['first_success_attempt'] is not None
    )
    
    for attempt_num in sorted(first_success_dist.keys()):
        count = first_success_dist[attempt_num]
        print(f"  Attempt {attempt_num}: {count} examples")
    
    never_succeeded = sum(
        1 for e in results['per_example'] 
        if e['first_success_attempt'] is None
    )
    print(f"  Never succeeded: {never_succeeded} ({never_succeeded/total*100:.1f}%)")
    
    # Marginal gains
    print("\n" + "-"*70)
    print("Marginal Gain Analysis:")
    
    prev_rate = 0
    for n in attempt_counts:
        rate = successes_by_attempts[n] / total * 100
        marginal = rate - prev_rate
        print(f"  {n} attempts: {rate:5.1f}% (+{marginal:4.1f}% marginal)")
        prev_rate = rate
    
    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    
    rate_1 = successes_by_attempts[1] / total * 100
    rate_max = successes_by_attempts[max(attempt_counts)] / total * 100
    gain = rate_max - rate_1
    
    if gain > 15:
        print(f"  ✓ Multiple attempts help significantly! (+{gain:.1f}%)")
        print(f"  → Softmax exploration is valuable")
        print(f"  → Run {attempt_counts[-1]} attempts for best results")
    elif gain > 5:
        print(f"  ~ Moderate benefit from multiple attempts (+{gain:.1f}%)")
        print(f"  → 3-5 attempts is a good balance")
    else:
        print(f"  ✗ Multiple attempts don't help much (+{gain:.1f}%)")
        print(f"  → 1 attempt is sufficient")
        print(f"  → Focus on step selection instead")
    
    print("="*70)
    
    results['summary']['total_simulations'] = total_simulations
    results['summary']['never_succeeded_count'] = never_succeeded
    results['summary']['never_succeeded_rate'] = never_succeeded / total * 100
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", default="./simulation/failures.json")
    parser.add_argument("--success_segments", required=True,
                        help="Path to full_success_segments JSON")
    parser.add_argument("--model_path", default="./ckpts/web_click/epoch_9/model.pth")
    parser.add_argument("--attempts", default="1,3,5,9",
                        help="Comma-separated attempt counts")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default="./simulation/multi_attempt_results")
    parser.add_argument("--verbose", action='store_true', default=True)
    args = parser.parse_args()
    
    attempt_counts = [int(x) for x in args.attempts.split(',')]
    
    # Setup
    env = setup_environment()
    models = setup_model(args.model_path)
    agent = Agent(models)
    
    # Create labeled dataset
    labeled_data = create_labeled_dataset(args.failures, args.success_segments)
    
    if args.max_examples:
        labeled_data = labeled_data[:args.max_examples]
        print(f"Limited to {len(labeled_data)} examples")
    
    # Run experiment
    results = run_multi_attempt_experiment(
        env, agent, labeled_data,
        attempt_counts=attempt_counts,
        verbose=args.verbose
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = os.path.join(args.output_dir, f"multi_attempt_results_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path}")


if __name__ == "__main__":
    main()