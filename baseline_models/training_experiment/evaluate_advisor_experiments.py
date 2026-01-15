#!/usr/bin/env python3
"""
Evaluate 4 Models on Same Test Set

Compares:
- Train-1: All Recoverable (baseline)
- Train-2: Middle U Recoverable (quality hypothesis)
- Train-3: All States (with noise)
- Train-4: High/Low U Recoverable (control)

Key Questions:
1. Does Train-2 (smaller, curated) beat Train-1 (larger, all)?
2. Does Train-3 (with noise) hurt vs Train-1?
3. Does Train-4 (excluded data) perform worse?

Usage:
    python evaluate_advisor_experiments.py \
        --models_dir ./trained_models_advisor \
        --test_tasks 200 \
        --max_steps 15 \
        --output_dir ./eval_advisor_results
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm

# Add parent directory to path so we can import WebShop modules
sys.path.insert(0, '..')
sys.path.insert(0, '.')

from train_choice_il import tokenizer, process
from models.bert import BertModelForWebshop, BertConfigForWebshop
from train_rl import parse_args as webenv_args
from env import WebEnv


def setup_environment():
    """Setup WebShop environment"""
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    env_args = webenv_args()[0]
    sys.argv = original_argv
    
    env_args.get_image = 0
    env_args.human_goals = 0
    env = WebEnv(env_args, split='test')
    return env


def load_model(model_path, device):
    """Load trained model"""
    config = BertConfigForWebshop(image=False)
    model = BertModelForWebshop(config)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


def get_action(model, obs, valid_acts, tokenizer, device):
    """Get action from model (greedy)"""
    if not valid_acts:
        return 'click[back to search]'
    
    # Handle search page
    if valid_acts[0].startswith('search['):
        return valid_acts[-1]
    
    # Encode state
    state_encoding = tokenizer(
        process(obs), 
        max_length=512, 
        truncation=True, 
        padding='max_length'
    )
    
    # Encode actions
    action_encodings = tokenizer(
        [process(a) for a in valid_acts],
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    
    batch = {
        'state_input_ids': state_encoding['input_ids'],
        'state_attention_mask': state_encoding['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': [0.0] * 512,
        'labels': 0
    }
    
    from train_choice_il import data_collator
    batch = data_collator([batch])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits[0]
        idx = logits.argmax().item()
    
    return valid_acts[idx] if idx < len(valid_acts) else valid_acts[0]


def evaluate_model(model, env, n_tasks, max_steps, device):
    """Evaluate model on n_tasks"""
    successes = 0
    total_reward = 0
    
    for task_idx in tqdm(range(n_tasks), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            valid_acts = info.get('valid', [])
            action = get_action(model, obs, valid_acts, tokenizer, device)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        if reward == 10.0:  # Success
            successes += 1
        
        total_reward += reward * 10
    
    success_rate = successes / n_tasks
    avg_reward = total_reward / n_tasks
    
    return {
        'successes': successes,
        'total_tasks': n_tasks,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4 advisor experiment models")
    parser.add_argument("--models_dir", required=True, help="Directory with trained models")
    parser.add_argument("--test_tasks", type=int, default=200, help="Number of test tasks")
    parser.add_argument("--max_steps", type=int, default=15, help="Max steps per episode")
    parser.add_argument("--output_dir", default="./eval_advisor_results", help="Output directory")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("EVALUATING 4 MODELS FOR ADVISOR'S DATA SELECTION EXPERIMENT")
    print("="*70)
    print(f"Models directory: {args.models_dir}")
    print(f"Test tasks: {args.test_tasks}")
    print(f"Max steps: {args.max_steps}")
    print(f"Device: {device}")
    print()
    
    # Setup environment
    print("Setting up environment...")
    env = setup_environment()
    print("✓ Environment ready")
    print()
    
    # Define models to evaluate
    models_to_eval = [
        ("train1_all_recoverable_model.pth", "Train-1: All Recoverable"),
        ("train2_middle_u_recoverable_model.pth", "Train-2: Middle U Recoverable"),
        ("train3_all_states_model.pth", "Train-3: All States"),
        ("train4_high_low_u_recoverable_model.pth", "Train-4: High/Low U Recoverable"),
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each model
    results = []
    
    for model_file, description in models_to_eval:
        model_path = os.path.join(args.models_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"⚠️  WARNING: {model_path} not found, skipping...")
            continue
        
        print("="*70)
        print(f"EVALUATING: {description}")
        print("="*70)
        print(f"Model: {model_path}")
        
        # Load model
        print("Loading model...")
        model = load_model(model_path, device)
        print("✓ Model loaded")
        
        # Evaluate
        print(f"Running {args.test_tasks} test episodes...")
        eval_results = evaluate_model(model, env, args.test_tasks, args.max_steps, device)
        
        print(f"\n✓ Evaluation complete!")
        print(f"  Success rate: {eval_results['success_rate']:.1%} ({eval_results['successes']}/{eval_results['total_tasks']})")
        print(f"  Avg reward: {eval_results['avg_reward']:.1f}")
        print()
        
        # Store results
        result_entry = {
            'model_name': model_file.replace('_model.pth', ''),
            'description': description,
            'model_path': model_path,
            **eval_results,
        }
        results.append(result_entry)
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {results_path}")
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Model':<45} {'Success Rate':>15} {'Avg Reward':>10}")
    print("-"*70)
    
    for result in results:
        print(f"{result['description']:<45} {result['success_rate']:>14.1%} {result['avg_reward']:>10.1f}")
    
    # Key comparisons
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    train1 = next((r for r in results if 'train1' in r['model_name']), None)
    train2 = next((r for r in results if 'train2' in r['model_name']), None)
    train3 = next((r for r in results if 'train3' in r['model_name']), None)
    train4 = next((r for r in results if 'train4' in r['model_name']), None)
    
    if train1 and train2:
        delta = train2['success_rate'] - train1['success_rate']
        print(f"\n1. Middle U (Train-2) vs All Recoverable (Train-1):")
        print(f"   Train-2: {train2['success_rate']:.1%}")
        print(f"   Train-1: {train1['success_rate']:.1%}")
        print(f"   Δ = {delta:+.1%}")
        if delta > 0.02:
            print(f"   ✓ WINNER: Train-2 (Middle U) - Quality > Quantity!")
        elif delta < -0.02:
            print(f"   ✗ LOSER: Train-2 - More data wins")
        else:
            print(f"   ≈ TIED: Similar performance")
    
    if train1 and train3:
        delta = train3['success_rate'] - train1['success_rate']
        print(f"\n2. All States (Train-3) vs All Recoverable (Train-1):")
        print(f"   Train-3: {train3['success_rate']:.1%}")
        print(f"   Train-1: {train1['success_rate']:.1%}")
        print(f"   Δ = {delta:+.1%}")
        if delta < -0.02:
            print(f"   ✗ Including unrecoverable states HURTS performance")
        elif delta > 0.02:
            print(f"   ✓ Including unrecoverable states HELPS (surprising!)")
        else:
            print(f"   ≈ Unrecoverable states have NO EFFECT")
    
    if train2 and train4:
        delta = train2['success_rate'] - train4['success_rate']
        print(f"\n3. Middle U (Train-2) vs High/Low U (Train-4):")
        print(f"   Train-2: {train2['success_rate']:.1%}")
        print(f"   Train-4: {train4['success_rate']:.1%}")
        print(f"   Δ = {delta:+.1%}")
        if delta > 0.02:
            print(f"   ✓ Middle U is better than extremes")
        else:
            print(f"   ≈ No clear advantage for middle U")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()