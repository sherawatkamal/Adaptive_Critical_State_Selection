#!/usr/bin/env python3
"""
Create 4 Training Datasets for Advisor's Experiment

Splits:
1. All Recoverable (improvement > 0)
2. Middle U Recoverable (0.4 ≤ U ≤ 0.7 AND improvement > 0)
3. All States (recoverable + unrecoverable)
4. High/Low U Recoverable (U < 0.4 OR U > 0.7 AND improvement > 0)

Usage:
    python create_training_splits.py \
        --simulation_stats ./eef_500_stratified/simulation_stats_*.json \
        --success_segments ./eef_500_stratified/full_success_segments_*.json \
        --improvement_segments ./eef_500_stratified/improvement_segments_*.json \
        --failure_segments ./eef_500_stratified/failure_segments_*.json \
        --output_dir ./training_splits
"""

import json
import argparse
import os
from collections import defaultdict
import pandas as pd


def load_json(path):
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def extract_training_examples(segments, source_label):
    """
    Extract (state, goal, action) tuples from segment files.
    
    segments: List of segment dicts with 'steps' field
    source_label: 'success', 'improvement', or 'failure'
    
    Returns: List of training examples
    """
    examples = []
    
    for seg in segments:
        task_id = seg['task_id']
        goal = seg['goal']
        recovery_step = seg['recovery_step']
        true_entropy = seg.get('true_entropy', 0.0)
        normalized_entropy = seg.get('normalized_entropy', 0.0)
        final_reward = seg.get('final_reward', 0)
        original_reward = seg.get('original_reward', 0)
        improvement = final_reward - original_reward
        
        # Extract steps (non-replay actions)
        steps = seg.get('steps', [])
        
        for step in steps:
            obs = step.get('observation', '')
            action = step.get('action_taken', '')
            
            if not obs or not action:
                continue
            
            example = {
                'state': obs,
                'goal': goal,
                'action': action,
                'task_id': task_id,
                'recovery_step': recovery_step,
                'source': source_label,
                'true_entropy': true_entropy,
                'normalized_entropy': normalized_entropy,
                'final_reward': final_reward,
                'original_reward': original_reward,
                'improvement': improvement,
                'is_recoverable': improvement > 0,
            }
            
            examples.append(example)
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Create 4 training splits for advisor experiment")
    parser.add_argument("--simulation_stats", required=True, help="Path to simulation_stats JSON")
    parser.add_argument("--success_segments", required=True, help="Path to full_success_segments JSON")
    parser.add_argument("--improvement_segments", required=True, help="Path to improvement_segments JSON")
    parser.add_argument("--failure_segments", required=True, help="Path to failure_segments JSON")
    parser.add_argument("--output_dir", default="./training_splits", help="Output directory")
    parser.add_argument("--middle_u_min", type=float, default=0.4, help="Middle U lower bound")
    parser.add_argument("--middle_u_max", type=float, default=0.7, help="Middle U upper bound")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("CREATING 4 TRAINING SPLITS FOR ADVISOR EXPERIMENT")
    print("="*70)
    print(f"Middle U range: [{args.middle_u_min}, {args.middle_u_max}]")
    print()
    
    # Load all segment files
    print("Loading segment files...")
    success_segments = load_json(args.success_segments)
    improvement_segments = load_json(args.improvement_segments)
    failure_segments = load_json(args.failure_segments)
    
    print(f"  Success segments: {len(success_segments)}")
    print(f"  Improvement segments: {len(improvement_segments)}")
    print(f"  Failure segments: {len(failure_segments)}")
    print()
    
    # Extract training examples from each source
    print("Extracting training examples...")
    success_examples = extract_training_examples(success_segments, 'success')
    improvement_examples = extract_training_examples(improvement_segments, 'improvement')
    failure_examples = extract_training_examples(failure_segments, 'failure')
    
    print(f"  Success examples: {len(success_examples)}")
    print(f"  Improvement examples: {len(improvement_examples)}")
    print(f"  Failure examples: {len(failure_examples)}")
    print()
    
    # Combine all examples
    all_examples = success_examples + improvement_examples + failure_examples
    
    # Create 4 training splits
    print("Creating training splits...")
    print("-"*70)
    
    # Split 1: All Recoverable (improvement > 0)
    train1_all_recoverable = [ex for ex in all_examples if ex['is_recoverable']]
    
    # Split 2: Middle U Recoverable (0.4 ≤ U ≤ 0.7 AND improvement > 0)
    train2_middle_u = [
        ex for ex in all_examples 
        if ex['is_recoverable'] 
        and args.middle_u_min <= ex['normalized_entropy'] <= args.middle_u_max
    ]
    
    # Split 3: All States (recoverable + unrecoverable)
    train3_all_states = all_examples
    
    # Split 4: High/Low U Recoverable (U < 0.4 OR U > 0.7 AND improvement > 0)
    train4_high_low_u = [
        ex for ex in all_examples 
        if ex['is_recoverable'] 
        and (ex['normalized_entropy'] < args.middle_u_min or ex['normalized_entropy'] > args.middle_u_max)
    ]
    
    # Print statistics
    print("\n" + "="*70)
    print("TRAINING SPLIT STATISTICS")
    print("="*70)
    
    splits = [
        ("Train-1: All Recoverable", train1_all_recoverable),
        ("Train-2: Middle U Recoverable", train2_middle_u),
        ("Train-3: All States", train3_all_states),
        ("Train-4: High/Low U Recoverable", train4_high_low_u),
    ]
    
    for name, data in splits:
        recoverable_count = sum(1 for ex in data if ex['is_recoverable'])
        unrecoverable_count = len(data) - recoverable_count
        
        if data:
            entropies = [ex['normalized_entropy'] for ex in data]
            mean_u = sum(entropies) / len(entropies)
            min_u = min(entropies)
            max_u = max(entropies)
        else:
            mean_u = min_u = max_u = 0
        
        print(f"\n{name}:")
        print(f"  Total examples: {len(data)}")
        print(f"  Recoverable: {recoverable_count}")
        print(f"  Unrecoverable: {unrecoverable_count}")
        print(f"  Entropy range: [{min_u:.3f}, {max_u:.3f}], mean: {mean_u:.3f}")
    
    # Compare Train-2 vs Train-1
    print("\n" + "="*70)
    print("KEY COMPARISON: Train-2 (Middle U) vs Train-1 (All Recoverable)")
    print("="*70)
    print(f"Middle U size: {len(train2_middle_u)} ({len(train2_middle_u)/len(train1_all_recoverable)*100:.1f}% of All Recoverable)")
    print(f"Excluded from Middle U: {len(train1_all_recoverable) - len(train2_middle_u)}")
    print(f"  - Low U (stubborn): {len([ex for ex in train1_all_recoverable if ex['normalized_entropy'] < args.middle_u_min])}")
    print(f"  - High U (lucky): {len([ex for ex in train1_all_recoverable if ex['normalized_entropy'] > args.middle_u_max])}")
    
    # Save splits to disk
    print("\n" + "="*70)
    print("SAVING TRAINING SPLITS")
    print("="*70)
    
    split_files = [
        ("train1_all_recoverable.json", train1_all_recoverable),
        ("train2_middle_u_recoverable.json", train2_middle_u),
        ("train3_all_states.json", train3_all_states),
        ("train4_high_low_u_recoverable.json", train4_high_low_u),
    ]
    
    for filename, data in split_files:
        path = os.path.join(args.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved: {path} ({len(data)} examples)")
    
    # Also save summary statistics
    summary = {
        'middle_u_range': [args.middle_u_min, args.middle_u_max],
        'splits': {
            'train1_all_recoverable': {
                'count': len(train1_all_recoverable),
                'recoverable': sum(1 for ex in train1_all_recoverable if ex['is_recoverable']),
            },
            'train2_middle_u_recoverable': {
                'count': len(train2_middle_u),
                'recoverable': sum(1 for ex in train2_middle_u if ex['is_recoverable']),
            },
            'train3_all_states': {
                'count': len(train3_all_states),
                'recoverable': sum(1 for ex in train3_all_states if ex['is_recoverable']),
            },
            'train4_high_low_u_recoverable': {
                'count': len(train4_high_low_u),
                'recoverable': sum(1 for ex in train4_high_low_u if ex['is_recoverable']),
            },
        }
    }
    
    summary_path = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {summary_path}")
    
    print("\n" + "="*70)
    print("✓ TRAINING SPLITS CREATED SUCCESSFULLY")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Train 4 models using these splits")
    print(f"2. Evaluate all 4 on the same test set")
    print(f"3. Compare: Does Train-2 (smaller, curated) beat Train-1 (larger, all recoverable)?")
    print(f"4. Check: Does Train-3 (with noise) hurt performance vs Train-1?")


if __name__ == "__main__":
    main()