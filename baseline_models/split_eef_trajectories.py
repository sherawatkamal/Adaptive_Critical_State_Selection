#!/usr/bin/env python3
"""
Split EEF trajectories into 4 training datasets.

Key difference: Keep ENTIRE trajectories intact, not individual segments.
Classify each trajectory based on its overall characteristics.
"""

import json
import argparse
import numpy as np
from collections import defaultdict


def load_eef_data(simulation_stats, success_segments, improvement_segments, failure_segments):
    """Load all EEF results"""
    
    print("Loading EEF data...")
    
    with open(simulation_stats, 'r') as f:
        stats = json.load(f)
    
    with open(success_segments, 'r') as f:
        success = json.load(f)
    
    with open(improvement_segments, 'r') as f:
        improvement = json.load(f)
    
    with open(failure_segments, 'r') as f:
        failure = json.load(f)
    
    print(f"  Success trajectories: {len(success)}")
    print(f"  Improvement trajectories: {len(improvement)}")
    print(f"  Failure trajectories: {len(failure)}")
    
    return {
        'stats': stats,
        'success': success,
        'improvement': improvement,
        'failure': failure
    }


def classify_trajectory(trajectory, traj_type='unknown'):
    """
    Classify a trajectory based on its characteristics.
    
    Args:
        trajectory: The trajectory dict from EEF
        traj_type: 'success', 'improvement', or 'failure'
    
    Returns:
        - mean_entropy: Average normalized entropy across all steps
        - is_recoverable: Did the trajectory show any improvement?
        - max_entropy: Maximum entropy in the trajectory
        - min_entropy: Minimum entropy in the trajectory
    """
    
    # Extract entropy values from all steps
    entropies = []
    
    # EEF stores data differently depending on segment type
    # Check for states list (common structure)
    if 'states' in trajectory and isinstance(trajectory['states'], list):
        # Already has states - this is the trajectory data
        pass
    
    # Try to get entropy from different possible locations
    if 'normalized_entropy' in trajectory:
        entropies.append(trajectory['normalized_entropy'])
    
    if 'entropy' in trajectory:
        # Normalize if raw entropy (assuming max ~2.5 for typical action spaces)
        raw_entropy = trajectory['entropy']
        normalized = min(raw_entropy / 2.5, 1.0)
        entropies.append(normalized)
    
    # Check for step-level data
    if 'actions' in trajectory and isinstance(trajectory['actions'], list):
        # Count steps
        num_steps = len(trajectory['actions'])
        # Use a default middle value if no entropy info
        if not entropies:
            entropies = [0.5] * num_steps
    
    # Compute statistics
    if entropies:
        mean_entropy = np.mean(entropies)
        max_entropy = np.max(entropies)
        min_entropy = np.min(entropies)
    else:
        # Default if no entropy info
        mean_entropy = 0.5
        max_entropy = 0.5
        min_entropy = 0.5
    
    # Check recoverability based on trajectory type
    is_recoverable = False
    
    # Success trajectories are ALWAYS recoverable
    if traj_type == 'success':
        is_recoverable = True
    
    # Improvement trajectories are recoverable BY DEFINITION
    if traj_type == 'improvement':
        is_recoverable = True
    
    # Check explicit flags
    if trajectory.get('success', False):
        is_recoverable = True
    
    if trajectory.get('improvement', 0) > 0:
        is_recoverable = True
    
    if trajectory.get('is_recoverable', False):
        is_recoverable = True
    
    # Check reward improvement
    if 'reward' in trajectory and trajectory['reward'] > 0:
        is_recoverable = True
    
    # For failure segments, check if there was ANY improvement
    if traj_type == 'failure':
        # Even failures might have some recoverable states
        # But by default, failures are unrecoverable unless proven otherwise
        pass
    
    return {
        'mean_entropy': mean_entropy,
        'max_entropy': max_entropy,
        'min_entropy': min_entropy,
        'is_recoverable': is_recoverable,
        'trajectory_type': traj_type
    }


def split_trajectories(eef_data, middle_u_min=0.4, middle_u_max=0.7):
    """
    Split trajectories into 4 training sets.
    
    Strategy: Use mean entropy across the trajectory for classification.
    """
    
    print("\nClassifying trajectories...")
    
    # Process each type separately to preserve type information
    all_trajectories = []
    
    # Add success trajectories
    for traj in eef_data['success']:
        traj['_type'] = 'success'
        all_trajectories.append(traj)
    
    # Add improvement trajectories  
    for traj in eef_data['improvement']:
        traj['_type'] = 'improvement'
        all_trajectories.append(traj)
    
    # Add failure trajectories
    for traj in eef_data['failure']:
        traj['_type'] = 'failure'
        all_trajectories.append(traj)
    
    print(f"  Total trajectories: {len(all_trajectories)}")
    print(f"    Success: {len(eef_data['success'])}")
    print(f"    Improvement: {len(eef_data['improvement'])}")
    print(f"    Failure: {len(eef_data['failure'])}")
    
    # Classify each trajectory
    train1 = []  # All recoverable
    train2 = []  # Middle U AND recoverable
    train3 = []  # All (recoverable + unrecoverable)
    train4 = []  # Extreme U (high or low) AND recoverable
    
    stats = {
        'total': len(all_trajectories),
        'recoverable': 0,
        'unrecoverable': 0,
        'low_u': 0,
        'middle_u': 0,
        'high_u': 0
    }
    
    for traj in all_trajectories:
        traj_type = traj.get('_type', 'unknown')
        classification = classify_trajectory(traj, traj_type)
        
        U = classification['mean_entropy']
        is_recoverable = classification['is_recoverable']
        
        # Update stats
        if is_recoverable:
            stats['recoverable'] += 1
        else:
            stats['unrecoverable'] += 1
        
        if U < middle_u_min:
            stats['low_u'] += 1
        elif U <= middle_u_max:
            stats['middle_u'] += 1
        else:
            stats['high_u'] += 1
        
        # Add classification info to trajectory
        traj['_classification'] = classification
        
        # Train-1: All recoverable
        if is_recoverable:
            train1.append(traj)
        
        # Train-2: Middle U AND recoverable
        if (middle_u_min <= U <= middle_u_max) and is_recoverable:
            train2.append(traj)
        
        # Train-3: Everything
        train3.append(traj)
        
        # Train-4: Extreme U (low or high) AND recoverable
        if (U < middle_u_min or U > middle_u_max) and is_recoverable:
            train4.append(traj)
    
    print("\n" + "="*70)
    print("TRAJECTORY CLASSIFICATION STATISTICS")
    print("="*70)
    print(f"Total trajectories: {stats['total']}")
    print(f"  Recoverable: {stats['recoverable']}")
    print(f"  Unrecoverable: {stats['unrecoverable']}")
    print(f"\nBy uncertainty:")
    print(f"  Low U (< {middle_u_min}): {stats['low_u']}")
    print(f"  Middle U ({middle_u_min}-{middle_u_max}): {stats['middle_u']}")
    print(f"  High U (> {middle_u_max}): {stats['high_u']}")
    
    return {
        'train1': train1,
        'train2': train2,
        'train3': train3,
        'train4': train4,
        'stats': stats
    }


def save_splits(splits, output_dir):
    """Save the 4 training splits"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAINING SPLIT STATISTICS")
    print("="*70)
    
    for name, trajs in [
        ('train1_all_recoverable', splits['train1']),
        ('train2_middle_u_recoverable', splits['train2']),
        ('train3_all_states', splits['train3']),
        ('train4_high_low_u_recoverable', splits['train4'])
    ]:
        filename = os.path.join(output_dir, f"{name}_trajectories.json")
        
        with open(filename, 'w') as f:
            json.dump(trajs, f, indent=2)
        
        # Stats
        recoverable = sum(1 for t in trajs if t['_classification']['is_recoverable'])
        mean_entropies = [t['_classification']['mean_entropy'] for t in trajs]
        
        print(f"\n{name}:")
        print(f"  Total trajectories: {len(trajs)}")
        print(f"  Recoverable: {recoverable}")
        print(f"  Unrecoverable: {len(trajs) - recoverable}")
        if mean_entropies:
            print(f"  Mean entropy: {np.mean(mean_entropies):.3f} ± {np.std(mean_entropies):.3f}")
            print(f"  Entropy range: [{np.min(mean_entropies):.3f}, {np.max(mean_entropies):.3f}]")
        print(f"  ✓ Saved: {filename}")
    
    # Save summary
    summary = {
        'total_trajectories': splits['stats']['total'],
        'train1_size': len(splits['train1']),
        'train2_size': len(splits['train2']),
        'train3_size': len(splits['train3']),
        'train4_size': len(splits['train4']),
        'train2_vs_train1_ratio': len(splits['train2']) / len(splits['train1']) if splits['train1'] else 0,
        'classification_stats': splits['stats']
    }
    
    summary_file = os.path.join(output_dir, 'split_summary_trajectories.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    print("\n" + "="*70)
    print("KEY COMPARISON: Train-2 (Middle U) vs Train-1 (All Recoverable)")
    print("="*70)
    print(f"Train-2 size: {len(splits['train2'])} trajectories ({summary['train2_vs_train1_ratio']:.1%} of Train-1)")
    print(f"Train-1 size: {len(splits['train1'])} trajectories")


def main():
    parser = argparse.ArgumentParser(description="Split EEF trajectories into 4 training sets")
    parser.add_argument("--simulation_stats", required=True)
    parser.add_argument("--success_segments", required=True)
    parser.add_argument("--improvement_segments", required=True)
    parser.add_argument("--failure_segments", required=True)
    parser.add_argument("--output_dir", default="./training_splits_trajectories")
    parser.add_argument("--middle_u_min", type=float, default=0.4)
    parser.add_argument("--middle_u_max", type=float, default=0.7)
    args = parser.parse_args()
    
    print("="*70)
    print("SPLIT EEF TRAJECTORIES INTO 4 TRAINING DATASETS")
    print("="*70)
    print(f"Middle U range: [{args.middle_u_min}, {args.middle_u_max}]")
    print()
    
    # Load data
    eef_data = load_eef_data(
        args.simulation_stats,
        args.success_segments,
        args.improvement_segments,
        args.failure_segments
    )
    
    # Split trajectories
    splits = split_trajectories(eef_data, args.middle_u_min, args.middle_u_max)
    
    # Save
    save_splits(splits, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ TRAJECTORY SPLITTING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Convert each split to WebShop JSONL format")
    print("2. Train 4 models (one per split)")
    print("3. Evaluate and compare results")


if __name__ == "__main__":
    main()