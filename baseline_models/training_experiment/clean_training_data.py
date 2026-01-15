#!/usr/bin/env python3
"""
Clean training data - keep only fields needed for WebShop training.

Removes: action_probs, entropy, is_recoverable, etc.
Keeps: state, action, goal
"""

import json
import argparse
import os


def clean_training_data(input_file, output_file):
    """Remove unnecessary fields, keep only what's needed for training"""
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} examples")
    
    # Check what fields the first example has
    if data:
        print(f"  Current fields: {list(data[0].keys())}")
    
    # Clean each example - keep only essential fields
    cleaned = []
    for ex in data:
        clean_ex = {
            'instruction_text': ex.get('goal', ''),  # Required
            'state': ex.get('state', ''),            # Required
            'action': ex.get('action', ''),          # Required (correct action)
            'label': ex.get('action', ''),           # Same as action for single-choice
        }
        
        # Optional: keep trajectory metadata if present
        if 'trajectory_id' in ex:
            clean_ex['trajectory_id'] = ex['trajectory_id']
        if 'step' in ex:
            clean_ex['step'] = ex['step']
        
        cleaned.append(clean_ex)
    
    print(f"Saving cleaned data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(cleaned, f, indent=2)
    
    print(f"✓ Cleaned {len(cleaned)} examples")
    print(f"  Fields kept: {list(cleaned[0].keys())}")
    
    # Show size reduction
    input_size = os.path.getsize(input_file) / (1024 * 1024)
    output_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Size: {input_size:.1f}MB → {output_size:.1f}MB (saved {input_size - output_size:.1f}MB)")


def main():
    parser = argparse.ArgumentParser(description="Clean training data for WebShop")
    parser.add_argument("--input", required=True, help="Input JSON file (with extra fields)")
    parser.add_argument("--output", required=True, help="Output JSON file (cleaned)")
    args = parser.parse_args()
    
    clean_training_data(args.input, args.output)


if __name__ == "__main__":
    main()