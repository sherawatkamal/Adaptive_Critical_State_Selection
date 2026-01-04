#!/usr/bin/env python3
"""Extract examples for manual teachable moments analysis.

Works with eef_detailed_fixed.py / eef_complete_updated.py outputs.
"""

import json
import argparse
import glob
from pathlib import Path

def load_simulations(results_dir):
    """Load all_simulated_states from results directory."""
    results_path = Path(results_dir)
    
    # Try different possible filenames
    patterns = [
        "all_simulated_states*.json",
        "all_simulations.json",
        "*simulated*.json"
    ]
    
    for pattern in patterns:
        matches = list(results_path.glob(pattern))
        if matches:
            # Use most recent if multiple
            matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"Found: {matches[0]}")
            with open(matches[0]) as f:
                return json.load(f)
    
    return None

def extract_examples(results_dir, n_examples=10):
    """Extract high-entropy success and low-entropy failure examples."""
    
    all_sims = load_simulations(results_dir)
    if all_sims is None:
        print(f"No simulation data found in {results_dir}")
        print("Expected: all_simulated_states_*.json")
        return None, None
    
    print(f"Loaded {len(all_sims)} simulations")
    
    # Separate by entropy and outcome
    high_entropy_success = []
    low_entropy_fail = []
    
    for sim in all_sims:
        entropy = sim.get('true_entropy', 0)
        is_success = sim.get('is_success', False)
        final_reward = sim.get('final_reward', 0)
        
        if entropy > 1.5 and is_success:
            high_entropy_success.append(sim)
        elif entropy < 1.0 and not is_success and final_reward < 50:
            low_entropy_fail.append(sim)
    
    print(f"High entropy successes: {len(high_entropy_success)}")
    print(f"Low entropy failures: {len(low_entropy_fail)}")
    
    # Sort by entropy
    high_entropy_success.sort(key=lambda x: x.get('true_entropy', 0), reverse=True)
    low_entropy_fail.sort(key=lambda x: x.get('true_entropy', 0))
    
    return high_entropy_success[:n_examples], low_entropy_fail[:n_examples]

def format_for_annotation(examples, label):
    """Format examples for manual review."""
    output = []
    for i, ex in enumerate(examples):
        # Extract action types from valid_actions
        valid_acts = ex.get('valid_actions', [])
        action_types = categorize_actions(valid_acts)
        
        entry = {
            "id": f"{label}_{i+1}",
            "label": label,
            "entropy": round(ex.get('true_entropy', 0), 3),
            "final_reward": ex.get('final_reward', 0),
            "original_reward": ex.get('original_reward', 0),
            "is_success": ex.get('is_success', False),
            "task_id": ex.get('task_id', 'unknown'),
            "recovery_step": ex.get('recovery_step', -1),
            "trajectory_length": ex.get('trajectory_length', 0),
            "observation_preview": ex.get('state', '')[:500],
            "valid_actions": valid_acts[:10],  # First 10
            "num_valid_actions": ex.get('num_valid_actions', len(valid_acts)),
            "action_types": action_types,
            # Annotation fields
            "annotations": {
                "is_strategic_decision": None,
                "decision_type": None,
                "multiple_viable_options": None,
                "expert_made_mistake": None,
                "why_high_entropy": None,
                "notes": ""
            }
        }
        output.append(entry)
    return output

def categorize_actions(valid_acts):
    """Categorize actions into types."""
    cats = {
        "navigation": 0,  # Next, Prev, Back to Search
        "product_click": 0,  # B0xxxxx product IDs
        "option_select": 0,  # size, color options
        "purchase": 0,  # Buy Now
        "other": 0
    }
    
    for act in valid_acts:
        act_lower = act.lower() if isinstance(act, str) else str(act).lower()
        if 'next' in act_lower or 'prev' in act_lower or 'back' in act_lower:
            cats["navigation"] += 1
        elif act_lower.startswith('b0') and len(act_lower) == 10:
            cats["product_click"] += 1
        elif 'buy' in act_lower:
            cats["purchase"] += 1
        elif any(x in act_lower for x in ['small', 'medium', 'large', 'black', 'white', 'red', 'blue']):
            cats["option_select"] += 1
        else:
            cats["other"] += 1
    
    return cats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./eef_final_entropy')
    parser.add_argument('--output', default='./teachable_moments_analysis.json')
    parser.add_argument('--n_examples', type=int, default=10)
    args = parser.parse_args()
    
    high_success, low_fail = extract_examples(args.results_dir, args.n_examples)
    
    if high_success is None:
        return
    
    # Format for annotation
    analysis = {
        "high_entropy_success": format_for_annotation(high_success, "HIGH_ENTROPY_SUCCESS"),
        "low_entropy_failure": format_for_annotation(low_fail, "LOW_ENTROPY_FAILURE"),
        "summary": {
            "n_high_entropy_success": len(high_success),
            "n_low_entropy_failure": len(low_fail),
            "avg_entropy_high": round(sum(x.get('true_entropy', 0) for x in high_success) / max(len(high_success), 1), 3),
            "avg_entropy_low": round(sum(x.get('true_entropy', 0) for x in low_fail) / max(len(low_fail), 1), 3),
        },
        "annotation_guide": {
            "is_strategic_decision": "True if this is a meaningful choice (not obvious next step)",
            "decision_type": "navigation / product_selection / option_selection / search / purchase",
            "multiple_viable_options": "True if 2+ actions seem reasonable",
            "expert_made_mistake": "True if expert took suboptimal action leading here",
            "why_high_entropy": "For high-entropy: why is model uncertain here?"
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nSaved to {args.output}")
    
    # Preview
    print("\n" + "="*60)
    print("HIGH ENTROPY SUCCESS EXAMPLES")
    print("="*60)
    for ex in high_success[:3]:
        print(f"\nH={ex.get('true_entropy', 0):.2f} | reward: {ex.get('original_reward',0)}→{ex.get('final_reward',0)}")
        print(f"  Actions: {ex.get('num_valid_actions', 0)} valid")
        acts = ex.get('valid_actions', [])[:5]
        print(f"  Sample: {acts}")
    
    print("\n" + "="*60)
    print("LOW ENTROPY FAILURE EXAMPLES")
    print("="*60)
    for ex in low_fail[:3]:
        print(f"\nH={ex.get('true_entropy', 0):.2f} | reward: {ex.get('original_reward',0)}→{ex.get('final_reward',0)}")
        print(f"  Actions: {ex.get('num_valid_actions', 0)} valid")
        acts = ex.get('valid_actions', [])[:5]
        print(f"  Sample: {acts}")

if __name__ == "__main__":
    main()