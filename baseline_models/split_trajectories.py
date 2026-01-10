"""
Split trajectories into success and failure files.
Usage: python split_trajectories.py trajectories_3000.json
"""
import json
import sys

def split_trajectories(input_file):
    print(f"Loading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    
    trajectories = data.get('trajectories', [])
    successes = [t for t in trajectories if t.get('success', False)]
    failures = [t for t in trajectories if not t.get('success', False)]
    
    base_name = input_file.replace('.json', '')
    
    # Save successes
    success_file = f"{base_name}_successes.json"
    success_data = {
        'metadata': data.get('metadata', {}),
        'summary': {
            'total': len(successes),
            'avg_reward': sum(t['final_reward'] for t in successes) / len(successes) if successes else 0,
            'avg_steps': sum(t['num_steps'] for t in successes) / len(successes) if successes else 0,
        },
        'trajectories': successes,
    }
    
    with open(success_file, 'w') as f:
        json.dump(success_data, f, indent=2)
    
    print(f"✓ Saved {len(successes)} successes to: {success_file}")
    
    # Save failures
    failure_file = f"{base_name}_failures.json"
    failure_data = {
        'metadata': data.get('metadata', {}),
        'summary': {
            'total': len(failures),
            'avg_reward': sum(t['final_reward'] for t in failures) / len(failures) if failures else 0,
            'avg_steps': sum(t['num_steps'] for t in failures) / len(failures) if failures else 0,
        },
        'trajectories': failures,
    }
    
    with open(failure_file, 'w') as f:
        json.dump(failure_data, f, indent=2)
    
    print(f"✓ Saved {len(failures)} failures to: {failure_file}")
    
    print("\n" + "="*60)
    print("SPLIT SUMMARY:")
    print(f"  Total:     {len(trajectories)}")
    print(f"  Successes: {len(successes)} ({len(successes)/len(trajectories)*100:.1f}%)")
    print(f"  Failures:  {len(failures)} ({len(failures)/len(trajectories)*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_trajectories.py <input_file.json>")
        sys.exit(1)
    
    split_trajectories(sys.argv[1])
