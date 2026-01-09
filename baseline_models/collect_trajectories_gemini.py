"""
Simple Sequential Gemini Trajectory Collection
===============================================
No parallel workers = No API key propagation issues!

Usage:
    python collect_gemini_simple.py --num_tasks 100 --output test.json
    python collect_gemini_simple.py --num_tasks 3000 --output trajectories.json
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from tqdm import tqdm

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except:
    pass

# Import WebShop
from train_rl import parse_args as webenv_args
from env import WebEnv

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("ERROR: google-generativeai not installed")
    print("  pip install google-generativeai")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert online shopping assistant.

You interact with WebShop to find and purchase products. At each step:
1. See the user's shopping goal
2. See the current webpage
3. See available actions

Select the BEST action to complete the task.

Action formats:
- search[query] - Search (e.g., search[noise cancelling headphones])
- click[product_id] - Click product (e.g., click[B07XYZ123])
- click[button] - Navigate (e.g., click[Next], click[Back to Search])
- click[option] - Select option (e.g., click[large], click[blue])
- click[Buy Now] - Purchase

Respond with ONLY the action: action[argument]"""


def parse_action(response_text, valid_actions):
    """Extract action from Gemini response."""
    text = response_text.strip()
    
    # Find action pattern
    patterns = [r'(search\[.+?\])', r'(click\[.+?\])']
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            action = match.group(1)
            if action.lower().startswith('click'):
                action = 'click' + action[5:]
            return action
    
    # Try to match valid actions
    for valid in valid_actions:
        if valid.lower() in text.lower():
            return valid
    
    return valid_actions[0] if valid_actions else text


class GeminiCollector:
    def __init__(self, model="gemini-2.0-flash-exp"):
        # Get API key
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        api_key = 'AIzaSyDE96Htaa-cJsRPKZ2Gbv0SoYFeX0YO7Sw'
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment")
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.total_calls = 0
        
        print(f"✓ Using Gemini: {model} (FREE!)")
    
    def get_action(self, obs, goal, valid_actions):
        """Get action from Gemini."""
        prompt = f"""{SYSTEM_PROMPT}

SHOPPING GOAL: {goal}

CURRENT PAGE:
{obs[:1500]}

AVAILABLE ACTIONS:
{chr(10).join(f'- {a}' for a in valid_actions[:15])}

Select the best action. Respond with ONLY the action."""
        
        try:
            self.total_calls += 1
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=100,
                )
            )
            
            action_text = response.text
            action = parse_action(action_text, valid_actions)
            return action, action_text
            
        except Exception as e:
            print(f"  API error: {e}")
            return valid_actions[0], f"ERROR: {e}"
    
    def collect_episode(self, env, idx, max_steps=15, verbose=False):
        """Collect one trajectory."""
        obs, info = env.reset(idx)
        goal = info['goal']
        
        if verbose:
            print(f"\n=== Task {idx} ===")
            print(f"Goal: {goal}")
        
        trajectory = {
            'idx': idx,
            'goal': goal,
            'steps': [],
            'final_reward': 0,
            'success': False,
            'num_steps': 0,
            'model': self.model_name
        }
        
        for step in range(max_steps):
            valid_actions = info['valid']
            
            action, raw_response = self.get_action(obs, goal, valid_actions)
            
            trajectory['steps'].append({
                'step': step,
                'observation': obs[:1000],
                'valid_actions': valid_actions,
                'action': action,
                'raw_response': raw_response
            })
            
            if verbose:
                print(f"  Step {step}: {action[:60]}")
            
            obs, reward, done, info = env.step(action)
            
            if done:
                trajectory['final_reward'] = reward * 10
                trajectory['success'] = (reward == 1.0)
                trajectory['num_steps'] = step + 1
                break
        
        if verbose:
            status = '✓ SUCCESS' if trajectory['success'] else '✗ FAILURE'
            print(f"  {status} | Reward: {trajectory['final_reward']:.1f}")
        
        return trajectory


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-exp")
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--output", type=str, default="trajectories_gemini.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GEMINI TRAJECTORY COLLECTION (Sequential)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.num_tasks}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Initialize
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split='train')
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    
    collector = GeminiCollector(model=args.model)
    
    # Collect
    trajectories = []
    task_indices = list(range(args.start_idx, args.start_idx + args.num_tasks))
    
    start_time = time.time()
    
    for idx in tqdm(task_indices, desc="Collecting"):
        try:
            traj = collector.collect_episode(env, idx, max_steps=args.max_steps, verbose=args.verbose)
            trajectories.append(traj)
        except Exception as e:
            print(f"\nError on task {idx}: {e}")
    
    elapsed = time.time() - start_time
    
    # Statistics
    successes = [t for t in trajectories if t['success']]
    failures = [t for t in trajectories if not t['success']]
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(trajectories)}")
    print(f"Success: {len(successes)} ({len(successes)/len(trajectories)*100:.1f}%)")
    print(f"Failures: {len(failures)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(trajectories):.2f}s per task)")
    print(f"API calls: {collector.total_calls}")
    print(f"Cost: $0.00 (Gemini is FREE!)")
    print("=" * 60)
    
    # Save
    output_data = {
        'metadata': {
            'model': args.model,
            'date': datetime.now().isoformat(),
            'num_tasks': len(trajectories),
        },
        'summary': {
            'total': len(trajectories),
            'successes': len(successes),
            'failures': len(failures),
            'success_rate': len(successes) / len(trajectories) * 100 if trajectories else 0,
        },
        'trajectories': trajectories,
        'success_trajectories': successes,
        'failure_task_ids': [t['idx'] for t in failures],
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to: {args.output}")
    
    # Save just failure IDs for Stage 2
    if failures:
        failures_file = args.output.replace('.json', '_failures.json')
        with open(failures_file, 'w') as f:
            json.dump({'failure_ids': [t['idx'] for t in failures]}, f)
        print(f"✓ Failure IDs saved to: {failures_file}")


if __name__ == "__main__":
    main()