"""
GPT-based Trajectory Collection for WebShop
============================================
Uses OpenAI API (GPT-4o, GPT-4o-mini, etc.) to generate expert trajectories.

Usage:
    # Test with GPT-4o (30 tasks to validate performance)
    python collect_trajectories_gpt.py --model gpt-4o --num_tasks 30 --output ./test_gpt4o.json
    
    # Full collection with GPT-4o
    python collect_trajectories_gpt.py --model gpt-4o --num_tasks 1000 --workers 16 --output ./trajectories_gpt4o.json
    
    # Budget-friendly: GPT-4o-mini for comparison
    python collect_trajectories_gpt.py --model gpt-4o-mini --num_tasks 30 --output ./test_gpt4o_mini.json

Setup:
    1. Create a .env file in the same directory with: OPENAI_API_KEY=sk-your-key-here
    2. Add .env to .gitignore
"""

import os
import sys
import json
import argparse
import asyncio
import aiohttp
from openai import OpenAI, AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from tqdm import tqdm
import time
import re


def load_env():
    """Load environment variables from .env file."""
    # Check multiple possible locations for .env
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.getcwd(), '.env'),
        '.env'
    ]
    
    for env_file in possible_paths:
        if os.path.exists(env_file):
            print(f"Loading API key from: {env_file}")
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            return True
    
    return False


# Load .env file
if not os.environ.get('OPENAI_API_KEY'):
    if not load_env():
        print("Warning: No .env file found and OPENAI_API_KEY not set.")
        print("Create a .env file with: OPENAI_API_KEY=sk-your-key-here")


# Import WebShop environment
from train_rl import parse_args as webenv_args
from env import WebEnv


# System prompt for GPT
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
- Use click[Next] to see more products if current ones don't match
- Use click[Back to Search] to try a different search
- Use click[< Prev] to go back to previous page

Think step by step about which action best matches the user's requirements, then respond with ONLY the action in the exact format: action[argument]

Examples of valid actions:
- search[red running shoes size 10]
- click[B07ABC123]
- click[Buy Now]
- click[Next]
- click[large]
- click[Back to Search]
"""


def parse_action(response_text, valid_actions):
    """Extract action from GPT response."""
    text = response_text.strip()
    
    # Try exact match first (case-insensitive)
    text_lower = text.lower()
    for valid in valid_actions:
        if valid.lower() == text_lower:
            return valid
    
    # Try to find action pattern
    patterns = [
        r'(search\[.+?\])',
        r'(click\[.+?\])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            action = match.group(1)
            # Find closest valid action
            for valid in valid_actions:
                if valid.lower() == action.lower():
                    return valid
            return action
    
    # Substring match as fallback
    for valid in valid_actions:
        if valid.lower() in text_lower or text_lower in valid.lower():
            return valid
    
    # Default: return first valid action
    return valid_actions[0] if valid_actions else text


def format_observation(obs, goal, valid_actions):
    """Format observation for GPT prompt."""
    prompt = f"""SHOPPING GOAL: {goal}

CURRENT PAGE:
{obs[:2000]}

AVAILABLE ACTIONS:
{chr(10).join(f'- {a}' for a in valid_actions[:20])}

Select the best action to complete the shopping goal. Respond with ONLY the action."""
    
    return prompt


class GPTTrajectoryCollector:
    def __init__(self, model="gpt-4o", max_retries=3):
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.lock = threading.Lock()
        
    def get_action(self, obs, goal, valid_actions):
        """Get action from GPT model."""
        prompt = format_observation(obs, goal, valid_actions)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=100,
                    temperature=0.0,  # Deterministic expert behavior
                )
                
                # Track tokens
                with self.lock:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                action_text = response.choices[0].message.content
                action = parse_action(action_text, valid_actions)
                return action, action_text
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"API error after {self.max_retries} attempts: {e}")
                    return valid_actions[0], f"ERROR: {e}"
        
        return valid_actions[0], "ERROR: max retries"

    def collect_episode(self, env, idx, max_steps=20, verbose=False):
        """Collect one trajectory using GPT."""
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
            'model': self.model
        }
        
        for step in range(max_steps):
            valid_actions = info['valid']
            
            # Get action from GPT
            action, raw_response = self.get_action(obs, goal, valid_actions)
            
            # Store step
            step_data = {
                'step': step,
                'observation': obs[:1000],  # Truncate for storage
                'valid_actions': valid_actions,
                'action': action,
                'raw_response': raw_response
            }
            trajectory['steps'].append(step_data)
            
            if verbose:
                print(f"  Step {step}: {action[:60]}")
            
            # Take action
            obs, reward, done, info = env.step(action)
            
            if done:
                trajectory['final_reward'] = reward * 10  # Scale to 0-100
                trajectory['success'] = (reward == 10.0)
                trajectory['num_steps'] = step + 1
                break
        
        if verbose:
            status = 'SUCCESS' if trajectory['success'] else 'FAILURE'
            print(f"  Result: {status} | Reward: {trajectory['final_reward']:.1f}")
        
        return trajectory
    
    def get_cost_estimate(self):
        """Estimate cost based on tokens used."""
        # Pricing per 1M tokens (as of Jan 2025)
        pricing = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
            'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
            'gpt-5.1-mini': {'input': 0.50, 'output': 2.00},
            'gpt-5-mini': {'input': 0.50, 'output': 2.00},
        }
        
        model_pricing = pricing.get(self.model, {'input': 2.5, 'output': 10.0})
        
        input_cost = (self.total_input_tokens / 1_000_000) * model_pricing['input']
        output_cost = (self.total_output_tokens / 1_000_000) * model_pricing['output']
        
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }


def collect_worker(args):
    """Worker function for parallel collection."""
    worker_id, task_indices, model, split, max_steps, verbose = args
    
    # Each worker creates its own environment and collector
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split=split)
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    
    collector = GPTTrajectoryCollector(model=model)
    
    trajectories = []
    for idx in task_indices:
        try:
            traj = collector.collect_episode(env, idx, max_steps=max_steps, verbose=verbose)
            trajectories.append(traj)
        except Exception as e:
            print(f"Worker {worker_id} error on task {idx}: {e}")
    
    return trajectories, collector.get_cost_estimate()


def collect_parallel(model, task_indices, num_workers, split='train', max_steps=20, verbose=False):
    """Collect trajectories in parallel."""
    
    # Split tasks among workers
    chunks = [[] for _ in range(num_workers)]
    for i, idx in enumerate(task_indices):
        chunks[i % num_workers].append(idx)
    
    # Prepare worker arguments
    worker_args = [
        (worker_id, chunks[worker_id], model, split, max_steps, verbose)
        for worker_id in range(num_workers)
    ]
    
    all_trajectories = []
    total_cost = {'input_tokens': 0, 'output_tokens': 0, 'input_cost': 0, 'output_cost': 0, 'total_cost': 0}
    
    print(f"\nStarting {num_workers} workers for {len(task_indices)} tasks...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(collect_worker, args): args[0] for args in worker_args}
        
        with tqdm(total=len(task_indices), desc=f"Collecting ({model})") as pbar:
            completed = 0
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    trajectories, cost = future.result()
                    all_trajectories.extend(trajectories)
                    
                    # Aggregate cost
                    for key in total_cost:
                        total_cost[key] += cost[key]
                    
                    pbar.update(len(trajectories))
                    completed += len(trajectories)
                    
                except Exception as e:
                    print(f"Worker {worker_id} failed: {e}")
    
    return all_trajectories, total_cost


def collect_sequential(model, task_indices, split='train', max_steps=20, verbose=False):
    """Collect trajectories sequentially (for testing/debugging)."""
    
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split=split)
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    
    collector = GPTTrajectoryCollector(model=model)
    
    trajectories = []
    for idx in tqdm(task_indices, desc=f"Collecting ({model})"):
        try:
            traj = collector.collect_episode(env, idx, max_steps=max_steps, verbose=verbose)
            trajectories.append(traj)
        except Exception as e:
            print(f"Error on task {idx}: {e}")
    
    return trajectories, collector.get_cost_estimate()


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-based trajectory collection")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Number of tasks to collect")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting task index")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (0 for sequential)")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum steps per episode")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--output", type=str, default="./trajectories_gpt4o.json",
                        help="Output file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print episode details")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"=" * 60)
    print(f"GPT TRAJECTORY COLLECTION")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.num_tasks}")
    print(f"Workers: {args.workers}")
    print(f"Split: {args.split}")
    print(f"Max steps: {args.max_steps}")
    print(f"=" * 60)
    
    # Task indices
    task_indices = list(range(args.start_idx, args.start_idx + args.num_tasks))
    
    # Collect
    start_time = time.time()
    
    if args.workers > 0:
        trajectories, cost = collect_parallel(
            model=args.model,
            task_indices=task_indices,
            num_workers=args.workers,
            split=args.split,
            max_steps=args.max_steps,
            verbose=args.verbose
        )
    else:
        trajectories, cost = collect_sequential(
            model=args.model,
            task_indices=task_indices,
            split=args.split,
            max_steps=args.max_steps,
            verbose=args.verbose
        )
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    successes = [t for t in trajectories if t['success']]
    failures = [t for t in trajectories if not t['success']]
    rewards = [t['final_reward'] for t in trajectories]
    
    # Check for navigation skills
    uses_next = sum(1 for t in successes if any('Next' in s['action'] for s in t['steps']))
    uses_back = sum(1 for t in successes if any('Back' in s['action'] for s in t['steps']))
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"COLLECTION SUMMARY")
    print(f"=" * 60)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Successes: {len(successes)} ({len(successes)/len(trajectories)*100:.1f}%)")
    print(f"Failures: {len(failures)} ({len(failures)/len(trajectories)*100:.1f}%)")
    print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Average steps: {sum(t['num_steps'] for t in trajectories)/len(trajectories):.1f}")
    print(f"Uses Next (in successes): {uses_next}")
    print(f"Uses Back (in successes): {uses_back}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(trajectories):.2f}s per trajectory)")
    print(f"\n--- Cost ---")
    print(f"Input tokens: {cost['input_tokens']:,}")
    print(f"Output tokens: {cost['output_tokens']:,}")
    print(f"Estimated cost: ${cost['total_cost']:.4f}")
    if len(trajectories) > 0:
        print(f"Cost per trajectory: ${cost['total_cost']/len(trajectories):.4f}")
        print(f"Projected cost for 1000 tasks: ${cost['total_cost'] * 1000 / len(trajectories):.2f}")
    print(f"=" * 60)
    
    # Save results
    output_data = {
        'metadata': {
            'model': args.model,
            'collection_date': datetime.now().isoformat(),
            'num_tasks': len(trajectories),
            'split': args.split,
            'max_steps': args.max_steps,
            'elapsed_seconds': elapsed,
        },
        'summary': {
            'total': len(trajectories),
            'successes': len(successes),
            'failures': len(failures),
            'success_rate': len(successes) / len(trajectories) * 100 if trajectories else 0,
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
            'avg_steps': sum(t['num_steps'] for t in trajectories) / len(trajectories) if trajectories else 0,
            'uses_next': uses_next,
            'uses_back': uses_back,
        },
        'cost': cost,
        'trajectories': trajectories,
        'success_trajectories': successes,
        'failed_trajectories': failures,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved to: {args.output}")
    print(f"\nTo analyze: python -c \"import json; d=json.load(open('{args.output}')); print(d['summary'])\"")


if __name__ == "__main__":
    main()