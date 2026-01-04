"""
Local LLM Trajectory Collection for WebShop
============================================
Uses local models (Qwen3-8B, Llama, etc.) to generate expert trajectories.

Usage:
    # Test with Qwen3-8B
    python collect_trajectories_local.py --model Qwen/Qwen3-8B --num_tasks 10 --output ./test_qwen3.json
    
    # Full collection
    python collect_trajectories_local.py --model Qwen/Qwen3-8B --num_tasks 11000 --output ./trajectories_qwen3_11k.json

Requirements:
    pip install transformers accelerate torch
"""

import os
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from tqdm import tqdm
import time
import re

# Import WebShop environment
from train_rl import parse_args as webenv_args
from env import WebEnv


# System prompt for the model
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
- Use click[Next >] to see more products if current ones don't match
- Use click[Back to Search] to try a different search
- Use click[< Prev] to go back to previous page

Think step by step about which action best matches the user's requirements, then respond with ONLY the action in the exact format: action[argument]

Examples of valid actions:
- search[red running shoes size 10]
- click[B07ABC123]
- click[Buy Now]
- click[Next >]
- click[large]
- click[Back to Search]
"""


def parse_action(response_text, valid_actions):
    """Extract action from model response."""
    text = response_text.strip()
    
    # Try to find action pattern
    patterns = [
        r'(search\[.+?\])',
        r'(click\[.+?\])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            action = match.group(1)
            # Normalize click to lowercase
            if action.lower().startswith('click'):
                action = 'click' + action[5:]
            return action
    
    # If no pattern found, try to match with valid actions
    text_lower = text.lower()
    for valid in valid_actions:
        if valid.lower() in text_lower:
            return valid
    
    # Default: return first valid action
    return valid_actions[0] if valid_actions else text


def format_prompt(obs, goal, valid_actions, tokenizer):
    """Format prompt for the model."""
    user_content = f"""SHOPPING GOAL: {goal}

CURRENT PAGE:
{obs[:2000]}

AVAILABLE ACTIONS:
{chr(10).join(f'- {a}' for a in valid_actions[:20])}

Select the best action to complete the shopping goal. Respond with ONLY the action."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for models without chat template
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant:"
    
    return prompt


class LocalModelCollector:
    def __init__(self, model_name="Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        print(f"Model loaded in {time.time() - start:.1f}s")
        print(f"Device: {self.model.device}")
        
        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @torch.no_grad()
    def get_action(self, obs, goal, valid_actions, max_new_tokens=100, temperature=0.1):
        """Get action from local model."""
        prompt = format_prompt(obs, goal, valid_actions, self.tokenizer)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs['input_ids'].shape[1]
        
        self.total_input_tokens += input_len
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only new tokens
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        self.total_output_tokens += len(new_tokens)
        
        action = parse_action(response, valid_actions)
        return action, response

    def collect_episode(self, env, idx, max_steps=15, verbose=False):
        """Collect one trajectory using local model."""
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
            
            # Get action from model
            action, raw_response = self.get_action(obs, goal, valid_actions)
            
            # Store step
            step_data = {
                'step': step,
                'observation': obs[:1000],
                'valid_actions': valid_actions,
                'action': action,
                'raw_response': raw_response[:500]  # Truncate for storage
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

    def get_stats(self):
        """Get token usage stats."""
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }


def collect_trajectories(model_name, task_indices, split='train', max_steps=15, 
                         verbose=False, temperature=0.1):
    """Collect trajectories using local model."""
    
    # Setup environment
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split=split)
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    
    # Setup model
    collector = LocalModelCollector(model_name=model_name)
    
    trajectories = []
    for idx in tqdm(task_indices, desc=f"Collecting ({model_name.split('/')[-1]})"):
        try:
            traj = collector.collect_episode(env, idx, max_steps=max_steps, verbose=verbose)
            trajectories.append(traj)
        except Exception as e:
            print(f"Error on task {idx}: {e}")
    
    return trajectories, collector.get_stats()


def parse_args():
    parser = argparse.ArgumentParser(description="Local LLM trajectory collection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model name")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of tasks to collect")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting task index")
    parser.add_argument("--max_steps", type=int, default=15,
                        help="Maximum steps per episode")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--output", type=str, default="./trajectories_local.json",
                        help="Output file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print episode details")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"=" * 60)
    print(f"LOCAL MODEL TRAJECTORY COLLECTION")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.num_tasks}")
    print(f"Temperature: {args.temperature}")
    print(f"Split: {args.split}")
    print(f"=" * 60)
    
    # Task indices
    task_indices = list(range(args.start_idx, args.start_idx + args.num_tasks))
    
    # Collect
    start_time = time.time()
    
    trajectories, stats = collect_trajectories(
        model_name=args.model,
        task_indices=task_indices,
        split=args.split,
        max_steps=args.max_steps,
        verbose=args.verbose,
        temperature=args.temperature
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
    print(f"Uses Next (in successes): {uses_next}")
    print(f"Uses Back (in successes): {uses_back}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(trajectories):.2f}s per trajectory)")
    print(f"\n--- Token Usage ---")
    print(f"Input tokens: {stats['input_tokens']:,}")
    print(f"Output tokens: {stats['output_tokens']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"=" * 60)
    
    # Save results
    output_data = {
        'metadata': {
            'model': args.model,
            'collection_date': datetime.now().isoformat(),
            'num_tasks': len(trajectories),
            'split': args.split,
            'elapsed_seconds': elapsed,
            'temperature': args.temperature,
        },
        'summary': {
            'total': len(trajectories),
            'successes': len(successes),
            'failures': len(failures),
            'success_rate': len(successes) / len(trajectories) * 100,
            'avg_reward': sum(rewards) / len(rewards),
            'uses_next': uses_next,
            'uses_back': uses_back,
        },
        'token_usage': stats,
        'trajectories': trajectories,
        'success_trajectories': successes,
        'failed_trajectories': failures,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()