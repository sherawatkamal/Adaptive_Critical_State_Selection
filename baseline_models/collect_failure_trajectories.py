"""
Trajectory Collection Script for ACSS Pipeline
Collects trajectories with logits for entropy computation using OpenAI API
"""

import os
import sys
import json
import argparse
import textwrap
import numpy as np
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

# Import WebShop environment
from utils import webenv_args
from env import WebEnv


class TokenTracker:
    """Track token usage across API calls."""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.num_calls = 0
    
    def add_usage(self, usage):
        """Add usage from an API response."""
        if usage:
            self.total_input_tokens += getattr(usage, 'prompt_tokens', 0)
            self.total_output_tokens += getattr(usage, 'completion_tokens', 0)
            # OpenAI may provide cached tokens in prompt_tokens_details
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                self.total_cached_tokens += getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            self.num_calls += 1
    
    def get_summary(self):
        """Get summary statistics."""
        return {
            'num_api_calls': self.num_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cached_tokens': self.total_cached_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'avg_input_per_call': self.total_input_tokens / self.num_calls if self.num_calls > 0 else 0,
            'avg_output_per_call': self.total_output_tokens / self.num_calls if self.num_calls > 0 else 0
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("TOKEN USAGE SUMMARY")
        print("="*50)
        print(f"Total API calls: {summary['num_api_calls']:,}")
        print(f"Input tokens: {summary['total_input_tokens']:,}")
        print(f"Output tokens: {summary['total_output_tokens']:,}")
        print(f"Cached tokens: {summary['total_cached_tokens']:,}")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"Avg input/call: {summary['avg_input_per_call']:.1f}")
        print(f"Avg output/call: {summary['avg_output_per_call']:.1f}")


def compute_entropy(logprobs_dict):
    """
    Compute Shannon entropy from OpenAI logprobs.
    
    Args:
        logprobs_dict: Dict with 'token_logprobs' containing log probabilities
    
    Returns:
        entropy: Shannon entropy value
        probs: Probability distribution as list
    """
    if not logprobs_dict or 'token_logprobs' not in logprobs_dict:
        return None, None
    
    log_probs = np.array(logprobs_dict['token_logprobs'])
    probs = np.exp(log_probs)
    probs = probs / probs.sum()  # Normalize
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy, probs.tolist()


def openai_predict(obs, available_actions, goal, client, model="gpt-4", temperature=0.0, token_tracker=None):
    """
    Use OpenAI API to predict action given observation and available actions.
    
    Args:
        obs: Current observation (string)
        available_actions: Dict with 'has_search_bar' (bool) and 'clickables' (list)
        goal: Goal/instruction for the task
        client: OpenAI client instance
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        temperature: Sampling temperature (0 for greedy)
        token_tracker: TokenTracker instance for tracking usage
    
    Returns:
        action: Selected action string (e.g., "search[query]" or "click[button]")
        logprobs_data: Raw logprobs data from API
        entropy: Entropy value
        probs: Probability distribution
        is_search: Whether this was a search action
    """
    import re
    
    has_search = available_actions.get('has_search_bar', False)
    clickables = available_actions.get('clickables', [])
    
    # Truncate observation to avoid token limits
    # obs_truncated = obs[:2000] if len(obs) > 2000 else obs
    
    # Build action options description
    action_options = []
    if has_search:
        action_options.append("- search[ENTER YOUR SEARCH QUERY HERE]: Enter a search query to find products")
        
    for i, clickable in enumerate(clickables):
        action_options.append(f"- click[{clickable}]: Click on '{clickable}'")
    
    
    actions_text = "\n".join(action_options)
    
    prompt = textwrap.dedent(f"""\
        You are a shopping assistant navigating a website. Choose the best action to achieve the goal.

        Goal: {goal}

        Current page:
        {obs}

        Available actions:
        {actions_text}

        Instructions:
        - If you want to click something, respond with: click[element_name]
        {"- If you want to search, respond with: search[ENTER YOUR SEARCH QUERY HERE] (replace the placeholder with your actual query)" if has_search else "- You may not search on this action step"}
        - Choose the action that best helps achieve the goal

        Return only your action choice, nothing else.\
    """)
    
    # print(f"prompt: {prompt}")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100,
        logprobs=True,
        top_logprobs=5
    )
    
    # Track token usage
    if token_tracker and hasattr(response, 'usage'):
        token_tracker.add_usage(response.usage)
    
    # Parse response
    response_text = response.choices[0].message.content.strip().lower()
    
    # Initialize action as None
    action = None
    is_search = False
    
    # Use regex to extract search[...] - get the last instance
    search_matches = re.findall(r'search\[(.*?)\]', response_text)
    if search_matches and has_search:
        search_query = search_matches[-1].strip()  # Get last match
        if search_query:  # Only if there's an actual query
            action = f'search[{search_query}]'
            is_search = True
    
    # Use regex to extract click[...] - get the last instance
    click_matches = re.findall(r'click\[(.*?)\]', response_text)
    if click_matches and action is None:  # Only if we haven't found a search action
        clickable_element = click_matches[-1].strip()  # Get last match
        
        # Validate element is in clickables (exact match required)
        if clickable_element in clickables:
            action = f'click[{clickable_element}]'
            is_search = False
    
    
    # If we still don't have a valid action, return None tuple
    if action is None:
        print(f"Warning: Invalid action format from model: '{response_text}'. Performing no-op.")
        return None, None, None, None, None
    
    
    # Extract logprobs for entropy
    logprobs_data = None
    if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
        logprobs_data = {'token_logprobs': [token.logprob for token in response.choices[0].logprobs.content]}
    
    
    entropy, probs = compute_entropy(logprobs_data) if logprobs_data else (None, None)
    
    return action, logprobs_data, entropy, probs, is_search        
    
        
        


def collect_episode(client, env, idx, model="gpt-4", temperature=0.0, verbose=False, max_steps=200, token_tracker=None):
    """
    Run one episode and collect full trajectory data.
    
    Returns:
        trajectory: dict with all trajectory information
    """
    obs, info = env.reset(idx)
    goal = info['goal']
    
    if verbose:
        print(f"\n=== Episode {idx} ===")
        print(f"Goal: {goal}")
    
    
    trajectory = {
        'idx': idx,
        'goal': goal,
        'steps': [],
        'final_reward': 0,
        'success': False,
        'num_steps': 0
    }
    
    
    for step in range(max_steps):
        # Get available actions from environment
        available_actions = env.get_available_actions()
        
        action, logprobs_data, entropy, probs, is_search = openai_predict(
            obs, available_actions, goal, client, 
            model=model, temperature=temperature, token_tracker=token_tracker
        )
        
        if action == None: 
            continue
            
        # Store step data
        step_data = {
            'step': step,
            'observation': obs[:500],  # Truncate for storage
            'available_actions': available_actions,
            'action': action,
            'is_search': is_search,
            'logprobs': logprobs_data,
            'entropy': entropy,
            'probs': probs
        }
        trajectory['steps'].append(step_data)
        
        if verbose:
            entropy_str = f"{entropy:.3f}" if entropy is not None else "N/A"
            print(f"  Step {step}: {action[:50]}... | Entropy: {entropy_str}")
        
        
        # Take action
        obs, reward, done, info = env.step(action)
        
        if done:
            trajectory['final_reward'] = reward * 10  # Scale 0-10 → 0-100
            trajectory['success'] = (reward == 10.0)  # Perfect score
            trajectory['num_steps'] = step + 1
            break
    
    if verbose:
        print(f"  Result: {'SUCCESS' if trajectory['success'] else 'FAILURE'} | Reward: {trajectory['final_reward']}")
    
    return trajectory



def collect_until_n_failures(client, env, target_failures, model="gpt-4", temperature=0.0, split='train', start_idx=0, max_episodes=5000, verbose=False):
    """Collect trajectories until we have target_failures failed trajectories."""
    
    failed_trajectories = []
    success_trajectories = []
    all_entropies = []
    token_tracker = TokenTracker()
    
    idx = start_idx
    pbar = tqdm(total=target_failures, desc="Collecting failures")
    
    while len(failed_trajectories) < target_failures and (idx - start_idx) < max_episodes:
        traj = collect_episode(client, env, idx, model=model, temperature=temperature, verbose=verbose, token_tracker=token_tracker)
        
        # Collect entropies from all states
        for step in traj['steps']:
            if step['entropy'] is not None:
                all_entropies.append(step['entropy'])
        
        if traj['success']:
            success_trajectories.append(traj)
        else:
            failed_trajectories.append(traj)
            pbar.update(1)
        
        idx += 1
        
        if idx % 10 == 0:
            pbar.set_postfix({
                'total': idx - start_idx, 
                'fail': len(failed_trajectories), 
                'success': len(success_trajectories)
            })
    
    pbar.close()
    
    total_episodes = idx - start_idx
    
    return {
        'metadata': {
            'collection_date': datetime.now().isoformat(),
            'split': split,
            'model': model,
            'temperature': temperature,
            'target_failures': target_failures,
            'total_episodes': total_episodes,
            'start_idx': start_idx
        },
        'failed_trajectories': failed_trajectories,
        'success_trajectories': success_trajectories,
        'token_usage': token_tracker.get_summary(),
        'summary': {
            'total_episodes': total_episodes,
            'num_failures': len(failed_trajectories),
            'num_successes': len(success_trajectories),
            'failure_rate': len(failed_trajectories) / total_episodes * 100 if total_episodes > 0 else 0,
            'success_rate': len(success_trajectories) / total_episodes * 100 if total_episodes > 0 else 0,
            'avg_entropy': np.mean(all_entropies) if all_entropies else 0
        }
    }, token_tracker


def parse_args():
    parser = argparse.ArgumentParser(description="Collect trajectories using OpenAI API")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--output_dir", type=str, default="./trajectories", help="Directory to save trajectories")
    parser.add_argument("--target_failures", type=int, default=100, help="Number of failed trajectories to collect")
    parser.add_argument("--max_episodes", type=int, default=5000, help="Maximum episodes to run (safety limit)")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting task index")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--verbose", action="store_true", help="Print episode details")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    print(f"\nTarget: Collect {args.target_failures} failed trajectories")
    
    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    print(f"OpenAI client initialized with model: {args.model}")
    
    # Setup environment
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split=args.split)
    print(f"Environment loaded (split={args.split})")
    
    # Collect trajectories until we have enough failures
    print(f"\nCollecting until {args.target_failures} failures...")
    results, token_tracker = collect_until_n_failures(
        client, env,
        target_failures=args.target_failures,
        model=args.model,
        temperature=args.temperature,
        split=args.split,
        start_idx=args.start_idx,
        max_episodes=args.max_episodes,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n" + "="*50)
    print("COLLECTION SUMMARY")
    print("="*50)
    print(f"Total episodes run: {results['summary']['total_episodes']}")
    print(f"Failed trajectories: {results['summary']['num_failures']}")
    print(f"Successful trajectories: {results['summary']['num_successes']}")
    print(f"Failure rate: {results['summary']['failure_rate']:.1f}%")
    print(f"Avg entropy: {results['summary']['avg_entropy']:.3f}")
    
    # Print token usage
    token_tracker.print_summary()
    
    # Save trajectories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save failed trajectories (main output for ACSS)
    failed_file = os.path.join(
        args.output_dir, 
        f"trajectories_{args.split}_failed_{args.target_failures}_{args.model.replace('/', '_')}.json"
    )
    failed_data = {
        'metadata': results['metadata'],
        'trajectories': results['failed_trajectories'],
        'token_usage': results['token_usage'],
        'summary': results['summary']
    }
    with open(failed_file, 'w') as f:
        json.dump(failed_data, f, indent=2)
    print(f"\nSaved {len(results['failed_trajectories'])} failed trajectories to: {failed_file}")
    
    # Also save successful trajectories for reference
    success_file = os.path.join(
        args.output_dir,
        f"trajectories_{args.split}_success_{len(results['success_trajectories'])}_{args.model.replace('/', '_')}.json"
    )
    success_data = {
        'metadata': results['metadata'],
        'trajectories': results['success_trajectories'],
        'token_usage': results['token_usage'],
        'summary': results['summary']
    }
    with open(success_file, 'w') as f:
        json.dump(success_data, f, indent=2)
    print(f"Saved {len(results['success_trajectories'])} successful trajectories to: {success_file}")


if __name__ == "__main__":
    main()