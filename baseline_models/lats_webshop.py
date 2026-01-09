#!/usr/bin/env python3
"""
LATS (Language Agent Tree Search) for WebShop
Clean implementation that uses WebShop's native valid_actions format
"""

import os
import sys
import json
import math
import argparse
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# WebShop imports
sys.path.insert(0, os.path.expanduser('~/SageWebShop2'))
from web_agent_site.envs import WebAgentTextEnv

# API imports
import google.generativeai as genai
from openai import OpenAI

class TreeNode:
    """Node in the MCTS tree"""
    def __init__(self, state: Dict, parent=None):
        self.state = state  # {'observation': str, 'action': str, 'reward': float}
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0.0
    
    def uct(self, exploration_constant: float = 1.4) -> float:
        """UCT formula for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.4):
        """Select best child using UCT"""
        return max(self.children, key=lambda c: c.uct(exploration_constant))


class LATSAgent:
    """LATS agent for WebShop"""
    
    def __init__(self, model: str, temperature: float = 0.7, verbose: bool = True):
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.total_api_calls = 0
        
        # Initialize API client
        if model.startswith('gemini'):
            self.api_type = 'gemini'
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.client = genai.GenerativeModel(model)
        elif model.startswith('gpt'):
            self.api_type = 'openai'
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            # Local model
            self.api_type = 'local'
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.client = AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype='auto',
                device_map="auto"
            )
    
    def get_action(self, observation: str, goal: str, valid_actions: List[str], 
                   trajectory: List[str] = None) -> str:
        """
        Get action from LLM using WebShop's native valid_actions.
        
        Args:
            observation: Current observation from WebShop
            goal: Shopping goal/instruction
            valid_actions: List of valid actions from WebShop (ALREADY FORMATTED!)
            trajectory: Previous actions for context
            
        Returns:
            Selected action (must be from valid_actions)
        """
        self.total_api_calls += 1
        
        # Build history from trajectory
        history = ""
        if trajectory and len(trajectory) > 0:
            history = "Previous steps:\n" + "\n".join(trajectory[-5:]) + "\n\n"
        
        # System instruction with example
        system_instruction = """You are an expert online shopping assistant helping users find and purchase products on WebShop.

Example task:
Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

Step 1 - Search:
Available actions: [search[3 ounce bright citrus deodorant], search[bright citrus deodorant sensitive skin], ...]
Action: search[bright citrus deodorant sensitive skin]

Step 2 - Select product:
Available actions: [click[item - bright citrus deodorant by earth mama...], click[item - ginger fresh deodorant...], click[next >], ...]
Action: click[item - bright citrus deodorant by earth mama | natural and safe for sensitive skin, pregnancy and breastfeeding, contains organic calendula 3-ounce]

Step 3 - Select scent option:
Available actions: [click[bright citrus], click[calming lavender], click[3 ounce (pack of 1)], click[buy now], ...]
Action: click[bright citrus]

Step 4 - Select size option:
Available actions: [click[3 ounce (pack of 1)], click[3-ounce (2-pack)], click[buy now], ...]
Action: click[3 ounce (pack of 1)]

Step 5 - Purchase:
Available actions: [click[buy now], click[back to search], ...]
Action: click[buy now]

CRITICAL RULES - READ CAREFULLY:
1. PRICE CHECK: Product price MUST be lower than the budget stated in the goal
2. OPTIONS FIRST: If you see actions like click[1], click[2], click[red], click[large], etc. - these are PRODUCT OPTIONS
   - You MUST select the correct option(s) BEFORE clicking [buy now]
   - Common options: quantity (click[1], click[2]), color, size, style, scent
   - Example: If goal says "60 capsules" and you see click[1], click[2], click[3] - pick click[1] for single bottle
3. VERIFY REQUIREMENTS: Product must match ALL criteria (brand, features, size, price, etc.)
4. EXPLORE: Use [next >] to see more products if current ones don't match
5. EXACT MATCH: Choose ONE action from the "Available actions" list - type it EXACTLY as shown
6. NO BUY WITHOUT OPTIONS: If options are present (numbers, colors, sizes), NEVER click [buy now] until you've selected them!

REMEMBER: Options appear as simple clicks like click[1], click[red], click[small] - select these BEFORE [buy now]!
"""
        
        # Prioritize options over navigation - show product options FIRST
        navigation_actions = ['click[back to search]', 'click[< prev]', 'click[next >]', 
                            'click[description]', 'click[features]', 'click[reviews]',
                            'click[buy now]']  # BUY NOW is also navigation, not an option!
        
        # Separate options from navigation
        options = [a for a in valid_actions if a not in navigation_actions]
        navigation = [a for a in valid_actions if a in navigation_actions]
        
        # Prioritized list: options first (like click[1], click[red]), then navigation (like buy now)
        prioritized_actions = options + navigation
        
        # Show up to 30 actions (more than before to capture all options)
        actions_list = "\n".join(f"{i+1}. {action}" for i, action in enumerate(prioritized_actions[:30]))
        
        # Create prompt
        prompt = f"""{system_instruction}

Current task:
{goal}

{history}Current observation:
{observation[:1500]}

Available actions (YOU MUST CHOOSE ONE OF THESE):
{actions_list}

Choose the BEST action by typing it EXACTLY as shown above.
Your action:"""
        
        if self.verbose:
            print(f"\n    LLM call (depth {len(trajectory) if trajectory else 0}):")
            print(f"    First 5 valid actions: {valid_actions[:5]}")
        
        try:
            # Get LLM response
            if self.api_type == 'gemini':
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': 150
                    }
                )
                action = response.text.strip()
            elif self.api_type == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=150
                )
                action = response.choices[0].message.content.strip()
            else:
                # Local model
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.client.device)
                outputs = self.client.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                action = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            
            # Clean up response
            action = action.replace('Action:', '').replace('action:', '').strip()
            action = action.replace('**', '').strip()
            
            # Remove numbering if present (e.g., "1. search[...]" -> "search[...]")
            import re
            action = re.sub(r'^\d+\.\s*', '', action)
            
            # Extract first line if multi-line
            if '\n' in action:
                action = action.split('\n')[0].strip()
            
            # Try to find exact match in valid_actions
            if action in valid_actions:
                if self.verbose:
                    print(f"    ✓ Selected: {action[:80]}")
                return action
            
            # Try case-insensitive match
            action_lower = action.lower()
            for valid_action in valid_actions:
                if valid_action.lower() == action_lower:
                    if self.verbose:
                        print(f"    ✓ Case-matched: {valid_action[:80]}")
                    return valid_action
            
            # Try partial match for long action strings
            for valid_action in valid_actions:
                if len(action) > 20 and action[:20].lower() in valid_action.lower():
                    if self.verbose:
                        print(f"    ~ Partial match: {valid_action[:80]}")
                    return valid_action
                if len(valid_action) > 20 and valid_action[:20].lower() in action.lower():
                    if self.verbose:
                        print(f"    ~ Partial match: {valid_action[:80]}")
                    return valid_action
            
            # Fallback: use first valid action (safer than failing)
            if self.verbose:
                print(f"    ✗ No match for '{action[:80]}', using first valid action")
            return valid_actions[0]
            
        except Exception as e:
            print(f"    ERROR in get_action: {e}")
            # Fallback: return first valid action
            return valid_actions[0] if valid_actions else 'search[product]'
    
    def collect_episode(self, env, idx, max_steps=20, verbose=False):
        """
        Collect one trajectory using LATS.
        
        Returns trajectory in same format as collect_trajectories_gpt_sampled.py
        """
        obs, info = env.reset(idx)
        goal = info['goal']
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"LATS Task {idx}")
            print(f"Goal: {goal}")
            print(f"{'='*80}")
        
        # Initialize trajectory in EXACT format from collect_trajectories_gpt_sampled.py
        trajectory = {
            'idx': idx,
            'goal': goal,
            'steps': [],
            'final_reward': 0,
            'success': False,
            'num_steps': 0,
            'model': self.model
        }
        
        # Run LATS search to get best action sequence
        result = self.search(env, idx, iterations=10, simulations=2, exploration_constant=1.4)
        
        # Extract steps from LATS result
        if result['trajectory']:
            for i, step_data in enumerate(result['trajectory']):
                trajectory['steps'].append({
                    'step': i,
                    'observation': step_data['observation'],
                    'valid_actions': step_data.get('valid_actions', []),
                    'action': step_data['action'],
                    'raw_response': step_data['action']  # LATS doesn't have separate raw response
                })
        
        trajectory['final_reward'] = result['final_reward']
        trajectory['success'] = result['success']
        trajectory['num_steps'] = result['num_steps']
        
        if verbose:
            status = '✓ SUCCESS' if trajectory['success'] else '✗ FAILURE'
            print(f"\nTask {idx}: {status}")
            print(f"  Reward: {trajectory['final_reward']}")
            print(f"  Steps: {trajectory['num_steps']}")
        
        return trajectory
    
    def search(self, env, task_idx: int, iterations: int = 10, simulations: int = 2,
               exploration_constant: float = 1.4) -> Dict:
        """
        Run LATS search for a WebShop task.
        
        Returns:
            Dictionary with trajectory data compatible with collect_trajectories_gpt_sampled.py format
        """
        # Reset environment - WebEnv returns (obs, info)
        obs, info = env.reset(task_idx)
        goal = info['goal']
        valid_actions = info['valid']  # WebEnv stores valid actions in info['valid']
        
        if self.verbose:
            print(f"  Starting LATS with {iterations} iterations, {simulations} simulations")
            print(f"  Initial valid actions: {len(valid_actions)}")
        
        # Initialize root node
        root = TreeNode(state={
            'observation': obs,
            'action': None,
            'reward': 0.0,
            'valid_actions': valid_actions
        })
        
        # MCTS iterations
        for iteration in range(iterations):
            if self.verbose and iteration % 3 == 0:
                print(f"  Iteration {iteration + 1}/{iterations}")
            
            # 1. Selection
            node = self._select(root, exploration_constant)
            
            # 2. Expansion
            if not node.is_terminal:
                self._expand(node, env, task_idx, goal, simulations)
            
            # 3. Backpropagation
            self._backpropagate(node, node.value)
            
            # Check if we found a successful trajectory
            if self._has_successful_child(root):
                if self.verbose:
                    print(f"  ✓ Found successful trajectory at iteration {iteration + 1}")
                break
        
        # Extract best trajectory
        best_node = self._get_best_terminal_node(root)
        trajectory_steps = self._collect_trajectory(best_node)
        
        # Format result in compatible structure
        result = {
            'task_idx': task_idx,
            'goal': goal,
            'success': best_node.reward >= 10.0,  # WebShop uses 0-10 scale, 10=success
            'final_reward': best_node.reward * 10,  # Scale to 0-100 for compatibility (10 * 10 = 100)
            'num_steps': len(trajectory_steps),
            'trajectory': trajectory_steps,
            'api_calls': self.total_api_calls
        }
        
        return result
    
    def _select(self, node: TreeNode, exploration_constant: float) -> TreeNode:
        """Select a leaf node using UCT"""
        while node.children:
            # If all children are terminal, return current node
            if all(child.is_terminal for child in node.children):
                return node
            
            # Select best non-terminal child
            non_terminal = [c for c in node.children if not c.is_terminal]
            if non_terminal:
                # Prefer unvisited children
                unvisited = [c for c in non_terminal if c.visits == 0]
                if unvisited:
                    node = random.choice(unvisited)
                else:
                    node = max(non_terminal, key=lambda c: c.uct(exploration_constant))
            else:
                return node
        
        return node
    
    def _expand(self, node: TreeNode, env, task_idx: int, goal: str, simulations: int):
        """Expand node by generating child states"""
        # Reconstruct state by replaying actions
        env.reset(task_idx)
        trajectory = []
        current = node
        path = []
        
        while current.parent is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        
        # Replay actions to reach this state
        last_info = None
        for n in path:
            if n.state['action']:
                trajectory.append(f"Action: {n.state['action']}")
                step_result = env.step(n.state['action'])
                if isinstance(step_result, tuple):
                    obs_str, reward, done, last_info = step_result if len(step_result) == 4 else (step_result[0], 0, False, {})
                trajectory.append(f"Observation: {obs_str[:200]}")
        
        # Get current state
        current_obs = node.state['observation']
        
        # Get valid actions from last step info
        valid_actions = last_info.get('valid', []) if last_info else node.state.get('valid_actions', [])
        
        if not valid_actions:
            # No valid actions, mark as terminal
            node.is_terminal = True
            return
        
        # Generate multiple actions (simulations)
        for _ in range(simulations):
            action = self.get_action(current_obs, goal, valid_actions, trajectory)
            
            # Take action in environment
            step_result = env.step(action)
            
            # WebEnv step returns: observation, reward, done, info
            if isinstance(step_result, tuple):
                if len(step_result) == 4:
                    obs_str, reward, done, info = step_result
                elif len(step_result) == 3:
                    obs_str, reward, done = step_result
                    info = {}
                else:
                    obs_str = step_result[0]
                    reward = 0.0
                    done = False
                    info = {}
            else:
                obs_str = step_result
                reward = 0.0
                done = False
                info = {}
            
            # Get valid actions for next state
            next_valid_actions = info.get('valid', [])
            
            # Create child node
            child = TreeNode(
                state={
                    'observation': obs_str,
                    'action': action,
                    'reward': reward,
                    'valid_actions': next_valid_actions
                },
                parent=node
            )
            
            child.is_terminal = done or reward >= 10.0  # WebShop uses 0-10 scale
            child.reward = reward  # Store raw 0-10 scale reward
            child.value = reward  # Use raw reward for value
            
            node.children.append(child)
            
            # Reset for next simulation
            if _ < simulations - 1:
                env.reset(task_idx)
                for n in path:
                    if n.state['action']:
                        env.step(n.state['action'])
    
    def _backpropagate(self, node: TreeNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _has_successful_child(self, node: TreeNode) -> bool:
        """Check if any descendant has reward >= 10.0 (WebShop success)"""
        if node.reward >= 10.0:
            return True
        for child in node.children:
            if self._has_successful_child(child):
                return True
        return False
    
    def _get_best_terminal_node(self, root: TreeNode) -> TreeNode:
        """Get best terminal node from tree"""
        terminals = []
        self._collect_terminals(root, terminals)
        
        if not terminals:
            return root
        
        # Prefer successful terminals (reward >= 10.0 in WebShop's 0-10 scale)
        successful = [n for n in terminals if n.reward >= 10.0]
        if successful:
            return max(successful, key=lambda n: n.reward)
        
        # Otherwise, return highest reward
        return max(terminals, key=lambda n: n.reward)
    
    def _collect_terminals(self, node: TreeNode, terminals: List[TreeNode]):
        """Collect all terminal nodes"""
        if node.is_terminal or not node.children:
            terminals.append(node)
        else:
            for child in node.children:
                self._collect_terminals(child, terminals)
    
    def _collect_trajectory(self, node: TreeNode) -> List[Dict]:
        """Collect trajectory from root to node in format compatible with GPT collector"""
        trajectory = []
        path = []
        
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        
        for i, n in enumerate(path[1:]):  # Skip root
            trajectory.append({
                'action': n.state['action'],
                'observation': n.state['observation'][:1000],  # Truncate like GPT collector
                'valid_actions': n.state.get('valid_actions', []),
                'reward': n.state['reward']
            })
        
        return trajectory


def main():
    parser = argparse.ArgumentParser(description='LATS for WebShop')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model: gpt-4o, gpt-4o-mini, gemini-2.0-flash-exp, or local path')
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='Number of tasks')
    parser.add_argument('--start_task', type=int, default=0,
                        help='Starting task index')
    parser.add_argument('--iterations', type=int, default=10,
                        help='LATS iterations per task')
    parser.add_argument('--simulations', type=int, default=2,
                        help='Simulations per expansion')
    parser.add_argument('--output', type=str, default='./lats_trajectories.json',
                        help='Output file path')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 80)
    print("LATS WebShop - GPT Collector Compatible")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.num_tasks} (starting from {args.start_task})")
    print(f"Iterations: {args.iterations}, Simulations: {args.simulations}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Initialize environment using WebEnv (same as GPT collector)
    print("\nInitializing WebShop...")
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split='train')
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    print("✓ WebShop ready")
    
    # Initialize agent
    print(f"\nInitializing LATS agent with {args.model}...")
    agent = LATSAgent(model=args.model, verbose=args.verbose)
    print("✓ Agent ready")
    
    # Collect trajectories
    print(f"\nCollecting {args.num_tasks} trajectories...")
    successful_trajectories = []
    all_trajectories = []
    
    for i in range(args.start_task, args.start_task + args.num_tasks):
        try:
            trajectory = agent.collect_episode(env, i, max_steps=20, verbose=args.verbose)
            all_trajectories.append(trajectory)
            
            if trajectory['success']:
                successful_trajectories.append(trajectory)
                
        except Exception as e:
            print(f"\nTask {i}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary (same format as GPT collector)
    expert_successes = successful_trajectories
    total_tasks = len(all_trajectories)
    
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    if total_tasks > 0:
        print(f"Total tasks: {total_tasks}")
        print(f"Expert success: {len(expert_successes)} ({len(expert_successes)/total_tasks*100:.1f}%)")
        print(f"Expert failure: {total_tasks - len(expert_successes)} ({(total_tasks - len(expert_successes))/total_tasks*100:.1f}%)")
        
        if expert_successes:
            avg_reward = sum(t['final_reward'] for t in expert_successes) / len(expert_successes)
            avg_steps = sum(t['num_steps'] for t in expert_successes) / len(expert_successes)
            print(f"\n--- Successful Trajectories Stats ---")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.1f}")
        
        total_calls = agent.total_api_calls
        print(f"\n--- Performance ---")
        print(f"Total API calls: {total_calls}")
        print(f"Avg API calls per task: {total_calls/total_tasks:.1f}")
        
        # Estimate cost
        if args.model == 'gpt-4o':
            cost_per_call = 0.006
        elif args.model == 'gpt-4o-mini':
            cost_per_call = 0.0002
        else:
            cost_per_call = 0.0
        
        total_cost = total_calls * cost_per_call
        print(f"\n--- Cost ---")
        print(f"Estimated cost: ${total_cost:.4f}")
        print(f"Cost per task: ${total_cost/total_tasks:.4f}")
        print(f"Projected for 3000 tasks: ${total_cost * 3000 / total_tasks:.2f}")
    else:
        print("No tasks completed")
    
    print("=" * 80)
    
    # Save in same format as collect_trajectories_gpt_sampled.py
    from datetime import datetime
    output_data = {
        'metadata': {
            'model': args.model,
            'method': 'LATS',
            'collection_date': datetime.now().isoformat(),
            'num_tasks': total_tasks,
            'iterations': args.iterations,
            'simulations': args.simulations,
        },
        'summary': {
            'total_tasks': total_tasks,
            'expert_successes': len(expert_successes),
            'expert_failures': total_tasks - len(expert_successes),
            'expert_success_rate': (len(expert_successes) / total_tasks * 100) if total_tasks > 0 else 0,
        },
        'successful_trajectories': successful_trajectories,
        'all_trajectories': all_trajectories,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    if expert_successes:
        print(f"\n✓ {len(expert_successes)} VALID EXPERT TRAJECTORIES COLLECTED")
        print(f"\nCompatible with collect_trajectories_gpt_sampled.py format!")


if __name__ == '__main__':
    main()