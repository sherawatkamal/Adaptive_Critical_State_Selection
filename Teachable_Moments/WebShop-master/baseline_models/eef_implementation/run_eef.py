#!/usr/bin/env python3
"""
Run EEF with ACSS on WebShop Tasks
Main entry point for EEF experimentation
"""
import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, './eef_implementation')

from env import WebEnv
from train_choice_il import tokenizer, data_collator, process
from models.bert import BertModelForWebshop, BertConfigForWebshop

# Import EEF components
from eef_pipeline import run_eef_pipeline
from acss import ACSS


def create_webshop_env_args():
    """
    Create WebShop environment arguments with automatic defaults for missing attributes
    """
    class EnvArgs:
        def __getattribute__(self, name):
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                # Provide sensible defaults for any missing attribute
                print(f"Warning: Missing attribute '{name}', using default")
                if 'reward' in name or 'click' in name or 'human' in name:
                    return False
                elif 'num' in name or 'prev' in name:
                    return 0
                elif 'limit' in name:
                    return 100
                elif 'server' in name:
                    return "http://localhost:3000"
                else:
                    return None
        
        def __init__(self):
            self.state_format = 'text'
            self.num = None
            self.get_image = 0
            self.human_goals = 0
            self.num_prev_obs = 0
            self.num_prev_actions = 0
            self.step_limit = 100
            self.click_item_name = False
            self.harsh_reward = False
            self.extra_search_path = ""
            self.server = "http://localhost:3000"
            self.session = None
    
    return EnvArgs()


class AgentWrapper:
    """Wrapper to make model compatible with EEF simulator"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_state(self, obs, info):
        """Build state representation"""
        class State:
            def __init__(self, obs_str, goal_str):
                self.obs_str = obs_str
                self.goal_str = goal_str
        
        return State(obs.replace('\n', '[SEP]'), info.get('goal', ''))
    
    def act(self, states, valid_acts_list, method='greedy', eps=0.0):
        """Select action using the model"""
        # Use the predict function from train_choice_il
        state = states[0]
        valid_acts = valid_acts_list[0]
        
        # Build observation string
        obs = state.obs_str.replace('[SEP]', '\n')
        
        # Create info dict
        info = {
            'valid': valid_acts,
            'goal': state.goal_str
        }
        
        try:
            # Use model to predict
            action = predict_action(obs, info, self.model, 
                                   softmax=(method=='softmax'))
            return [action], [[]], [torch.tensor(0.0)]
        except Exception as e:
            print(f"Error in act: {e}")
            # Fallback to random
            import random
            action = random.choice(valid_acts) if valid_acts else 'click[back to search]'
            return [action], [[]], [torch.tensor(0.0)]


def predict_action(obs, info, model, softmax=False):
    """Predict action using the IL model"""
    valid_acts = info['valid']
    
    # Handle search actions
    if valid_acts and valid_acts[0].startswith('search['):
        return valid_acts[-1] if valid_acts else 'search[query]'
    
    # Process state and actions
    state_encodings = tokenizer(process(obs), max_length=512, 
                                truncation=True, padding='max_length')
    action_encodings = tokenizer(list(map(process, valid_acts)), 
                                 max_length=512, truncation=True, 
                                 padding='max_length')
    
    # Create batch
    batch = {
        'state_input_ids': state_encodings['input_ids'],
        'state_attention_mask': state_encodings['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': [0.0] * 512,  # Dummy images
        'labels': 0
    }
    batch = data_collator([batch])
    batch = {k: v.cuda() if torch.cuda.is_available() else v 
             for k, v in batch.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**batch)
        if softmax:
            idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[0].item()
        else:
            idx = outputs.logits[0].argmax(0).item()
    
    return valid_acts[idx] if idx < len(valid_acts) else valid_acts[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run EEF with ACSS on WebShop")
    
    # Model and environment
    parser.add_argument("--model_path", type=str, 
                       default="./ckpts/web_click/epoch_9/model.pth",
                       help="Path to pretrained IL model")
    parser.add_argument("--failure_data", type=str,
                       default="./trajectory_data/failure_trajectories_20251106_134924.json",  # FIXED: Path updated
                       help="Path to failure trajectories JSON")
    
    # EEF parameters
    parser.add_argument("--acss_strategy", type=str, default="baseline",
                       choices=['baseline', 'entropy'],
                       help="ACSS state selection strategy: 'baseline' (skip-length) or 'entropy' (Shannon Entropy)")
    parser.add_argument("--M", type=int, default=5,
                       help="Number of states to select per trajectory (parameter M from paper)")
    parser.add_argument("--simulation_budget", type=int, default=100,
                       help="Maximum number of simulations to run")
    parser.add_argument("--max_failures", type=int, default=None,
                       help="Maximum number of failure trajectories to load (default: None = use ALL)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./eef_output",
                       help="Directory for EEF outputs")
    
    # Evaluation
    parser.add_argument("--run_evaluation", action='store_true',
                       help="Run evaluation after EEF")
    parser.add_argument("--num_eval_tasks", type=int, default=20,
                       help="Number of tasks for evaluation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("EEF WITH ACSS - EXPLORING EXPERT FAILURES")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Failure Data: {args.failure_data}")
    print(f"ACSS Strategy: {args.acss_strategy}")
    print(f"M (states per trajectory): {args.M}")
    print(f"Simulation Budget: {args.simulation_budget}")
    print(f"Max Failures: {args.max_failures if args.max_failures else 'ALL'}")
    print("="*80)
    
    # Check files exist
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.failure_data):
        print(f"ERROR: Failure data not found at {args.failure_data}")
        return
    
    # Setup environment
    print("\n[1/5] Setting up WebShop environment...")
    env_args = create_webshop_env_args()
    env_args.get_image = 0
    env_args.human_goals = 0
    env_args.extra_search_path = ""
    env = WebEnv(env_args, split='test')
    print("✓ Environment loaded")
    
    # Load model
    print("\n[2/5] Loading pretrained IL model...")
    config = BertConfigForWebshop(image=False)
    model = BertModelForWebshop(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device), 
                         strict=False)
    model.eval()
    print(f"✓ Model loaded from {args.model_path}")
    print(f"✓ Device: {device}")
    
    # Create agent wrapper
    print("\n[3/5] Creating agent wrapper...")
    agent = AgentWrapper(model, tokenizer)
    print("✓ Agent wrapper created")
    
    # Run EEF pipeline
    print("\n[4/5] Running EEF pipeline...")
    print("-"*80)
    
    training_samples, pipeline = run_eef_pipeline(
        env=env,
        agent=agent,
        failure_json_path=args.failure_data,
        output_dir=args.output_dir,
        acss_strategy=args.acss_strategy,
        M=args.M,
        simulation_budget=args.simulation_budget,
        max_failures=args.max_failures,
        success_threshold=7.0
    )
    
    print("-"*80)
    print("✓ EEF pipeline completed")
    
    # Summary
    print("\n[5/5] Summary")
    print("="*80)
    print(f"  Failures analyzed:        {pipeline.stats['total_failures_loaded']}")
    print(f"  Critical states selected: {pipeline.stats['total_states_selected']}")
    print(f"  Simulations executed:     {pipeline.stats['total_simulations_run']}")
    print(f"  Successful recoveries:    {pipeline.stats['successful_recoveries']}")
    
    if pipeline.stats['total_simulations_run'] > 0:
        success_rate = (pipeline.stats['successful_recoveries'] / 
                       pipeline.stats['total_simulations_run'])
        print(f"  Recovery success rate:    {success_rate:.1%}")
    
    print(f"  Training samples created: {pipeline.stats['training_samples_created']}")
    print(f"\n  Output directory: {args.output_dir}")
    print(f"    - beneficial_training_samples.json")
    print(f"    - eef_statistics.json")
    print("="*80)
    
    # Optional evaluation
    if args.run_evaluation:
        print(f"\nRunning evaluation on {args.num_eval_tasks} tasks...")
        print("(This will take a few minutes)")
        
        scores = []
        for i in range(args.num_eval_tasks):
            obs, info = env.reset(i)
            total_reward = 0
            
            for step in range(100):
                valid_acts = info.get('valid', [])
                if not valid_acts:
                    break
                
                action = predict_action(obs, info, model, softmax=True)
                obs, reward, done, info = env.step(action)
                total_reward = reward
                
                if done:
                    break
            
            scores.append(total_reward)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{args.num_eval_tasks} tasks")
        
        avg_score = sum(scores) / len(scores)
        success_rate = len([s for s in scores if s >= 1.0]) / len(scores)
        
        print(f"\nEvaluation Results:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Success Rate:  {success_rate:.1%}")
        
        # Save evaluation results
        eval_results = {
            'avg_score': avg_score,
            'success_rate': success_rate,
            'scores': scores
        }
        eval_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"  Saved to: {eval_path}")
    
    print("\n✓ EEF pipeline completed successfully!")
    print(f"✓ Next step: Use beneficial_training_samples.json for fine-tuning")


if __name__ == "__main__":
    main()