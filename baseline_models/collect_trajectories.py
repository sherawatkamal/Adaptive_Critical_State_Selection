"""
Trajectory Collection Script for ACSS Pipeline
Collects trajectories with logits for entropy computation
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Import WebShop environment
from train_rl import parse_args as webenv_args
from env import WebEnv

# Import model components
from train_choice_il import (
    tokenizer, process, process_goal, data_collator,
    BertConfigForWebshop, BertModelForWebshop
)
from transformers import BartForConditionalGeneration, BartTokenizer

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def compute_entropy(logits):
    """Compute Shannon entropy from logits."""
    probs = F.softmax(logits, dim=0)
    log_probs = F.log_softmax(logits, dim=0)
    entropy = -torch.sum(probs * log_probs).item()
    return entropy, probs.cpu().numpy().tolist()


def bart_predict(input_text, model, skip_special_tokens=True, **kwargs):
    """Generate search query using BART model."""
    input_ids = bart_tokenizer(input_text)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    output = model.generate(input_ids, max_length=512, **kwargs)
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)


def predict_with_logits(obs, info, model, bart_model=None, softmax=False):
    """
    Predict action and return logits for entropy computation.
    
    Returns:
        action: selected action string
        logits: raw logits (None for search states)
        entropy: entropy value (None for search states)
        probs: probability distribution (None for search states)
        is_search: whether this was a search state
    """
    valid_acts = info['valid']
    
    # Search state - use BART, no entropy computation
    if valid_acts[0].startswith('search['):
        if bart_model is None:
            action = valid_acts[-1]
        else:
            goal = process_goal(obs)
            query = bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
            query = query[0]  # Use top-1
            action = f'search[{query}]'
        return action, None, None, None, True
    
    # Choice state - use BERT, compute entropy
    state_encodings = tokenizer(
        process(obs), max_length=512, truncation=True, padding='max_length'
    )
    action_encodings = tokenizer(
        list(map(process, valid_acts)), max_length=512, truncation=True, padding='max_length'
    )
    
    batch = {
        'state_input_ids': state_encodings['input_ids'],
        'state_attention_mask': state_encodings['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': info['image_feat'].tolist(),
        'labels': 0
    }
    batch = data_collator([batch])
    batch = {k: v.cuda() for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits[0]  # Shape: (num_actions,)
    entropy, probs = compute_entropy(logits)
    
    # Select action
    if softmax:
        idx = torch.multinomial(F.softmax(logits, dim=0), 1)[0].item()
    else:
        idx = logits.argmax(0).item()
    
    action = valid_acts[idx]
    
    return action, logits.cpu().numpy().tolist(), entropy, probs, False


def collect_episode(model, env, idx, bart_model=None, softmax=False, verbose=False):
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
    
    for step in range(100):
        action, logits, entropy, probs, is_search = predict_with_logits(
            obs, info, model, bart_model=bart_model, softmax=softmax
        )
        
        # Store step data
        step_data = {
            'step': step,
            'observation': obs[:500],  # Truncate for storage
            'valid_actions': info['valid'],
            'action': action,
            'is_search': is_search,
            'logits': logits,
            'entropy': entropy,
            'probs': probs
        }
        trajectory['steps'].append(step_data)
        
        if verbose:
            entropy_str = f"{entropy:.3f}" if entropy is not None else "N/A (search)"
            print(f"  Step {step}: {action[:50]}... | Entropy: {entropy_str}")
        
        # Take action
        obs, reward, done, info = env.step(action)
        
        if done:
            trajectory['final_reward'] = reward * 10  # Scale 0-10 → 0-100 (paper scale)
            trajectory['success'] = (reward == 10.0)  # Perfect score in env's 0-10 scale
            trajectory['num_steps'] = step + 1
            break
    
    if verbose:
        print(f"  Result: {'SUCCESS' if trajectory['success'] else 'FAILURE'} | Reward: {trajectory['final_reward']}")
    
    return trajectory


def collect_until_n_failures(model, env, bart_model, target_failures, softmax=False, 
                             split='train', start_idx=0, max_episodes=5000, verbose=False):
    """Collect trajectories until we have target_failures failed trajectories."""
    
    failed_trajectories = []
    success_trajectories = []
    all_entropies = []
    
    idx = start_idx
    pbar = tqdm(total=target_failures, desc="Collecting failures")
    
    while len(failed_trajectories) < target_failures and idx < start_idx + max_episodes:
        traj = collect_episode(
            model, env, idx, bart_model=bart_model, 
            softmax=softmax, verbose=verbose
        )
        
        # Collect entropies from choice states
        for step in traj['steps']:
            if step['entropy'] is not None:
                all_entropies.append(step['entropy'])
        
        if traj['success']:
            success_trajectories.append(traj)
        else:
            failed_trajectories.append(traj)
            pbar.update(1)
        
        idx += 1
        
        if idx % 100 == 0:
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
            'target_failures': target_failures,
            'total_episodes': total_episodes,
            'softmax': softmax,
            'start_idx': start_idx
        },
        'failed_trajectories': failed_trajectories,
        'success_trajectories': success_trajectories,
        'summary': {
            'total_episodes': total_episodes,
            'num_failures': len(failed_trajectories),
            'num_successes': len(success_trajectories),
            'failure_rate': len(failed_trajectories) / total_episodes * 100,
            'success_rate': len(success_trajectories) / total_episodes * 100,
            'avg_entropy': np.mean(all_entropies) if all_entropies else 0
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Collect trajectories for ACSS pipeline")
    parser.add_argument("--model_path", type=str, 
                        default="./ckpts/web_click/epoch_9/model.pth",
                        help="Path to BERT choice model")
    parser.add_argument("--bart_path", type=str,
                        default='./ckpts/web_search/checkpoint-800',
                        help="Path to BART search model")
    parser.add_argument("--output_dir", type=str, default="./trajectories",
                        help="Directory to save trajectories")
    parser.add_argument("--target_failures", type=int, default=1000,
                        help="Number of failed trajectories to collect")
    parser.add_argument("--max_episodes", type=int, default=5000,
                        help="Maximum episodes to run (safety limit)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting task index")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Dataset split to use")
    parser.add_argument("--softmax", action="store_true",
                        help="Use softmax sampling instead of argmax")
    parser.add_argument("--verbose", action="store_true",
                        help="Print episode details")
    parser.add_argument("--mem", type=int, default=0,
                        help="Use memory (previous observations)")
    parser.add_argument("--image", action="store_true", default=True,
                        help="Use image features")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    print(f"\nTarget: Collect {args.target_failures} failed trajectories")
    
    # Setup environment
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split=args.split)
    print(f"Environment loaded (split={args.split})")
    
    # Memory settings
    if args.mem:
        env.env.num_prev_obs = 1
        env.env.num_prev_actions = 5
        print("Memory enabled")
    else:
        env.env.num_prev_obs = 0
        env.env.num_prev_actions = 0
    
    # Load BART model
    bart_model = BartForConditionalGeneration.from_pretrained(args.bart_path)
    bart_model.cuda()
    bart_model.eval()
    print(f"BART model loaded: {args.bart_path}")
    
    # Load BERT model
    config = BertConfigForWebshop(image=args.image)
    model = BertModelForWebshop(config)
    model.cuda()
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.eval()
    print(f"BERT model loaded: {args.model_path}")
    
    # Collect trajectories until we have enough failures
    print(f"\nCollecting until {args.target_failures} failures...")
    results = collect_until_n_failures(
        model, env, bart_model,
        target_failures=args.target_failures,
        softmax=args.softmax,
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
    print(f"Avg entropy (choice states): {results['summary']['avg_entropy']:.3f}")
    
    # Save trajectories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save failed trajectories (main output for ACSS)
    failed_file = os.path.join(args.output_dir, f"trajectories_{args.split}_failed_{args.target_failures}.json")
    failed_data = {
        'metadata': results['metadata'],
        'trajectories': results['failed_trajectories'],
        'summary': results['summary']
    }
    with open(failed_file, 'w') as f:
        json.dump(failed_data, f, indent=2)
    print(f"\nSaved {len(results['failed_trajectories'])} failed trajectories to: {failed_file}")


if __name__ == "__main__":
    main()