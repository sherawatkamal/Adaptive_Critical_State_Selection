import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Value
import time
import traceback
from collections import defaultdict

# Set start method for CUDA compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


def compute_entropy(logits):
    """Compute Shannon entropy from logits."""
    probs = F.softmax(logits, dim=0)
    log_probs = F.log_softmax(logits, dim=0)
    entropy = -torch.sum(probs * log_probs).item()
    return entropy, probs.cpu().numpy().tolist()


def worker_init(gpu_id, worker_id, model_path, bart_path, use_image, split, num_prev_obs, num_prev_actions):
    """
    Initialize a worker with its own environment and models.
    Called once per worker process.
    """
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import here to avoid issues with multiprocessing
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    from train_choice_il import (
        tokenizer, process, process_goal, data_collator,
        BertConfigForWebshop, BertModelForWebshop
    )
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    # Setup environment
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split=split)
    env.env.num_prev_obs = num_prev_obs
    env.env.num_prev_actions = num_prev_actions
    
    # Load BART tokenizer and model
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
    bart_model.cuda()
    bart_model.eval()
    
    # Load BERT model
    config = BertConfigForWebshop(image=use_image)
    model = BertModelForWebshop(config)
    model.cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
    model.eval()
    
    # Store in global dict for this process
    return {
        'env': env,
        'model': model,
        'bart_model': bart_model,
        'bart_tokenizer': bart_tokenizer,
        'tokenizer': tokenizer,
        'process': process,
        'process_goal': process_goal,
        'data_collator': data_collator,
        'gpu_id': gpu_id,
        'worker_id': worker_id
    }


def bart_predict(input_text, model, tokenizer, skip_special_tokens=True, **kwargs):
    """Generate search query using BART model."""
    input_ids = tokenizer(input_text)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    output = model.generate(input_ids, max_length=512, **kwargs)
    return tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)


def predict_with_logits(obs, info, ctx, softmax=False):
    """
    Predict action and return logits for entropy computation.
    """
    valid_acts = info['valid']
    model = ctx['model']
    bart_model = ctx['bart_model']
    bart_tokenizer = ctx['bart_tokenizer']
    tokenizer = ctx['tokenizer']
    process = ctx['process']
    process_goal = ctx['process_goal']
    data_collator = ctx['data_collator']
    
    # Search state - use BART, no entropy computation
    if valid_acts[0].startswith('search['):
        goal = process_goal(obs)
        query = bart_predict(goal, bart_model, bart_tokenizer, num_return_sequences=5, num_beams=5)
        query = query[0]
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
    
    logits = outputs.logits[0]
    entropy, probs = compute_entropy(logits)
    
    if softmax:
        idx = torch.multinomial(F.softmax(logits, dim=0), 1)[0].item()
    else:
        idx = logits.argmax(0).item()
    
    action = valid_acts[idx]
    
    return action, logits.cpu().numpy().tolist(), entropy, probs, False


def collect_episode(ctx, idx, softmax=False):
    """Run one episode and collect full trajectory data."""
    env = ctx['env']
    
    obs, info = env.reset(idx)
    goal = info['goal']
    
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
            obs, info, ctx, softmax=softmax
        )
        
        step_data = {
            'step': step,
            'observation': obs[:500],
            'valid_actions': info['valid'],
            'action': action,
            'is_search': is_search,
            'logits': logits,
            'entropy': entropy,
            'probs': probs
        }
        trajectory['steps'].append(step_data)
        
        obs, reward, done, info = env.step(action)
        
        if done:
            trajectory['final_reward'] = reward * 10
            trajectory['success'] = (reward == 1.0)
            trajectory['num_steps'] = step + 1
            break
    
    return trajectory


def worker_process(worker_id, gpu_id, task_queue, result_queue, progress_dict, 
                   config, stop_flag):
    """
    Worker process that collects trajectories from assigned tasks.
    """
    try:
        # Initialize models and environment
        ctx = worker_init(
            gpu_id=gpu_id,
            worker_id=worker_id,
            model_path=config['model_path'],
            bart_path=config['bart_path'],
            use_image=config['use_image'],
            split=config['split'],
            num_prev_obs=config['num_prev_obs'],
            num_prev_actions=config['num_prev_actions']
        )
        
        print(f"Worker {worker_id} initialized on GPU {gpu_id}")
        
        failed_trajectories = []
        success_trajectories = []
        episodes_run = 0
        
        while not stop_flag.value:
            # Get next task
            try:
                task_idx = task_queue.get(timeout=1)
            except:
                continue
            
            if task_idx is None:  # Poison pill
                break
            
            # Collect episode
            try:
                traj = collect_episode(ctx, task_idx, softmax=config['softmax'])
                episodes_run += 1
                
                if traj['success']:
                    success_trajectories.append(traj)
                else:
                    failed_trajectories.append(traj)
                    # Report failure to main process
                    result_queue.put(('failure', traj, worker_id))
                
                # Update progress
                progress_dict[worker_id] = {
                    'episodes': episodes_run,
                    'failures': len(failed_trajectories),
                    'successes': len(success_trajectories)
                }
                
            except Exception as e:
                print(f"Worker {worker_id} error on task {task_idx}: {e}")
                traceback.print_exc()
                continue
        
        # Send final results
        result_queue.put(('done', {
            'worker_id': worker_id,
            'failed': failed_trajectories,
            'success': success_trajectories,
            'episodes': episodes_run
        }, worker_id))
        
    except Exception as e:
        print(f"Worker {worker_id} fatal error: {e}")
        traceback.print_exc()
        result_queue.put(('error', str(e), worker_id))


def collect_parallel(config):
    """
    Main function to coordinate parallel collection.
    """
    num_workers = config['num_workers']
    target_failures = config['target_failures']
    gpus = config['gpus']
    
    print(f"\n{'='*60}")
    print(f"PARALLEL TRAJECTORY COLLECTION")
    print(f"{'='*60}")
    print(f"Target failures: {target_failures}")
    print(f"Workers: {num_workers}")
    print(f"GPUs: {gpus}")
    print(f"Workers per GPU: {num_workers // len(gpus)}")
    print(f"{'='*60}\n")
    
    # Setup multiprocessing
    manager = Manager()
    task_queue = Queue(maxsize=1000)
    result_queue = Queue()
    progress_dict = manager.dict()
    stop_flag = Value('b', False)
    
    # Initialize progress tracking
    for i in range(num_workers):
        progress_dict[i] = {'episodes': 0, 'failures': 0, 'successes': 0}
    
    # Start workers
    workers = []
    for worker_id in range(num_workers):
        gpu_id = gpus[worker_id % len(gpus)]
        p = Process(
            target=worker_process,
            args=(worker_id, gpu_id, task_queue, result_queue, progress_dict, config, stop_flag)
        )
        p.start()
        workers.append(p)
        time.sleep(0.5)  # Stagger starts to avoid GPU memory contention
    
    print(f"Started {num_workers} workers")
    
    # Feed tasks and collect results
    all_failed = []
    all_success = []
    task_idx = config['start_idx']
    workers_done = 0
    
    # Pre-fill task queue
    for _ in range(min(num_workers * 10, config['max_episodes'])):
        task_queue.put(task_idx)
        task_idx += 1
    
    # Progress bar
    pbar = tqdm(total=target_failures, desc="Collecting failures")
    start_time = time.time()
    last_update = start_time
    
    try:
        while len(all_failed) < target_failures and workers_done < num_workers:
            # Check for results
            try:
                msg_type, data, worker_id = result_queue.get(timeout=0.5)
                
                if msg_type == 'failure':
                    all_failed.append(data)
                    pbar.update(1)
                    
                    # Add more tasks if not done
                    if len(all_failed) < target_failures and task_idx < config['start_idx'] + config['max_episodes']:
                        task_queue.put(task_idx)
                        task_idx += 1
                
                elif msg_type == 'done':
                    workers_done += 1
                    all_success.extend(data['success'])
                    print(f"\nWorker {worker_id} finished: {data['episodes']} episodes")
                
                elif msg_type == 'error':
                    print(f"\nWorker {worker_id} error: {data}")
                    workers_done += 1
                    
            except:
                pass
            
            # Update progress display
            current_time = time.time()
            if current_time - last_update > 2:
                elapsed = current_time - start_time
                total_episodes = sum(p['episodes'] for p in progress_dict.values())
                total_failures = len(all_failed)
                
                if total_failures > 0:
                    rate = total_failures / elapsed
                    eta = (target_failures - total_failures) / rate if rate > 0 else 0
                    pbar.set_postfix({
                        'episodes': total_episodes,
                        'rate': f'{rate:.1f}/s',
                        'ETA': f'{eta/60:.1f}min'
                    })
                
                last_update = current_time
                
                # Checkpoint every 500 failures
                if len(all_failed) > 0 and len(all_failed) % 500 == 0:
                    save_checkpoint(all_failed, config['output_dir'], len(all_failed))
        
        # Signal workers to stop
        stop_flag.value = True
        
        # Send poison pills
        for _ in range(num_workers):
            task_queue.put(None)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        stop_flag.value = True
    
    finally:
        pbar.close()
        
        # Wait for workers to finish
        print("Waiting for workers to finish...")
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        # Collect any remaining results
        while not result_queue.empty():
            try:
                msg_type, data, worker_id = result_queue.get_nowait()
                if msg_type == 'failure':
                    all_failed.append(data)
                elif msg_type == 'done':
                    all_success.extend(data.get('success', []))
            except:
                break
    
    # Calculate final statistics
    elapsed = time.time() - start_time
    total_episodes = len(all_failed) + len(all_success)
    
    # Compute entropy statistics
    all_entropies = []
    for traj in all_failed + all_success:
        for step in traj.get('steps', []):
            if step.get('entropy') is not None:
                all_entropies.append(step['entropy'])
    
    results = {
        'metadata': {
            'collection_date': datetime.now().isoformat(),
            'split': config['split'],
            'target_failures': target_failures,
            'total_episodes': total_episodes,
            'num_workers': num_workers,
            'gpus': gpus,
            'elapsed_seconds': elapsed,
            'softmax': config['softmax'],
            'start_idx': config['start_idx']
        },
        'failed_trajectories': all_failed,
        'success_trajectories': all_success,
        'summary': {
            'total_episodes': total_episodes,
            'num_failures': len(all_failed),
            'num_successes': len(all_success),
            'failure_rate': len(all_failed) / total_episodes * 100 if total_episodes > 0 else 0,
            'success_rate': len(all_success) / total_episodes * 100 if total_episodes > 0 else 0,
            'avg_entropy': np.mean(all_entropies) if all_entropies else 0,
            'collection_time_minutes': elapsed / 60,
            'trajectories_per_second': total_episodes / elapsed if elapsed > 0 else 0
        }
    }
    
    return results


def save_checkpoint(trajectories, output_dir, count):
    """Save intermediate checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{count}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump({'trajectories': trajectories, 'count': count}, f)
    print(f"\n  Checkpoint saved: {checkpoint_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel trajectory collection for ACSS")
    
    # Parallelization settings
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Total number of worker processes")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--workers_per_gpu", type=int, default=None,
                        help="Workers per GPU (overrides num_workers)")
    
    # Collection settings
    parser.add_argument("--target_failures", type=int, default=5000,
                        help="Number of failed trajectories to collect")
    parser.add_argument("--max_episodes", type=int, default=50000,
                        help="Maximum episodes to run")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting task index")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--softmax", action="store_true",
                        help="Use softmax sampling instead of argmax")
    
    # Model paths
    parser.add_argument("--model_path", type=str,
                        default="./ckpts/web_click/epoch_9/model.pth")
    parser.add_argument("--bart_path", type=str,
                        default="./ckpts/web_search/checkpoint-800")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./trajectories")
    
    # Environment settings
    parser.add_argument("--mem", type=int, default=0,
                        help="Use memory (previous observations)")
    parser.add_argument("--image", action="store_true", default=True,
                        help="Use image features")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse GPU list
    gpus = [int(g) for g in args.gpus.split(',')]
    
    # Calculate workers
    if args.workers_per_gpu:
        num_workers = len(gpus) * args.workers_per_gpu
    else:
        num_workers = args.num_workers
    
    # Build config
    config = {
        'num_workers': num_workers,
        'gpus': gpus,
        'target_failures': args.target_failures,
        'max_episodes': args.max_episodes,
        'start_idx': args.start_idx,
        'split': args.split,
        'softmax': args.softmax,
        'model_path': args.model_path,
        'bart_path': args.bart_path,
        'output_dir': args.output_dir,
        'use_image': args.image,
        'num_prev_obs': 1 if args.mem else 0,
        'num_prev_actions': 5 if args.mem else 0,
    }
    
    # Run parallel collection
    results = collect_parallel(config)
    
    # Print summary
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Total episodes:     {results['summary']['total_episodes']}")
    print(f"Failed trajectories: {results['summary']['num_failures']}")
    print(f"Success trajectories: {results['summary']['num_successes']}")
    print(f"Failure rate:       {results['summary']['failure_rate']:.1f}%")
    print(f"Success rate:       {results['summary']['success_rate']:.1f}%")
    print(f"Avg entropy:        {results['summary']['avg_entropy']:.3f}")
    print(f"Collection time:    {results['summary']['collection_time_minutes']:.1f} minutes")
    print(f"Speed:              {results['summary']['trajectories_per_second']:.2f} traj/sec")
    print("="*60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save failed trajectories
    failed_file = os.path.join(args.output_dir, f"failed_trajectories_{timestamp}.json")
    failed_data = {
        'metadata': results['metadata'],
        'trajectories': results['failed_trajectories'],
        'summary': results['summary']
    }
    with open(failed_file, 'w') as f:
        json.dump(failed_data, f)
    print(f"\nSaved {len(results['failed_trajectories'])} failed trajectories to: {failed_file}")
    
    # Save success trajectories
    if results['success_trajectories']:
        success_file = os.path.join(args.output_dir, f"success_trajectories_{timestamp}.json")
        with open(success_file, 'w') as f:
            json.dump(results['success_trajectories'], f)
        print(f"Saved {len(results['success_trajectories'])} success trajectories to: {success_file}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"collection_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'metadata': results['metadata'],
            'summary': results['summary']
        }, f, indent=2)
    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()