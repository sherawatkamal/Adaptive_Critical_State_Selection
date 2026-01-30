#!/usr/bin/env python3
"""
Detailed evaluation of failure diagnosis model.
- Shows full trajectory context
- Extracts top-3 step predictions
- Saves results to JSON for analysis
"""

import json
import torch
import re
import argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path, base_model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load fine-tuned model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Loading adapter from: {model_path}")
    model = PeftModel.from_pretrained(base, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def extract_step(text):
    """Extract first step number mentioned."""
    match = re.search(r'step\s*(\d+)', text.lower())
    return int(match.group(1)) if match else -1


def extract_all_steps(text):
    """Extract all step numbers mentioned in order."""
    matches = re.findall(r'step\s*(\d+)', text.lower())
    # Return unique steps in order of appearance
    seen = set()
    result = []
    for m in matches:
        step = int(m)
        if step not in seen:
            seen.add(step)
            result.append(step)
    return result


def extract_trajectory_summary(user_content):
    """Extract key info from trajectory for display."""
    lines = user_content.split('\n')
    
    # Get goal
    goal = ""
    for line in lines:
        if "Goal:" in line or "Instruction:" in line:
            goal = line.strip()[:200]
            break
    
    # Get steps summary
    steps = []
    current_step = None
    for line in lines:
        if line.startswith("Step "):
            # Extract step number and action
            match = re.match(r'Step (\d+) \| action: (.+)', line)
            if match:
                step_num = int(match.group(1))
                action = match.group(2)[:100]  # Truncate long actions
                steps.append({"step": step_num, "action": action})
    
    return {
        "goal": goal,
        "steps": steps,
        "total_steps": len(steps)
    }


def generate_with_top_k_steps(model, tokenizer, messages, max_new_tokens=200):
    """Generate response and extract predictions."""
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Extract all mentioned steps as "top predictions"
    all_steps = extract_all_steps(response)
    primary_step = all_steps[0] if all_steps else -1
    
    return {
        "response": response,
        "primary_prediction": primary_step,
        "all_predictions": all_steps[:5],  # Top 5 mentioned steps
    }


def evaluate_detailed(model, tokenizer, test_file, output_dir, max_samples=None):
    """Run detailed evaluation and save results."""
    
    results = {
        "metadata": {
            "test_file": str(test_file),
            "timestamp": datetime.now().isoformat(),
            "max_samples": max_samples,
        },
        "summary": {},
        "examples": []
    }
    
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    with open(test_file) as f:
        lines = f.readlines()
    
    if max_samples:
        lines = lines[:max_samples]
    
    print(f"\nEvaluating {len(lines)} examples...\n")
    print("=" * 80)
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        messages = data['messages'][:-1]  # Remove assistant response
        true_response = data['messages'][-1]['content']
        true_step = extract_step(true_response)
        
        # Get user content for trajectory summary
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        trajectory_info = extract_trajectory_summary(user_content)
        
        # Generate prediction
        pred_result = generate_with_top_k_steps(model, tokenizer, messages)
        pred_step = pred_result['primary_prediction']
        all_preds = pred_result['all_predictions']
        
        # Check accuracy
        is_correct_top1 = (pred_step == true_step)
        is_correct_top3 = (true_step in all_preds[:3])
        
        correct_top1 += is_correct_top1
        correct_top3 += is_correct_top3
        total += 1
        
        # Store example
        example = {
            "id": i + 1,
            "trajectory": trajectory_info,
            "true_step": true_step,
            "true_response": true_response,
            "predicted_step": pred_step,
            "top_predictions": all_preds[:3],
            "model_response": pred_result['response'],
            "correct_top1": is_correct_top1,
            "correct_top3": is_correct_top3,
        }
        results['examples'].append(example)
        
        # Print detailed output for first 20 and all errors
        if i < 20 or not is_correct_top1:
            status = "✓" if is_correct_top1 else ("~" if is_correct_top3 else "✗")
            print(f"\n[Example {i+1}] {status}")
            print(f"  Goal: {trajectory_info['goal'][:100]}...")
            print(f"  Total steps: {trajectory_info['total_steps']}")
            print(f"  Steps summary:")
            for s in trajectory_info['steps'][:8]:  # Show first 8 steps
                marker = " <<<TRUE" if s['step'] == true_step else ""
                marker += " <<<PRED" if s['step'] == pred_step else ""
                print(f"    Step {s['step']}: {s['action'][:60]}{marker}")
            if len(trajectory_info['steps']) > 8:
                print(f"    ... ({len(trajectory_info['steps']) - 8} more steps)")
            print(f"  TRUE mistake: step {true_step}")
            print(f"  PRED mistake: step {pred_step} (top-3: {all_preds[:3]})")
            print(f"  Model says: {pred_result['response'][:150]}...")
            print("-" * 80)
    
    # Compute summary stats
    acc_top1 = correct_top1 / total * 100
    acc_top3 = correct_top3 / total * 100
    
    results['summary'] = {
        "total_examples": total,
        "correct_top1": correct_top1,
        "correct_top3": correct_top3,
        "accuracy_top1": acc_top1,
        "accuracy_top3": acc_top3,
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total examples: {total}")
    print(f"Top-1 Accuracy: {correct_top1}/{total} = {acc_top1:.1f}%")
    print(f"Top-3 Accuracy: {correct_top3}/{total} = {acc_top3:.1f}%")
    print("=" * 80)
    
    # Analyze error patterns
    print("\nERROR ANALYSIS:")
    
    # Step distribution in errors
    error_true_steps = [e['true_step'] for e in results['examples'] if not e['correct_top1']]
    error_pred_steps = [e['predicted_step'] for e in results['examples'] if not e['correct_top1']]
    
    if error_true_steps:
        from collections import Counter
        true_dist = Counter(error_true_steps)
        pred_dist = Counter(error_pred_steps)
        
        print(f"\nMost common TRUE steps in errors: {true_dist.most_common(5)}")
        print(f"Most common PREDICTED steps in errors: {pred_dist.most_common(5)}")
        
        # Compute average error distance
        errors_with_valid = [(e['true_step'], e['predicted_step']) 
                            for e in results['examples'] 
                            if not e['correct_top1'] and e['predicted_step'] >= 0]
        if errors_with_valid:
            avg_distance = sum(abs(t - p) for t, p in errors_with_valid) / len(errors_with_valid)
            print(f"Average step distance in errors: {avg_distance:.1f}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Also save a summary CSV for easy viewing
    summary_file = output_dir / f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(summary_file, 'w') as f:
        f.write("id,true_step,pred_step,top3_preds,correct_top1,correct_top3,goal_preview\n")
        for e in results['examples']:
            goal_preview = e['trajectory']['goal'][:50].replace(',', ';').replace('\n', ' ')
            f.write(f"{e['id']},{e['true_step']},{e['predicted_step']},\"{e['top_predictions']}\",{e['correct_top1']},{e['correct_top3']},\"{goal_preview}\"\n")
    print(f"Summary CSV saved to: {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Detailed evaluation of failure diagnosis model")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--test-file", required=True, help="Path to test JSONL file")
    parser.add_argument("--output-dir", default="./eval_results", help="Directory to save results")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path, args.base_model)
    results = evaluate_detailed(
        model, 
        tokenizer, 
        args.test_file, 
        args.output_dir,
        args.max_samples
    )
    
    return results


if __name__ == "__main__":
    main()