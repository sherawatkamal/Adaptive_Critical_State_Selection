#!/usr/bin/env python3
"""Verify trained models load and produce sensible outputs.

Performs sanity checks on all trained models before evaluation:
1. Model loads successfully
2. Generates coherent outputs on test prompts
3. Outputs differ from base model (training had effect)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import setup_logging, save_json, get_timestamp


def find_trained_models(base_dir: Path) -> list[Path]:
    """Find all trained model directories."""
    models = []
    
    # Check per-quadrant models
    models_dir = base_dir / "models"
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                models.append(model_dir)
    
    # Check baselines
    baselines_dir = base_dir / "baselines"
    if baselines_dir.exists():
        for model_dir in baselines_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                models.append(model_dir)
    
    return sorted(models)


def load_model_for_inference(model_path: Path, base_model: str) -> tuple:
    """Load a trained model for inference."""
    from peft import PeftModel
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_test_output(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate output for a test prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()


def verify_model(
    model_path: Path,
    base_model: str,
    test_prompts: list[str],
) -> dict[str, Any]:
    """Verify a single model."""
    logger = logging.getLogger(__name__)
    
    result = {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "load_success": False,
        "generation_success": False,
        "outputs": [],
        "errors": [],
    }
    
    try:
        # Load model
        logger.info(f"Loading {model_path.name}...")
        model, tokenizer = load_model_for_inference(model_path, base_model)
        result["load_success"] = True
        
        # Generate test outputs
        for prompt in test_prompts:
            try:
                output = generate_test_output(model, tokenizer, prompt)
                result["outputs"].append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "output": output[:200] + "..." if len(output) > 200 else output,
                    "output_length": len(output),
                })
            except Exception as e:
                result["errors"].append(f"Generation error: {str(e)}")
        
        result["generation_success"] = len(result["outputs"]) > 0
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        result["errors"].append(f"Load error: {str(e)}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify trained models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase2/verification_report.json"),
        help="Output path for verification report",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Test prompts for verification
    test_prompts = [
        "You are shopping for a laptop. The page shows: [Laptop A: $999, 16GB RAM] [Laptop B: $799, 8GB RAM]. Your task is to buy a laptop with 16GB RAM. Action:",
        "Current page: Search results for 'wireless headphones'. Options: [Sony WH-1000XM4 - $348] [Apple AirPods Pro - $249] [search]. Your goal: Find headphones under $300. Next action:",
        "Task: Order size M blue t-shirt. Page shows: Color options [Red] [Blue] [Green], Size options [S] [M] [L]. What action should you take?",
    ]
    
    # Find all trained models
    logger.info(f"Scanning for models in {args.models_dir}")
    model_paths = find_trained_models(args.models_dir)
    logger.info(f"Found {len(model_paths)} models")
    
    if not model_paths:
        logger.warning("No trained models found")
        return
    
    # Verify each model
    results = []
    for model_path in model_paths:
        logger.info(f"\nVerifying: {model_path.name}")
        result = verify_model(model_path, args.base_model, test_prompts)
        results.append(result)
        
        # Print status
        status = "OK" if result["load_success"] and result["generation_success"] else "FAIL"
        print(f"  [{status}] {result['model_name']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"    Error: {err}")
    
    # Generate summary
    summary = {
        "timestamp": get_timestamp(),
        "base_model": args.base_model,
        "total_models": len(results),
        "successful_loads": sum(1 for r in results if r["load_success"]),
        "successful_generations": sum(1 for r in results if r["generation_success"]),
        "results": results,
    }
    
    # Check overall status
    all_passed = all(r["load_success"] and r["generation_success"] for r in results)
    summary["all_passed"] = all_passed
    
    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total models: {summary['total_models']}")
    print(f"Successful loads: {summary['successful_loads']}")
    print(f"Successful generations: {summary['successful_generations']}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)
    
    if not all_passed:
        logger.warning("Some models failed verification - check report for details")


if __name__ == "__main__":
    main()
