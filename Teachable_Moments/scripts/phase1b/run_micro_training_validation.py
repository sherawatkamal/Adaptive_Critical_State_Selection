#!/usr/bin/env python3
"""
Complete Micro-Training Validation for CPT.

This implements the full micro-training validation loop:
1. Load validation panel (N=200 snapshots, stratified by quadrant)
2. For each snapshot:
   a. Compute CPT scores (if not already computed)
   b. Create single-example training data in route-matched format
   c. Fine-tune base model with 1-2 LoRA gradient steps
   d. Evaluate micro-trained model via rollouts
   e. Record improvement: Δ_micro = success_after - success_before
3. Compute correlation between CPT ELP and actual improvement

Usage:
    python scripts/phase1b/run_micro_training_validation.py \
        --panel results/phase1/validation_panel.json \
        --base-model meta-llama/Llama-3-8B-Instruct \
        --output results/phase1b/cpt_validation_results.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from copy import deepcopy
import time

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging, set_seed, ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class MicroTrainingConfig:
    """Configuration for micro-training validation."""
    
    # Training
    n_gradient_steps: int = 2
    learning_rate: float = 1e-4
    
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    
    # Evaluation
    n_rollouts_per_task: int = 3
    n_validation_tasks: int = 10
    max_steps_per_episode: int = 15
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class MicroTrainingResult:
    """Result of micro-training on a single snapshot."""
    
    snapshot_id: str
    quadrant: str
    
    # CPT scores
    elp_net: float
    elp_raw: float
    route: str
    
    # Training metrics
    loss_before: float
    loss_after: float
    loss_delta: float
    
    # Rollout evaluation (the key metric)
    success_before: float  # Success rate before micro-training
    success_after: float   # Success rate after micro-training
    delta_micro: float     # success_after - success_before
    
    # Additional metrics
    n_evaluation_episodes: int = 0
    training_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0


class MicroTrainer:
    """
    Micro-training implementation with actual rollout evaluation.
    """
    
    def __init__(
        self,
        base_model_path: str,
        config: MicroTrainingConfig,
        env_factory: Callable,
        device: str = "cuda",
    ):
        self.base_model_path = base_model_path
        self.config = config
        self.env_factory = env_factory
        self.device = device
        
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _load_base_model(self):
        """Lazily load the base model."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading base model from {self.base_model_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self._loaded = True
        logger.info("Base model loaded")
    
    def _create_training_example(
        self,
        snapshot: dict,
        supervision_format: str,
    ) -> Tuple[str, str]:
        """
        Create training example in the specified supervision format.
        
        Returns:
            Tuple of (input_text, target_text)
        """
        observation = snapshot.get("observation", snapshot.get("state", ""))
        teacher_action = snapshot.get("teacher_hint", {}).get("suggested_action", "")
        rationale = snapshot.get("teacher_hint", {}).get("rationale", "")
        student_action = snapshot.get("action_taken", snapshot.get("policy_action", ""))
        
        if supervision_format == "demo":
            input_text = f"""Task: {snapshot.get('task_description', '')}

Current observation:
{observation}

What action should be taken?"""
            target_text = teacher_action
            
        elif supervision_format == "contrast":
            input_text = f"""Task: {snapshot.get('task_description', '')}

Current observation:
{observation}

The action "{student_action}" was suboptimal because it doesn't advance toward the goal effectively.
The better choice is "{teacher_action}" because: {rationale}

What is the correct action?"""
            target_text = teacher_action
            
        elif supervision_format == "hint":
            diagnosis = rationale.split("because")[-1].strip() if "because" in rationale else rationale
            input_text = f"""Task: {snapshot.get('task_description', '')}

Current observation:
{observation}

Hint: {diagnosis}

What action should be taken?"""
            target_text = teacher_action
            
        else:
            # Default to demo
            input_text = f"{observation}\n\nAction:"
            target_text = teacher_action
        
        return input_text, target_text
    
    def _micro_train(
        self,
        input_text: str,
        target_text: str,
    ) -> Tuple[object, float, float]:
        """
        Apply 1-2 LoRA gradient steps on a single example.
        
        Returns:
            Tuple of (micro_trained_model, loss_before, loss_after)
        """
        import torch
        from peft import get_peft_model, LoraConfig, TaskType
        
        self._load_base_model()
        
        # Tokenize
        full_text = f"{input_text}\n{target_text}"
        encoding = self._tokenizer(
            full_text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self._model.device)
        attention_mask = encoding["attention_mask"].to(self._model.device)
        labels = input_ids.clone()
        
        # Create LoRA model (fresh copy each time)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )
        
        # Get base model copy with LoRA
        model = get_peft_model(deepcopy(self._model), lora_config)
        
        # Compute loss before training
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_before = outputs.loss.item()
        
        # Training steps
        model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
        )
        
        for _ in range(self.config.n_gradient_steps):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Compute loss after training
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_after = outputs.loss.item()
        
        return model, loss_before, loss_after
    
    def _evaluate_policy(
        self,
        model,
        snapshot: dict,
        validation_task_ids: List[str],
    ) -> float:
        """
        Evaluate policy success rate via rollouts.
        
        Args:
            model: The policy model (base or micro-trained)
            snapshot: The original snapshot (for failure trajectory evaluation)
            validation_task_ids: Similar task IDs for validation
        
        Returns:
            Success rate across all evaluation episodes
        """
        import torch
        
        self._load_base_model()  # Ensure tokenizer is loaded
        
        successes = 0
        total = 0
        
        # Evaluate on failure trajectory (from snapshot state)
        if snapshot.get("env_state_bytes"):
            env = self.env_factory()
            try:
                # Restore state
                obs = env.set_state(snapshot["env_state_bytes"])
                
                for _ in range(self.config.n_rollouts_per_task):
                    success = self._run_single_rollout(
                        model, env, obs,
                        max_steps=self.config.max_steps_per_episode,
                    )
                    if success:
                        successes += 1
                    total += 1
            finally:
                env.close()
        
        # Evaluate on validation tasks
        for task_id in validation_task_ids[:self.config.n_validation_tasks]:
            env = self.env_factory()
            try:
                obs = env.reset(task_id)
                
                for _ in range(self.config.n_rollouts_per_task):
                    success = self._run_single_rollout(
                        model, env, obs,
                        max_steps=self.config.max_steps_per_episode,
                    )
                    if success:
                        successes += 1
                    total += 1
            finally:
                env.close()
        
        return successes / total if total > 0 else 0.0
    
    def _run_single_rollout(
        self,
        model,
        env,
        initial_obs: dict,
        max_steps: int,
    ) -> bool:
        """
        Run a single rollout and return success.
        """
        import torch
        
        observation = initial_obs.get("observation", "")
        valid_actions = initial_obs.get("valid_actions", [])
        
        done = False
        steps = 0
        final_reward = 0.0
        
        while not done and steps < max_steps and valid_actions:
            # Get action from model
            action = self._get_model_action(model, observation, valid_actions)
            
            # Take action
            obs, reward, done, info = env.step(action)
            observation = obs.get("observation", "")
            valid_actions = obs.get("valid_actions", [])
            final_reward = reward
            steps += 1
        
        return env.is_success(final_reward)
    
    def _get_model_action(
        self,
        model,
        observation: str,
        valid_actions: List[str],
    ) -> str:
        """
        Get action from model given observation and valid actions.
        Uses constrained decoding to ensure valid action selection.
        """
        import torch
        
        # Format prompt
        actions_str = ", ".join(valid_actions)
        prompt = f"""Observation: {observation[:1500]}

Valid actions: {actions_str}

Select the best action:"""
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        
        # Decode
        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action from generated text
        generated = generated[len(prompt):].strip()
        
        # Match to valid actions
        for action in valid_actions:
            if action.lower() in generated.lower():
                return action
        
        # Fallback: score each action and pick best
        action_scores = {}
        for action in valid_actions:
            action_inputs = self._tokenizer(
                f"{prompt}\n{action}",
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            action_inputs = {k: v.to(model.device) for k, v in action_inputs.items()}
            
            with torch.no_grad():
                outputs = model(**action_inputs, labels=action_inputs["input_ids"])
                action_scores[action] = -outputs.loss.item()  # Higher is better
        
        return max(action_scores, key=action_scores.get)
    
    def run_validation(
        self,
        snapshot: dict,
        validation_task_ids: List[str],
    ) -> MicroTrainingResult:
        """
        Run full micro-training validation on a single snapshot.
        """
        start_time = time.time()
        
        snapshot_id = snapshot.get("id", snapshot.get("snapshot_id", ""))
        quadrant = snapshot.get("quadrant", "")
        
        # Get CPT scores
        cpt = snapshot.get("cpt", {})
        elp_net = cpt.get("ELP_net", 0.0)
        elp_raw = cpt.get("ELP_raw", 0.0)
        route = cpt.get("route_net", "demo")
        
        # Create training example
        input_text, target_text = self._create_training_example(snapshot, route)
        
        # Evaluate base model BEFORE micro-training
        eval_start = time.time()
        self._load_base_model()
        success_before = self._evaluate_policy(
            self._model,
            snapshot,
            validation_task_ids,
        )
        eval_time_before = time.time() - eval_start
        
        # Micro-train
        train_start = time.time()
        micro_model, loss_before, loss_after = self._micro_train(input_text, target_text)
        train_time = time.time() - train_start
        
        # Evaluate AFTER micro-training
        eval_start = time.time()
        success_after = self._evaluate_policy(
            micro_model,
            snapshot,
            validation_task_ids,
        )
        eval_time_after = time.time() - eval_start
        
        # Compute improvement
        delta_micro = success_after - success_before
        
        # Clean up micro model
        del micro_model
        import torch
        torch.cuda.empty_cache()
        
        return MicroTrainingResult(
            snapshot_id=snapshot_id,
            quadrant=quadrant,
            elp_net=elp_net,
            elp_raw=elp_raw,
            route=route,
            loss_before=loss_before,
            loss_after=loss_after,
            loss_delta=loss_before - loss_after,
            success_before=success_before,
            success_after=success_after,
            delta_micro=delta_micro,
            n_evaluation_episodes=(self.config.n_validation_tasks + 1) * self.config.n_rollouts_per_task * 2,
            training_time_seconds=train_time,
            evaluation_time_seconds=eval_time_before + eval_time_after,
        )


def compute_cpt_correlations(results: List[MicroTrainingResult]) -> dict:
    """
    Compute correlations between CPT ELP and actual improvement.
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Overall correlation
    elp_values = [r.elp_net for r in results]
    delta_values = [r.delta_micro for r in results]
    
    correlations = {
        "overall": {},
        "by_quadrant": {},
        "validation_checks": {},
    }
    
    if len(results) > 2:
        r, p = pearsonr(elp_values, delta_values)
        rho, p_rho = spearmanr(elp_values, delta_values)
        
        correlations["overall"] = {
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": p_rho,
            "n": len(results),
        }
    
    # Per-quadrant correlations
    quadrants = set(r.quadrant for r in results if r.quadrant)
    for quadrant in quadrants:
        q_results = [r for r in results if r.quadrant == quadrant]
        if len(q_results) > 2:
            q_elp = [r.elp_net for r in q_results]
            q_delta = [r.delta_micro for r in q_results]
            r, p = pearsonr(q_elp, q_delta)
            correlations["by_quadrant"][quadrant] = {
                "pearson_r": r,
                "pearson_p": p,
                "n": len(q_results),
            }
    
    # Validation checks (from research plan)
    overall_r = correlations["overall"].get("pearson_r", 0)
    correlations["validation_checks"] = {
        "cpt_predicts_training": overall_r > 0.3,
        "median_elp_positive": np.median(elp_values) > 0,
        "median_improvement_positive": np.median(delta_values) > 0,
        "any_quadrant_above_0.2": any(
            q.get("pearson_r", 0) > 0.2 
            for q in correlations["by_quadrant"].values()
        ),
    }
    
    return correlations


def create_stratified_panel(
    all_snapshots: List[dict],
    n_per_quadrant: int = 50,
    seed: int = 42,
) -> List[dict]:
    """
    Create stratified validation panel with equal representation per quadrant.
    """
    import random
    random.seed(seed)
    
    # Group by quadrant
    by_quadrant = {}
    for snap in all_snapshots:
        q = snap.get("quadrant", "unknown")
        if q not in by_quadrant:
            by_quadrant[q] = []
        by_quadrant[q].append(snap)
    
    # Sample from each quadrant
    panel = []
    for quadrant, snapshots in by_quadrant.items():
        if quadrant == "unknown":
            continue
        n_sample = min(n_per_quadrant, len(snapshots))
        panel.extend(random.sample(snapshots, n_sample))
    
    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Run CPT validation via micro-training"
    )
    parser.add_argument(
        "--panel",
        type=Path,
        required=True,
        help="Path to validation panel JSON",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Base model path or name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for results",
    )
    parser.add_argument(
        "--n-gradient-steps",
        type=int,
        default=2,
        help="Number of gradient steps for micro-training",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=3,
        help="Rollouts per task for evaluation",
    )
    parser.add_argument(
        "--n-validation-tasks",
        type=int,
        default=10,
        help="Number of validation tasks per snapshot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    # Load panel
    logger.info(f"Loading validation panel from {args.panel}")
    with open(args.panel) as f:
        panel_data = json.load(f)
    
    snapshots = panel_data.get("snapshots", panel_data)
    logger.info(f"Loaded {len(snapshots)} snapshots")
    
    # Get validation task IDs (from panel metadata or generate)
    validation_task_ids = panel_data.get("validation_task_ids", [])
    if not validation_task_ids:
        # Use task IDs from snapshots as validation
        validation_task_ids = list(set(s.get("task_id", "") for s in snapshots))
    
    # Configure
    config = MicroTrainingConfig(
        n_gradient_steps=args.n_gradient_steps,
        n_rollouts_per_task=args.n_rollouts,
        n_validation_tasks=args.n_validation_tasks,
    )
    
    # Environment factory
    from src.data.webshop_env import create_env, WebShopConfig
    env_config = WebShopConfig(max_steps=15)
    
    def env_factory():
        return create_env(env_config, mock=False)
    
    # Create trainer
    trainer = MicroTrainer(
        base_model_path=args.base_model,
        config=config,
        env_factory=env_factory,
    )
    
    # Run validation
    results = []
    tracker = ProgressTracker(len(snapshots), "Micro-training validation")
    
    for snapshot in snapshots:
        try:
            result = trainer.run_validation(snapshot, validation_task_ids)
            results.append(result)
            
            logger.info(
                f"Snapshot {result.snapshot_id}: "
                f"ELP={result.elp_net:.3f}, "
                f"Δ_micro={result.delta_micro:.3f}"
            )
        except Exception as e:
            logger.error(f"Error on snapshot {snapshot.get('id', '?')}: {e}")
            continue
        
        tracker.update()
    
    tracker.finish()
    
    # Compute correlations
    correlations = compute_cpt_correlations(results)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "results": [asdict(r) for r in results],
        "correlations": correlations,
        "config": asdict(config),
        "summary": {
            "n_snapshots": len(results),
            "mean_elp_net": np.mean([r.elp_net for r in results]),
            "mean_delta_micro": np.mean([r.delta_micro for r in results]),
            "overall_correlation": correlations["overall"].get("pearson_r", 0),
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved results to {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CPT Validation Results")
    print("=" * 60)
    print(f"Snapshots evaluated: {len(results)}")
    print(f"Mean ELP_net: {output_data['summary']['mean_elp_net']:.3f}")
    print(f"Mean Δ_micro: {output_data['summary']['mean_delta_micro']:.3f}")
    print(f"Overall correlation (ρ): {correlations['overall'].get('pearson_r', 0):.3f}")
    print(f"P-value: {correlations['overall'].get('pearson_p', 1):.4f}")
    print("\nValidation checks:")
    for check, passed in correlations["validation_checks"].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
    print("\nPer-quadrant correlations:")
    for quadrant, stats in correlations["by_quadrant"].items():
        print(f"  {quadrant}: ρ={stats['pearson_r']:.3f} (n={stats['n']})")


if __name__ == "__main__":
    main()
