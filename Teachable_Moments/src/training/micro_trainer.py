"""
Micro-training for CPT validation.

Micro-training applies 1-2 gradient steps on a single example to test
whether CPT scores predict actual fine-tuning improvement.

Key fixes from initial prototype:
- No deepcopy of the full base model (infeasible for large models)
- Prompt tokens are masked in the loss
- Evaluation uses env state restoration
- LoRA parameters are reset between samples
"""

import base64
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Dict, List, Any
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MicroTrainingConfig:
    """Configuration for micro-training validation."""
    
    # Training
    n_steps: int = 2
    learning_rate: float = 1e-4
    max_seq_length: int = 1024
    
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Evaluation
    n_validation_rollouts: int = 3
    rollout_max_steps: int = 15
    n_validation_tasks: int = 10
    
    @classmethod
    def from_yaml(cls, path: str) -> "MicroTrainingConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("micro_training", {}))


def reset_lora_parameters(peft_model, adapter_name: Optional[str] = None) -> None:
    """Reinitialize LoRA weights in-place.
    
    Works for common PEFT LoRA modules where `lora_A` and `lora_B` are dicts.
    We set B to zeros so the initial LoRA update is zero (model == base).
    
    Args:
        peft_model: PEFT model with LoRA adapters
        adapter_name: Optional specific adapter name (uses default if None)
    """
    import torch
    import torch.nn as nn
    
    for name, module in peft_model.named_modules():
        # Check for PEFT LoRA module structure
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        
        # lora_A and lora_B are typically ModuleDicts with adapter name keys
        lora_A = module.lora_A
        lora_B = module.lora_B
        
        if isinstance(lora_A, nn.ModuleDict) and isinstance(lora_B, nn.ModuleDict):
            # Get the adapter key
            adapter_key = adapter_name or (list(lora_A.keys())[0] if lora_A else None)
            if adapter_key is None:
                continue
            
            if adapter_key in lora_A and adapter_key in lora_B:
                # Reset A with kaiming uniform (similar to PEFT default)
                nn.init.kaiming_uniform_(lora_A[adapter_key].weight, a=np.sqrt(5))
                # Reset B to zeros so initial LoRA contribution is zero
                nn.init.zeros_(lora_B[adapter_key].weight)
        elif hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
            # Direct Linear modules
            nn.init.kaiming_uniform_(lora_A.weight, a=np.sqrt(5))
            nn.init.zeros_(lora_B.weight)
    
    logger.debug(f"Reset LoRA parameters for adapter: {adapter_name or 'default'}")


def build_masked_batch(
    tokenizer,
    prompt: str,
    completion: str,
    max_seq_length: int,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Tokenize prompt+completion and mask prompt tokens in labels.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt: The prompt/input text
        completion: The completion/output text
        max_seq_length: Maximum sequence length
        device: Target device for tensors
        
    Returns:
        Dict with input_ids, attention_mask, labels (prompt tokens masked with -100)
    """
    import torch
    
    # Ensure stable boundary so tokenization doesn't merge across prompt/completion
    prompt_prefix = (prompt or "").rstrip() + "\n"
    full_text = prompt_prefix + (completion or "").rstrip()
    
    # Add EOS token if available
    if tokenizer.eos_token:
        full_text = full_text + tokenizer.eos_token
    
    # Tokenize full sequence
    enc_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    # Tokenize prompt only to get its length
    enc_prompt = tokenizer(
        prompt_prefix,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    prompt_len = int(enc_prompt["input_ids"].shape[1])
    
    # Create labels with prompt tokens masked
    labels = enc_full["input_ids"].clone()
    labels[:, :prompt_len] = -100  # Mask prompt tokens
    
    result = {
        "input_ids": enc_full["input_ids"],
        "attention_mask": enc_full["attention_mask"],
        "labels": labels,
    }
    
    if device is not None:
        result = {k: v.to(device) for k, v in result.items()}
    
    return result


def _decode_b64(env_state_b64: Optional[str]) -> Optional[bytes]:
    """Decode base64-encoded environment state."""
    if not env_state_b64:
        return None
    return base64.b64decode(env_state_b64.encode("ascii"))


def micro_train_single_example(
    peft_model,
    tokenizer,
    prompt: str,
    completion: str,
    config: Optional[MicroTrainingConfig] = None,
) -> float:
    """
    Fine-tune model on single example for 1-2 gradient steps.
    
    This version does NOT deepcopy the model (infeasible for large models).
    Instead, use reset_lora_parameters() before calling this to reset LoRA weights.
    
    Args:
        peft_model: PEFT model with LoRA adapters already attached
        tokenizer: Tokenizer for the model
        prompt: The prompt/input text
        completion: The completion/target text
        config: Micro-training configuration
        
    Returns:
        Final training loss
    """
    import torch
    
    if config is None:
        config = MicroTrainingConfig()
    
    # Build batch with masked labels
    batch = build_masked_batch(
        tokenizer=tokenizer,
        prompt=prompt,
        completion=completion,
        max_seq_length=config.max_seq_length,
        device=peft_model.device,
    )
    
    # Training steps
    peft_model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, peft_model.parameters()),
        lr=config.learning_rate,
    )
    
    final_loss = 0.0
    for step in range(config.n_steps):
        outputs = peft_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        final_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    peft_model.eval()
    return final_loss


def run_policy_rollout_from_state(
    env,
    model_factory,
    env_state_bytes: bytes,
    max_steps: int = 15,
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    """Run a policy rollout from a restored environment state.
    
    Args:
        env: Environment instance
        model_factory: ModelFactory for action selection
        env_state_bytes: Serialized environment state to restore
        max_steps: Maximum steps per rollout
        
    Returns:
        Dict with success, total_reward, n_steps
    """
    # Restore state
    obs_dict = env.set_state(env_state_bytes)
    observation = obs_dict.get("observation", "")
    valid_actions = obs_dict.get("valid_actions", [])
    instruction_text = obs_dict.get("instruction_text", "")
    
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done and steps < max_steps and valid_actions:
        # Get action from model
        action, _, _ = model_factory.decode_action(
            observation=observation,
            valid_actions=valid_actions,
            task_description=instruction_text,
            model=model,
            tokenizer=tokenizer,
        )
        
        # Take action
        obs_dict, reward, done, _info = env.step(action)
        
        observation = obs_dict.get("observation", "")
        valid_actions = obs_dict.get("valid_actions", [])
        total_reward += float(reward)
        steps += 1
    
    return {
        "success": env.is_success(total_reward),
        "total_reward": total_reward,
        "n_steps": steps,
    }


def evaluate_micro_training(
    env,
    peft_model,
    tokenizer,
    sample: Dict[str, Any],
    model_factory,
    cfg: MicroTrainingConfig,
    retention_task_ids: Optional[List[Any]] = None,
    env_factory: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Evaluate micro-training on a single sample.
    
    This is the main evaluation function for CPT validation:
    1. Reset LoRA so model == base
    2. Base eval (failure rollout from saved state)
    3. Base retention (optional)
    4. Micro-train on sample
    5. Post-train eval
    
    Args:
        env: Environment instance
        peft_model: PEFT model with LoRA adapters
        tokenizer: Model tokenizer
        sample: Sample dict with prompt, completion, env_state_b64
        model_factory: ModelFactory for action decoding
        cfg: Micro-training config
        retention_task_ids: Optional task IDs for retention evaluation
        env_factory: Factory to create new envs (required for retention)
        
    Returns:
        Dict with evaluation results
    """
    import torch
    
    results = {}
    
    # 1) Reset LoRA so model == base
    reset_lora_parameters(peft_model)
    
    # 2) Base eval (failure rollout from saved state)
    env_state_bytes = _decode_b64(sample.get("env_state_b64"))

    # NOTE: We evaluate multiple rollouts to reduce noise (cfg.n_validation_rollouts)
    base_successes = 0
    base_rewards: List[float] = []
    base_steps: List[int] = []
    if env_state_bytes:
        for _ in range(max(1, cfg.n_validation_rollouts)):
            try:
                r = run_policy_rollout_from_state(
                    env=env,
                    model_factory=model_factory,
                    env_state_bytes=env_state_bytes,
                    max_steps=cfg.rollout_max_steps,
                    model=peft_model,
                    tokenizer=tokenizer,
                )
                base_successes += int(bool(r.get('success', False)))
                base_rewards.append(float(r.get('total_reward', 0.0)))
                base_steps.append(int(r.get('n_steps', 0)))
            except Exception as e:
                logger.warning(f"Base eval rollout failed: {e}")

    base_success_rate = base_successes / max(1, cfg.n_validation_rollouts)
    base_reward_mean = float(np.mean(base_rewards)) if base_rewards else 0.0
    base_steps_mean = float(np.mean(base_steps)) if base_steps else 0.0

    results["base_success_rate"] = base_success_rate
    results["base_reward_mean"] = base_reward_mean
    results["base_steps_mean"] = base_steps_mean
    
    # 3) Base retention (optional)
    base_retention_rate = None
    if retention_task_ids and env_factory:
        retention_env = env_factory()
        try:
            successes = 0
            for task_id in retention_task_ids:
                obs_dict = retention_env.reset(task_id)
                obs = obs_dict.get("observation", "")
                valid_actions = obs_dict.get("valid_actions", [])
                instruction = obs_dict.get("instruction_text", "")
                
                done = False
                total_reward = 0.0
                for _ in range(cfg.rollout_max_steps):
                    action, _, _ = model_factory.decode_action(obs, valid_actions, instruction, model=peft_model, tokenizer=tokenizer)
                    obs_dict, reward, done, _ = retention_env.step(action)
                    obs = obs_dict.get("observation", "")
                    valid_actions = obs_dict.get("valid_actions", [])
                    total_reward += float(reward)
                    if done:
                        break
                
                if retention_env.is_success(total_reward):
                    successes += 1
            
            base_retention_rate = successes / len(retention_task_ids)
        finally:
            if hasattr(retention_env, "close"):
                retention_env.close()
    
    results["base_retention"] = base_retention_rate
    
    # 4) Micro-train on sample
    prompt = sample.get("prompt", sample.get("input", ""))
    completion = sample.get("completion", sample.get("output", ""))
    
    train_loss = micro_train_single_example(
        peft_model=peft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        completion=completion,
        config=cfg,
    )
    results["train_loss"] = train_loss
    
    # 5) Post-train eval (failure rollout)

    tuned_successes = 0
    tuned_rewards: List[float] = []
    tuned_steps: List[int] = []
    if env_state_bytes:
        for _ in range(max(1, cfg.n_validation_rollouts)):
            try:
                r = run_policy_rollout_from_state(
                    env=env,
                    model_factory=model_factory,
                    env_state_bytes=env_state_bytes,
                    max_steps=cfg.rollout_max_steps,
                    model=peft_model,
                    tokenizer=tokenizer,
                )
                tuned_successes += int(bool(r.get('success', False)))
                tuned_rewards.append(float(r.get('total_reward', 0.0)))
                tuned_steps.append(int(r.get('n_steps', 0)))
            except Exception as e:
                logger.warning(f"Tuned eval rollout failed: {e}")

    tuned_success_rate = tuned_successes / max(1, cfg.n_validation_rollouts)
    tuned_reward_mean = float(np.mean(tuned_rewards)) if tuned_rewards else 0.0
    tuned_steps_mean = float(np.mean(tuned_steps)) if tuned_steps else 0.0

    results["tuned_success_rate"] = tuned_success_rate
    results["tuned_reward_mean"] = tuned_reward_mean
    results["tuned_steps_mean"] = tuned_steps_mean
    
    # Compute deltas
    results["delta_success_rate"] = tuned_success_rate - base_success_rate
    results["delta_reward_mean"] = tuned_reward_mean - base_reward_mean
    results["delta_steps_mean"] = tuned_steps_mean - base_steps_mean
    
    # 6) Post-train retention (optional)
    tuned_retention_rate = None
    if retention_task_ids and env_factory:
        retention_env = env_factory()
        try:
            successes = 0
            for task_id in retention_task_ids:
                obs_dict = retention_env.reset(task_id)
                obs = obs_dict.get("observation", "")
                valid_actions = obs_dict.get("valid_actions", [])
                instruction = obs_dict.get("instruction_text", "")
                
                done = False
                total_reward = 0.0
                for _ in range(cfg.rollout_max_steps):
                    action, _, _ = model_factory.decode_action(obs, valid_actions, instruction, model=peft_model, tokenizer=tokenizer)
                    obs_dict, reward, done, _ = retention_env.step(action)
                    obs = obs_dict.get("observation", "")
                    valid_actions = obs_dict.get("valid_actions", [])
                    total_reward += float(reward)
                    if done:
                        break
                
                if retention_env.is_success(total_reward):
                    successes += 1
            
            tuned_retention_rate = successes / len(retention_task_ids)
        finally:
            if hasattr(retention_env, "close"):
                retention_env.close()
    
    results["tuned_retention"] = tuned_retention_rate
    if base_retention_rate is not None and tuned_retention_rate is not None:
        results["delta_retention"] = tuned_retention_rate - base_retention_rate
    
    return results


def _run_single_rollout(
    model,
    tokenizer,
    env,
    initial_obs: dict,
    max_steps: int,
) -> bool:
    """
    Run a single rollout and return whether it succeeded.
    
    Args:
        model: The policy model
        tokenizer: Model tokenizer
        env: Environment instance
        initial_obs: Initial observation dict with 'observation' and 'valid_actions'
        max_steps: Maximum steps
        
    Returns:
        True if episode succeeded
    """
    import torch
    
    observation = initial_obs.get("observation", "")
    valid_actions = initial_obs.get("valid_actions", [])
    
    done = False
    steps = 0
    final_reward = 0.0
    
    while not done and steps < max_steps and valid_actions:
        # Get action from model
        action = _get_model_action(model, tokenizer, observation, valid_actions)
        
        # Take action
        obs, reward, done, info = env.step(action)
        
        if isinstance(obs, dict):
            observation = obs.get("observation", "")
            valid_actions = obs.get("valid_actions", [])
        else:
            observation = obs
            valid_actions = env.get_available_actions().get("clickables", [])
        
        final_reward = reward
        steps += 1
    
    # Check success (WebShop uses reward threshold)
    return final_reward >= 0.1  # Partial credit threshold


def _get_model_action(
    model,
    tokenizer,
    observation: str,
    valid_actions: list[str],
) -> str:
    """
    Get action from model given observation and valid actions.
    """
    import torch
    
    # Format prompt
    actions_str = ", ".join(valid_actions[:15])  # Limit displayed actions
    prompt = f"""Observation: {observation[:1500]}

Valid actions: {actions_str}

Select the best action:"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = generated[len(prompt):].strip()
    
    # Match to valid actions
    generated_lower = generated.lower()
    
    # Exact match
    for action in valid_actions:
        if action.lower() == generated_lower:
            return action
    
    # Substring match
    for action in valid_actions:
        if action.lower() in generated_lower:
            return action
    
    # Reverse substring
    for action in valid_actions:
        if generated_lower in action.lower() and len(generated_lower) > 2:
            return action
    
    # Fallback: score each action
    action_scores = {}
    for action in valid_actions[:10]:  # Limit for speed
        action_prompt = f"{prompt}\n{action}"
        action_inputs = tokenizer(action_prompt, return_tensors="pt", truncation=True, max_length=2048)
        action_inputs = {k: v.to(model.device) for k, v in action_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**action_inputs, labels=action_inputs["input_ids"])
            action_scores[action] = -outputs.loss.item()
    
    if action_scores:
        return max(action_scores, key=action_scores.get)
    
    return valid_actions[0] if valid_actions else ""


def run_micro_training_validation(
    panel_path: str,
    base_model,
    tokenizer,
    config: Optional[MicroTrainingConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> "pd.DataFrame":
    """
    Full CPT validation via micro-training.
    
    For each snapshot in the panel:
    1. Get CPT labels (already computed)
    2. Micro-train on the example
    3. Evaluate improvement
    4. Record correlation between CPT and actual improvement
    
    Args:
        panel_path: Path to validation panel JSON
        base_model: Base model
        tokenizer: Tokenizer
        config: Micro-training configuration
        progress_callback: Called with (completed, total)
        
    Returns:
        DataFrame with validation results
    """
    import pandas as pd
    import json
    
    if config is None:
        config = MicroTrainingConfig()
    
    # Load panel
    # Load panel (IDs)
    with open(panel_path) as f:
        panel_data = json.load(f)
    
    # Load labeled snapshots (data)
    labeled_path = Path(panel_path).parent / "labeled_snapshots.json"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Could not find labeled_snapshots.json at {labeled_path}")
        
    with open(labeled_path) as f:
        all_snapshots = json.load(f)
        
    # Map ID -> Snapshot
    snap_map = {s["id"]: s for s in all_snapshots}
    
    # Filter panel
    panel_ids = panel_data.get("ids", []) if isinstance(panel_data, dict) else panel_data
    panel_snapshots = []
    
    for pid in panel_ids:
        if pid in snap_map:
            panel_snapshots.append(snap_map[pid])
        else:
            logger.warning(f"Panel ID {pid} not found in labeled snapshots")

    results = []
    total = len(panel_snapshots)
    
    for i, snap in enumerate(panel_snapshots):
        try:
            # Get CPT labels
            elp_net = snap.get("cpt", {}).get("ELP_net", 0)
            route = snap.get("cpt", {}).get("route_net", "demo")
            
            # Micro-train
            model_after, loss_before, loss_after = micro_train_single_example(
                base_model,
                tokenizer,
                snap,
                supervision_format=route,
                config=config,
            )
            
            # Record results
            results.append({
                "snapshot_id": snap.get("id", snap.get("snapshot", {}).get("id", "")),
                "quadrant": snap.get("quadrant", ""),
                "ELP_net": elp_net,
                "route": route,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "loss_delta": loss_before - loss_after,
            })
            
            if progress_callback:
                progress_callback(i + 1, total)
                
        except Exception as e:
            logger.error(f"Error processing snapshot {i}: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_cpt_correlations(df: "pd.DataFrame") -> dict:
    """
    Compute CPT validation correlations overall and per quadrant.
    
    Args:
        df: DataFrame with ELP_net and loss_delta columns
        
    Returns:
        Dict with correlation results and validation checks
    """
    from scipy.stats import pearsonr, spearmanr
    
    correlations = {}
    
    # Overall correlation
    if len(df) > 2:
        r, p = pearsonr(df["ELP_net"], df["loss_delta"])
        correlations["overall"] = {"pearson_r": r, "p_value": p}
        
        rho, p_rho = spearmanr(df["ELP_net"], df["loss_delta"])
        correlations["overall"]["spearman_rho"] = rho
    
    # Per-quadrant correlations
    for quadrant in df["quadrant"].unique():
        if not quadrant:
            continue
        
        q_df = df[df["quadrant"] == quadrant]
        if len(q_df) > 2:
            r, p = pearsonr(q_df["ELP_net"], q_df["loss_delta"])
            correlations[quadrant] = {"pearson_r": r, "p_value": p, "n": len(q_df)}
    
    # Validation checks
    overall_r = correlations.get("overall", {}).get("pearson_r", 0)
    
    validation = {
        "overall_pass": overall_r > 0.3,
        "median_elp_positive": df["ELP_net"].median() > 0,
        "median_improvement_positive": df["loss_delta"].median() > 0,
    }
    
    return {
        "correlations": correlations,
        "validation": validation,
        "summary": {
            "n_snapshots": len(df),
            "overall_pearson_r": overall_r,
            "mean_elp": df["ELP_net"].mean(),
            "mean_loss_delta": df["loss_delta"].mean(),
        },
    }
