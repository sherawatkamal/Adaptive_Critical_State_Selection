"""
Unified Model Factory for Teachable Moments.

This module provides consistent model loading across all components:
- Base model loading (HuggingFace transformers)
- LoRA adapter loading and merging (PEFT)
- Quantization support (bitsandbytes)
- Action decoding with WebShop constraints

Usage:
    from src.utils.model_factory import ModelFactory, ModelConfig
    
    config = ModelConfig(
        model_path="meta-llama/Llama-3-8B-Instruct",
        lora_path="results/phase2/Q1_demo/checkpoint-final",
        load_in_8bit=True,
    )
    
    factory = ModelFactory(config)
    model, tokenizer = factory.load()
    action, probs, raw_output = factory.decode_action(observation, valid_actions)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    
    # Model identification
    model_path: str = "meta-llama/Llama-3-8B-Instruct"
    model_type: str = "causal_lm"  # causal_lm, seq2seq
    
    # LoRA configuration
    lora_path: Optional[str] = None
    merge_lora: bool = True  # If True, merge LoRA weights into base model
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Device configuration
    device_map: str = "auto"
    torch_dtype: str = "float16"  # float16, bfloat16, float32
    
    # Generation defaults
    max_new_tokens: int = 64
    temperature: float = 0.1
    do_sample: bool = False
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("model", data))
    
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "ModelConfig":
        """Infer config from checkpoint directory."""
        checkpoint_path = Path(checkpoint_dir)

        # If this is already a HuggingFace model directory (base model), treat it as
        # the model_path directly.
        # Common markers:
        # - config.json
        # - model.safetensors.index.json (sharded safetensors)
        # - tokenizer_config.json
        hf_markers = [
            checkpoint_path / "config.json",
            checkpoint_path / "model.safetensors.index.json",
            checkpoint_path / "tokenizer_config.json",
        ]
        is_hf_model_dir = checkpoint_path.is_dir() and any(p.exists() for p in hf_markers)
        
        # Check for LoRA adapter
        lora_path = None
        if (checkpoint_path / "adapter_config.json").exists():
            lora_path = str(checkpoint_path)
        
        # Try to find base model from adapter config
        model_path = str(checkpoint_path) if (is_hf_model_dir and lora_path is None) else "meta-llama/Llama-3-8B-Instruct"  # Default
        adapter_config = checkpoint_path / "adapter_config.json"
        if adapter_config.exists():
            import json
            with open(adapter_config) as f:
                config = json.load(f)
                model_path = config.get("base_model_name_or_path", model_path)
        
        return cls(model_path=model_path, lora_path=lora_path)


class ModelFactory:
    """
    Factory for loading models with consistent configuration.
    
    Handles:
    - HuggingFace transformers models
    - PEFT/LoRA adapters
    - Quantization
    - Device placement
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def load(self) -> Tuple[Any, Any]:
        """
        Load model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self._loaded:
            return self._model, self._tokenizer
        
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required to load HuggingFace models. "
                "Install torch or run with --mock-policy for smoke tests."
            ) from e

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info(f"Loading model from {self.config.model_path}")
        
        # Configure quantization
        quantization_config = None
        if self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Get torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            cache_dir=self.config.cache_dir,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device_map,
            "cache_dir": self.config.cache_dir,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs,
        )
        
        # Load LoRA adapter if specified
        if self.config.lora_path:
            self._load_lora_adapter()
        
        self._loaded = True
        logger.info(f"Model loaded successfully on {self._model.device}")
        
        return self._model, self._tokenizer
    
    def _load_lora_adapter(self):
        """Load and optionally merge LoRA adapter."""
        from peft import PeftModel
        
        logger.info(f"Loading LoRA adapter from {self.config.lora_path}")
        
        self._model = PeftModel.from_pretrained(
            self._model,
            self.config.lora_path,
        )
        
        if self.config.merge_lora:
            logger.info("Merging LoRA weights into base model")
            self._model = self._model.merge_and_unload()
    
    def decode_action(
        self,
        observation: str,
        valid_actions: List[str],
        task_description: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, float], str]:
        """
        Decode action from model output with WebShop constraints.
        
        Args:
            observation: Current observation text
            valid_actions: List of valid action strings
            task_description: Optional task description for context
            model: Model (uses cached if not provided)
            tokenizer: Tokenizer (uses cached if not provided)
        
        Returns:
            Tuple of (selected_action, action_probabilities, raw_output)
            - selected_action: The parsed action string
            - action_probabilities: Dict mapping actions to probabilities
            - raw_output: The raw generated text from the model
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for ModelFactory.decode_action(). "
                "Install torch or avoid model-backed decoding in smoke tests."
            ) from e

        if model is None or tokenizer is None:
            model, tokenizer = self.load()
        
        # Format prompt
        prompt = self._format_action_prompt(observation, valid_actions, task_description)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with constrained decoding
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode generated text
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Extract action from generated text
        action = self._extract_action(generated_text, valid_actions)
        
        # Compute action probabilities via scoring
        action_probs = self._compute_action_probabilities(
            model, tokenizer, prompt, valid_actions
        )
        
        return action, action_probs, generated_text
    
    def _format_action_prompt(
        self,
        observation: str,
        valid_actions: List[str],
        task_description: Optional[str] = None,
    ) -> str:
        """Format prompt for action selection."""
        parts = []
        
        if task_description:
            parts.append(f"Task: {task_description}")
        
        parts.append(f"Observation:\n{observation[:1500]}")  # Truncate long observations
        
        # Format valid actions
        actions_str = ", ".join(valid_actions[:20])  # Limit to 20 actions
        parts.append(f"Valid actions: {actions_str}")
        
        parts.append("Select the best action:")
        
        return "\n\n".join(parts)
    
    def _extract_action(self, generated_text: str, valid_actions: List[str]) -> str:
        """
        Extract action from generated text with multiple fallback strategies.
        """
        generated_lower = generated_text.lower().strip()
        
        # Strategy 1: Exact match
        for action in valid_actions:
            if action.lower() == generated_lower:
                return action
        
        # Strategy 2: Generated text contains action
        for action in valid_actions:
            if action.lower() in generated_lower:
                return action
        
        # Strategy 3: Action contains generated text (for partial generation)
        for action in valid_actions:
            if generated_lower in action.lower() and len(generated_lower) > 2:
                return action
        
        # Strategy 4: Parse WebShop action format
        # Format: search[query] or click[element]
        search_match = re.search(r'search\s*\[\s*([^\]]+)\s*\]', generated_text, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            for action in valid_actions:
                if action.lower().startswith("search") and query.lower() in action.lower():
                    return action
        
        click_match = re.search(r'click\s*\[\s*([^\]]+)\s*\]', generated_text, re.IGNORECASE)
        if click_match:
            element = click_match.group(1).strip().lower()
            for action in valid_actions:
                if element in action.lower():
                    return action
        
        # Strategy 5: Return first action as fallback
        logger.warning(f"Could not extract action from: {generated_text[:100]}. Using fallback.")
        return valid_actions[0] if valid_actions else ""
    
    def _compute_action_probabilities(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        valid_actions: List[str],
    ) -> Dict[str, float]:
        """
        Compute probability for each valid action via scoring.
        
        CRITICAL FIX: Masks prompt tokens so score is the conditional
        likelihood P(action|prompt) instead of P(prompt+action).
        Without masking, prompt tokens dominate the loss and make
        all actions look nearly identical.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for ModelFactory action-probability scoring."
            ) from e

        action_probs = {}
        
        # Tokenize prompt to get its length
        prompt_encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        prompt_len = int(prompt_encoding["input_ids"].shape[1])
        
        for action in valid_actions[:40]:  # Limit for speed
            # Tokenize prompt + action
            full_text = f"{prompt} {action}"
            full_encoding = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = full_encoding["input_ids"].to(model.device)
            attention_mask = full_encoding["attention_mask"].to(model.device)
            
            # CRITICAL: Mask prompt tokens in labels so we only score the completion
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            
            # Count action tokens (unmasked tokens)
            n_action_tokens = int((labels != -100).sum().item())
            
            # Compute loss (negative log likelihood over action tokens only)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            if n_action_tokens > 0:
                # loss is average NLL over unmasked tokens
                # log P(action|prompt) = -loss * n_tokens
                avg_nll = outputs.loss.item()
                log_prob = -avg_nll * n_action_tokens
                action_probs[action] = torch.exp(torch.tensor(log_prob / n_action_tokens)).item()
            else:
                action_probs[action] = 0.0
        
        # Normalize probabilities
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v / total for k, v in action_probs.items()}
        
        return action_probs
    
    def get_action_distribution(
        self,
        observation: str,
        valid_actions: List[str],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Get full action probability distribution.
        
        This is used for uncertainty estimation (entropy, margin, etc.)
        """
        if model is None or tokenizer is None:
            model, tokenizer = self.load()
        
        prompt = self._format_action_prompt(observation, valid_actions)
        return self._compute_action_probabilities(model, tokenizer, prompt, valid_actions)


# Convenience functions for common use cases

def load_student_model(checkpoint_path: str, **kwargs) -> Tuple[Any, Any, ModelFactory]:
    """
    Load a student model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        **kwargs: Additional ModelConfig parameters
    
    Returns:
        Tuple of (model, tokenizer, factory)
    """
    config = ModelConfig.from_checkpoint(checkpoint_path)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    factory = ModelFactory(config)
    model, tokenizer = factory.load()
    return model, tokenizer, factory


def load_base_model(model_name: str = "meta-llama/Llama-3-8B-Instruct", **kwargs) -> Tuple[Any, Any, ModelFactory]:
    """
    Load a base model without LoRA.
    
    Args:
        model_name: HuggingFace model name or path
        **kwargs: Additional ModelConfig parameters
    
    Returns:
        Tuple of (model, tokenizer, factory)
    """
    config = ModelConfig(model_path=model_name, **kwargs)
    factory = ModelFactory(config)
    model, tokenizer = factory.load()
    return model, tokenizer, factory


def create_env_factory(
    mock: bool = False,
    max_steps: int = 15,
    observation_mode: str = "text",
):
    """
    Create an environment factory function.
    
    This provides a consistent way to create WebShop environments
    across all evaluation and simulation scripts.
    
    Args:
        mock: If True, use mock environment for testing
        max_steps: Maximum steps per episode
        observation_mode: Observation mode ('text', 'html', 'text_rich')
    
    Returns:
        Factory function that creates environments
    """
    from src.data.webshop_env import create_env, WebShopConfig
    
    config = WebShopConfig(
        max_steps=max_steps,
        observation_mode=observation_mode,
    )
    
    def factory():
        return create_env(config, mock=mock)
    
    return factory
