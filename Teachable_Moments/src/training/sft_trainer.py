"""
Unified SFT training infrastructure.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    
    # Model configuration
    base_model: str = "meta-llama/Llama-3-8B-Instruct"
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    
    # Optimizer
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # Saving
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    
    # Other
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        # Support multiple config schemas:
        # - v8 engine schema: per_quadrant_training
        # - legacy/simple schema: sft
        # - repo pilot schema: training: {sft: ..., lora: ...}
        config_data = data.get("per_quadrant_training") or data.get("sft") or {}
        training_block = data.get("training") or {}
        sft_block = training_block.get("sft") or {}
        lora_block = training_block.get("lora") or {}

        merged = {}
        if isinstance(config_data, dict):
            merged.update(config_data)
        if isinstance(sft_block, dict):
            merged.update(sft_block)

        # Map common aliases into SFTConfig field names.
        if "lr" in merged and "learning_rate" not in merged:
            merged["learning_rate"] = merged["lr"]
        if "max_length" in merged and "max_seq_length" not in merged:
            merged["max_seq_length"] = merged["max_length"]

        # Merge LoRA block (pilot schema) into fields.
        if isinstance(lora_block, dict):
            if "r" in lora_block and "lora_rank" not in merged:
                merged["lora_rank"] = lora_block["r"]
            if "alpha" in lora_block and "lora_alpha" not in merged:
                merged["lora_alpha"] = lora_block["alpha"]
            if "dropout" in lora_block and "lora_dropout" not in merged:
                merged["lora_dropout"] = lora_block["dropout"]
            if "target_modules" in lora_block and "target_modules" not in merged:
                merged["target_modules"] = lora_block["target_modules"]

        # Allow base model to come from common config locations.
        model_block = data.get("model") or {}
        if "base_model" not in merged:
            merged["base_model"] = (
                merged.get("base_model")
                or model_block.get("model_path")
                or data.get("base_model")
                or data.get("model_path")
                or cls.base_model
            )

        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in allowed}

        def _to_int(x):
            if x is None:
                return None
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return x
            if isinstance(x, float):
                return int(x)
            if isinstance(x, str):
                return int(float(x))
            return int(x)

        def _to_float(x):
            if x is None:
                return None
            if isinstance(x, bool):
                return float(int(x))
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                return float(x)
            return float(x)

        int_fields = {
            "lora_rank",
            "lora_alpha",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
            "save_total_limit",
            "logging_steps",
            "eval_steps",
            "seed",
        }
        float_fields = {
            "lora_dropout",
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
        }

        for k in list(filtered.keys()):
            if k in int_fields:
                filtered[k] = _to_int(filtered[k])
            elif k in float_fields:
                filtered[k] = _to_float(filtered[k])
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
        }


class SFTTrainer:
    """
    Trainer for supervised fine-tuning with LoRA.
    
    This is a wrapper that can use either HuggingFace Trainer or custom training loops.
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup(self) -> None:
        """Initialize model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType
            
            logger.info(f"Loading base model: {self.config.base_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            
            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            
        except ImportError as e:
            logger.error(f"Missing dependencies for training: {e}")
            raise
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./output",
        callbacks: list = None,
    ) -> dict:
        """
        Run training.
        
        Args:
            train_dataset: Training dataset (HF Dataset or list of dicts)
            eval_dataset: Optional evaluation dataset
            output_dir: Directory for saving outputs
            callbacks: Optional training callbacks
            
        Returns:
            Training metrics dict
        """
        try:
            from transformers import TrainingArguments, Trainer
            from trl import SFTTrainer as TRLSFTTrainer
        except ImportError:
            logger.warning("TRL not available, using basic training")
            return self._train_basic(train_dataset, output_dir)
        
        if self.peft_model is None:
            self.setup()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
        )
        
        trainer = TRLSFTTrainer(
            model=self.peft_model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            callbacks=callbacks,
        )
        
        result = trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        
        return {
            "loss": result.training_loss,
            "epochs": self.config.epochs,
            "output_dir": output_dir,
        }
    
    def _train_basic(self, train_data: list, output_dir: str) -> dict:
        """Basic training loop without TRL."""
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        
        if self.peft_model is None:
            self.setup()
        
        # Simple dataset class with prompt masking
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                """Return masked LM labels (prompt tokens masked with -100)."""
                item = self.data[idx]
                prompt = str(item.get("input", item.get("prompt", "")))
                completion = str(item.get("output", item.get("completion", "")))
                
                # Ensure stable boundary so tokenization doesn't merge across it
                prompt_prefix = prompt.rstrip() + "\n"
                full_text = prompt_prefix + completion.rstrip()
                if self.tokenizer.eos_token:
                    full_text = full_text + self.tokenizer.eos_token
                
                enc_full = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                enc_prompt = self.tokenizer(
                    prompt_prefix,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_len = int(enc_prompt["input_ids"].shape[1])
                
                labels = enc_full["input_ids"].clone()
                labels[:, :prompt_len] = -100  # MASK PROMPT TOKENS
                
                return {
                    "input_ids": enc_full["input_ids"].squeeze(),
                    "attention_mask": enc_full["attention_mask"].squeeze(),
                    "labels": labels.squeeze(),
                }
        
        
        dataset = SimpleDataset(train_data, self.tokenizer, self.config.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        optimizer = AdamW(self.peft_model.parameters(), lr=self.config.learning_rate)
        
        self.peft_model.train()
        total_loss = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self.peft_model.device) for k, v in batch.items()}
                
                outputs = self.peft_model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
            total_loss += avg_loss
        
        # Save model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(output_dir)
        
        return {
            "loss": total_loss / self.config.epochs,
            "epochs": self.config.epochs,
            "output_dir": output_dir,
        }
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.peft_model is not None:
            self.peft_model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)


def train_sft(
    base_model: str,
    training_data: list[dict],
    output_dir: str,
    config: Optional[SFTConfig] = None,
    **kwargs,
) -> str:
    """
    High-level function to train an SFT model.
    
    Args:
        base_model: Path or name of base model
        training_data: List of training examples (input/output dicts)
        output_dir: Output directory for saving model
        config: Optional SFTConfig (created from defaults + kwargs if None)
        **kwargs: Override config values
        
    Returns:
        Path to saved model
    """
    if config is None:
        config = SFTConfig(base_model=base_model, **kwargs)
    else:
        config.base_model = base_model
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    logger.info(f"Training SFT model with {len(training_data)} examples")
    logger.info(f"Config: {config.to_dict()}")
    
    trainer = SFTTrainer(config)
    
    # Convert to HF format if needed
    if training_data and isinstance(training_data[0], dict):
        hf_data = [{"text": f"{d['input']}\n{d['output']}"} for d in training_data]
    else:
        hf_data = training_data
    
    try:
        from datasets import Dataset
        train_dataset = Dataset.from_list(hf_data)
    except ImportError:
        train_dataset = hf_data
    
    result = trainer.train(train_dataset, output_dir=output_dir)
    
    # Save config
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    return output_dir
