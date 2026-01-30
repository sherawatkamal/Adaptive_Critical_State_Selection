#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_data(data, val_ratio: float):
    if val_ratio <= 0:
        return data, []
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def format_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        parts.append(f"{role}:\n{msg.get('content','')}")
    if add_generation_prompt:
        parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


class FailureDiagnosisDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        messages = record.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Each record must end with an assistant message.")

        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_text = format_chat(self.tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = format_chat(self.tokenizer, full_messages, add_generation_prompt=False)

        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]

        # Labels only for assistant portion; mask prompt tokens.
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        labels = labels[: len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
        }


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def maybe_enable_lora(model, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    if not use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required for --use_lora. Install it first.") from exc

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-3B on failure diagnosis SFT data.")
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-3B",
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--train_file",
        required=True,
        help="Path to SFT JSONL (messages format).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save the trained model.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    data = load_jsonl(args.train_file)
    train_data, val_data = split_data(data, args.val_ratio)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = maybe_enable_lora(model, args.use_lora, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_dataset = FailureDiagnosisDataset(train_data, tokenizer, args.max_length)
    eval_dataset = FailureDiagnosisDataset(val_data, tokenizer, args.max_length) if val_data else None
    collator = DataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()