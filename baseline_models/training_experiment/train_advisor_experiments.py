#!/usr/bin/env python3
"""
SIMPLIFIED Train 4 Models for Advisor's Data Selection Experiment

This version uses simple supervised learning instead of the complex data_collator.

Usage:
    python train_advisor_experiments_simple.py \
        --base_model ../ckpts/web_click/epoch_9/model.pth \
        --training_dir ./training_splits_advisor \
        --output_dir ./trained_models_advisor \
        --epochs 3 \
        --batch_size 16 \
        --learning_rate 2e-5
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, '..')
sys.path.insert(0, '.')

from train_choice_il import tokenizer, process
from models.bert import BertModelForWebshop, BertConfigForWebshop
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW


def train_simple(model, train_data, device, epochs=3, batch_size=16, lr=2e-5):
    """
    Simple training loop - process one example at a time.
    More robust than trying to batch with the complex data_collator.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    total_steps = len(train_data) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    total_loss = 0
    step = 0
    
    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        progress_bar = tqdm(train_data, desc=f"  Training")
        
        for ex in progress_bar:
            # Process state
            state_text = process(ex['state'])
            state_enc = tokenizer(
                state_text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'  # Return PyTorch tensors
            )
            
            # Process action
            action_text = process(ex['action'])
            action_enc = tokenizer(
                action_text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'  # Return PyTorch tensors
            )
            
            # Move to device and ensure correct shapes
            # tokenizer returns shape (1, seq_len), we need (seq_len,) for some models
            batch = {
                'state_input_ids': state_enc['input_ids'].squeeze(0).unsqueeze(0).to(device),
                'state_attention_mask': state_enc['attention_mask'].squeeze(0).unsqueeze(0).to(device),
                'action_input_ids': action_enc['input_ids'].squeeze(0).unsqueeze(0).to(device),
                'action_attention_mask': action_enc['attention_mask'].squeeze(0).unsqueeze(0).to(device),
                'sizes': torch.tensor([1], dtype=torch.long).to(device),  # Single action, as tensor
                'images': torch.zeros(1, 512, dtype=torch.float32).to(device),
                'labels': torch.tensor([0], dtype=torch.long).to(device),  # Correct action is at index 0, as tensor
            }
            
            # Forward pass
            outputs = model(**batch)
            
            # Check if model returned valid loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Model didn't return loss, compute it manually
                # This happens when the model expects different input format
                print("\n⚠️  WARNING: Model didn't return loss, computing manually...")
                if isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                else:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                labels = batch['labels']
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Sanity check loss value (only check occasionally to avoid spam)
            if step % 1000 == 0 and loss.item() == 0:
                print(f"\n⚠️  WARNING: Loss is 0 at step {step}. Model may not be learning.")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            step += 1
            
            if step % 100 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_data)
        print(f"  Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / total_steps
    return avg_total_loss


def main():
    parser = argparse.ArgumentParser(description="Train 4 models (simplified)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--training_dir", required=True)
    parser.add_argument("--output_dir", default="./trained_models_advisor")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--single_split", type=str, default=None,
                       help="Train only a single split (e.g., train1_all_recoverable)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("TRAINING 4 MODELS (SIMPLIFIED VERSION)")
    print("="*70)
    print(f"Base model: {args.base_model}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    splits = [
        ("train1_all_recoverable", "All Recoverable"),
        ("train2_middle_u_recoverable", "Middle U Recoverable"),
        ("train3_all_states", "All States"),
        ("train4_high_low_u_recoverable", "High/Low U Recoverable"),
    ]
    
    # If single_split specified, only train that one
    if args.single_split:
        splits = [(args.single_split, args.single_split.replace('_', ' ').title())]
        print(f"Training single split: {args.single_split}")
        print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    
    for split_name, description in splits:
        print("="*70)
        print(f"TRAINING: {description}")
        print("="*70)
        
        train_file = os.path.join(args.training_dir, f"{split_name}.json")
        if not os.path.exists(train_file):
            print(f"⚠️  WARNING: {train_file} not found, skipping...")
            continue
        
        print(f"Loading {train_file}...")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        print(f"  Loaded {len(train_data)} examples")
        
        if len(train_data) == 0:
            print(f"⚠️  WARNING: No examples, skipping...")
            continue
        
        # Load fresh model
        print("Loading model...")
        config = BertConfigForWebshop(image=False)
        model = BertModelForWebshop(config)
        model.load_state_dict(
            torch.load(args.base_model, map_location=device), 
            strict=False
        )
        model.to(device)
        
        # Train
        print("Training...")
        start_time = datetime.now()
        avg_loss = train_simple(
            model, train_data, device, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            lr=args.learning_rate
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Training complete!")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        # Save
        output_path = os.path.join(args.output_dir, f"{split_name}_model.pth")
        torch.save(model.state_dict(), output_path)
        print(f"  Saved: {output_path}")
        
        config_dict = {
            'split_name': split_name,
            'description': description,
            'train_examples': len(train_data),
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'avg_loss': avg_loss,
            'training_time_seconds': elapsed,
        }
        results.append(config_dict)
        
        config_path = os.path.join(args.output_dir, f"{split_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print()
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*70)
    print("✓ ALL TRAINING COMPLETE")
    print("="*70)
    print(f"\nTrained {len(results)} models")
    for r in results:
        print(f"  - {r['split_name']}: {r['train_examples']} examples, "
              f"loss={r['avg_loss']:.4f}, time={r['training_time_seconds']/60:.1f}min")


if __name__ == "__main__":
    main()