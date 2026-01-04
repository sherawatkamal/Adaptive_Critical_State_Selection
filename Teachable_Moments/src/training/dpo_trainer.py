"""
Minimal DPO Trainer implementation (No TRL dependency).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Optional, List, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_steps: int = 1000
    max_length: int = 1024
    log_every: int = 10

class DPODataset(Dataset):
    def __init__(self, examples: List['DPOExample'], tokenizer, max_length: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "prompt": ex.prompt,
            "chosen": ex.chosen,
            "rejected": ex.rejected
        }

    def collate_fn(self, batch):
        prompts = [b["prompt"] for b in batch]
        chosens = [b["chosen"] for b in batch]
        rejecteds = [b["rejected"] for b in batch]
        
        # Tokenize chosen: prompt + chosen
        chosen_input = [p + c for p, c in zip(prompts, chosens)]
        rejected_input = [p + r for p, r in zip(prompts, rejecteds)]
        
        # We need masks to ignore prompt loss, but DPO usually computes logprobs on (prompt+completion) and subtracts prompt logprobs?
        # Actually simpler approach: Just run forward on full sequence, but mask tokens in the prompt for loss calculation? 
        # Standard DPO formulation uses log(pi(y|x)) which implies conditional probability of completion given prompt.
        
        # Simple implementation:
        # Tokenize full sequences
        enc_chosen = self.tokenizer(chosen_input, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc_rejected = self.tokenizer(rejected_input, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Ideally we mask the prompt part, but for a minimal implementation we can just compute full sequence log probability difference?
        # Standard DPO requires strictly maximizing likelihood of chosen (conditioned on prompt) vs rejected.
        # If we include prompt in logprobs, they cancel out in (log_pi_chosen - log_pi_rejected) anyway 
        # IF the prompt is identical.
        # So evaluating logp(prompt+response) is sufficient because logp(prompt+response) = logp(prompt) + logp(response|prompt).
        # And logp(prompt) is same for both terms (since prompt is same).
        # So (logp(p+c) - logp(p+r)) == (logp(c|p) - logp(r|p)).
        # So we don't strictly need to mask prompt for the DPO *difference* term.
        # HOWEVER, the reference model term also cancels.
        
        return {
            "chosen_input_ids": enc_chosen["input_ids"],
            "chosen_attention_mask": enc_chosen["attention_mask"],
            "rejected_input_ids": enc_rejected["input_ids"],
            "rejected_attention_mask": enc_rejected["attention_mask"],
        }

class DPOTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        config: DPOConfig,
        train_dataset: Dataset,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
            
        self.tokenizer = tokenizer
        self.config = config
        self.loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=train_dataset.collate_fn
        )
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=config.learning_rate)

    def _get_batch_logprobs(self, model, input_ids, attention_mask):
        """Compute log probabilities of the *full* sequence."""
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits # [B, T, V]
        
        # Shift logits and labels for next-token prediction
        # standard LM loss: log P(x_t | x_<t)
        # logits[:, :-1] predicts input_ids[:, 1:]
        
        labels = input_ids[:, 1:].clone()
        shifted_logits = logits[:, :-1, :]
        
        # Compute logprobs
        # Gather log prob of the actual token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # CrossEntropyLoss returns -log(p). We want log(p).
        # shape: [B, T-1]
        nll = loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), labels.reshape(-1))
        all_feature_logprobs = -nll.view(labels.size(0), labels.size(1))
        
        # Apply mask (mask is for input_ids, so shift it too)
        mask = attention_mask[:, 1:]
        
        # Sum over sequence
        seq_logprobs = (all_feature_logprobs * mask).sum(dim=1)
        return seq_logprobs

    def train_step(self, batch):
        self.policy.train()
        
        # Move to device
        device = self.policy.device
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        # 1. Policy Logprobs
        policy_chosen_logps = self._get_batch_logprobs(self.policy, chosen_ids, chosen_mask)
        policy_rejected_logps = self._get_batch_logprobs(self.policy, rejected_ids, rejected_mask)

        # 2. Reference Logprobs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logprobs(self.ref_model, chosen_ids, chosen_mask)
            ref_rejected_logps = self._get_batch_logprobs(self.ref_model, rejected_ids, rejected_mask)

        # 3. DPO Loss
        # log(pi/ref)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        # -log sigmoid(beta * logits)
        losses = -F.logsigmoid(self.config.beta * logits)
        loss = losses.mean()
        
        # 4. Step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item(), (policy_chosen_logps > policy_rejected_logps).float().mean().item()

    def train(self):
        step = 0
        pbar = tqdm(total=self.config.max_steps)
        
        while step < self.config.max_steps:
            for batch in self.loader:
                if step >= self.config.max_steps:
                    break
                    
                loss, acc = self.train_step(batch)
                
                step += 1
                pbar.update(1)
                
                if step % self.config.log_every == 0:
                    pbar.set_description(f"Loss: {loss:.4f} Acc: {acc:.2f}")

        pbar.close()
