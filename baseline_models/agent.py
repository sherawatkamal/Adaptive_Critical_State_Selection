class Agent:
    """Agent wrapper - NO BART, with TRUE entropy computation"""
    
    def __init__(self, models_dict):
        self.model = models_dict['model']
        self.tokenizer = models_dict['tokenizer']
        self.data_collator = models_dict['data_collator']
        self.process = models_dict['process']
        self.process_goal = models_dict['process_goal']
        self.device = models_dict['device']
    


    def get_action_probs(self, obs: str, valid_acts: List[str]) -> Optional[torch.Tensor]:
        """
        Get action probability distribution from model.
        
        Returns:
            Tensor of probabilities over valid_acts, or None if can't compute
        """
        if not valid_acts:
            return None
        
        # Skip search states - can't compute entropy for generative actions
        if valid_acts[0].startswith('search['):
            return None
        
        # Encode state and actions
        state_encodings = self.tokenizer(self.process(obs), max_length=512, truncation=True, padding='max_length')
        action_encodings = self.tokenizer(list(map(self.process, valid_acts)), max_length=512, truncation=True, padding='max_length')
        
        batch = {
            'state_input_ids': state_encodings['input_ids'],
            'state_attention_mask': state_encodings['attention_mask'],
            'action_input_ids': action_encodings['input_ids'],
            'action_attention_mask': action_encodings['attention_mask'],
            'sizes': len(valid_acts),
            'images': [0.0] * 512,
            'labels': 0
        }
        batch = self.data_collator([batch])
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)
        
        return probs
    

    def compute_true_entropy(self, obs: str, valid_acts: List[str]) -> Tuple[float, float, float]:
        """
        Compute TRUE policy entropy: H(π|s) = -Σ π(a|s) log π(a|s)
        
        This is the CORRECT way to measure model uncertainty.
        NOT log(|valid_actions|) which just counts buttons!
        
        Args:
            obs: Observation string
            valid_acts: List of valid action strings
            
        Returns:
            (entropy, normalized_entropy, max_prob)
            - entropy: Raw entropy in nats
            - normalized_entropy: H / log(|A|), scaled to [0,1]
            - max_prob: Confidence in top action (1 - this = uncertainty)
        """
        probs = self.get_action_probs(obs, valid_acts)
        
        if probs is None:
            # Search state or error - return 0
            return 0.0, 0.0, 1.0
        
        # Compute entropy: H = -Σ p log p
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * torch.log(probs_clamped)).sum().item()
        
        # Normalized entropy: H / H_max where H_max = log(|A|)
        n_actions = len(valid_acts)
        max_entropy = np.log(n_actions) if n_actions > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Max probability (confidence)
        max_prob = probs.max().item()
        
        return entropy, normalized_entropy, max_prob
    
    def get_action(self, obs: str, info: dict, method='softmax') -> Tuple[str, dict]:
        """Get action from the model. Default is softmax for exploration."""
        valid_acts = info.get('valid', [])
        
        if not valid_acts:
            return 'click[back to search]', {'type': 'fallback'}
        
        # Handle search page - NO BART
        if valid_acts[0].startswith('search['):
            action = valid_acts[-1] if valid_acts else 'search[query]'
            return action, {
                'type': 'search', 
                'selected': 'valid_acts[-1]',
                'entropy': 0.0,
                'normalized_entropy': 0.0,
            }
        
        # Get probabilities
        probs = self.get_action_probs(obs, valid_acts)
        
        if probs is None:
            return valid_acts[0], {'type': 'error'}
        
        # Compute entropy
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * torch.log(probs_clamped)).sum().item()
        n_actions = len(valid_acts)
        max_entropy = np.log(n_actions) if n_actions > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Select action
        if method == 'greedy':
            idx = probs.argmax().item()
        else:  # softmax (default)
            idx = torch.multinomial(probs, 1)[0].item()
        
        action = valid_acts[idx] if idx < len(valid_acts) else valid_acts[0]
        return action, {
            'type': 'choice',
            'chosen_idx': idx,
            'num_valid': len(valid_acts),
            'confidence': probs[idx].item(),
            'action_probs': probs.cpu().tolist(),
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
        }