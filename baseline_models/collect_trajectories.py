"""
Trajectory Collection Script with Multi-Model Support

This script collects trajectories from WebShop tasks using various model backends:
- BERT+BART (original WebShop model)
- LLaMA/Qwen/other HuggingFace LLMs
- OpenAI API models (GPT-4, etc.)

Usage:
    # Using BERT+BART (original)
    python collect_trajectories_multimodel.py --model_type bert_bart --num_failures 500
    
    # Using LLaMA
    python collect_trajectories_multimodel.py --model_type llama --llm_path meta-llama/Llama-3.1-8B-Instruct --num_failures 500
    
    # Using Qwen
    python collect_trajectories_multimodel.py --model_type llm --llm_path Qwen/Qwen3-8B --num_failures 500
    
    # Using OpenAI
    python collect_trajectories_multimodel.py --model_type openai --openai_model gpt-4o-mini --num_failures 500
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F


# ============================================================================
# Abstract Model Interface
# ============================================================================

class WebShopAgent(ABC):
    """Abstract base class for WebShop agents."""
    
    @abstractmethod
    def predict(self, obs: str, info: Dict, verbose: bool = False) -> str:
        """
        Predict the next action given observation and info.
        
        Args:
            obs: Current observation string
            info: Info dict containing 'valid' actions, 'goal', etc.
            verbose: Whether to print debug info
            
        Returns:
            Selected action string
        """
        pass
    
    @abstractmethod
    def get_action_distribution(self, obs: str, info: Dict) -> Dict[str, float]:
        """
        Get probability distribution over valid actions.
        
        Args:
            obs: Current observation string
            info: Info dict containing 'valid' actions
            
        Returns:
            Dict mapping action -> probability
        """
        pass


# ============================================================================
# BERT+BART Agent (Original WebShop Model)
# ============================================================================

class BertBartAgent(WebShopAgent):
    """Original BERT+BART WebShop agent."""
    
    def __init__(
        self,
        model_path: str = "./ckpts/web_click/epoch_9/model.pth",
        bart_path: str = "./ckpts/web_search/checkpoint-800",
        use_bart: bool = True,
        use_image: bool = True,
        use_softmax: bool = True,
    ):
        # Lazy imports to avoid errors if not using this agent
        from transformers import BartForConditionalGeneration, BartTokenizer
        from train_choice_il import (
            BertConfigForWebshop, BertModelForWebshop, 
            tokenizer, data_collator, process, process_goal
        )
        
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.process = process
        self.process_goal = process_goal
        self.use_softmax = use_softmax
        
        # Load BART for search queries
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        if use_bart:
            self.bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
            print(f'BART model loaded from {bart_path}')
        else:
            self.bart_model = None
        
        # Load BERT model for action selection
        config = BertConfigForWebshop(image=use_image)
        self.model = BertModelForWebshop(config)
        self.model.cuda()
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print(f'BERT IL model loaded from {model_path}')
    
    def _bart_predict(self, input_text: str, **kwargs) -> List[str]:
        input_ids = self.bart_tokenizer(input_text)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        output = self.bart_model.generate(input_ids, max_length=512, **kwargs)
        return self.bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=True)
    
    def predict(self, obs: str, info: Dict, verbose: bool = False) -> str:
        valid_acts = info['valid']
        
        # Handle search action
        if valid_acts[0].startswith('search['):
            if self.bart_model is None:
                return valid_acts[-1]
            else:
                goal = self.process_goal(obs)
                query = self._bart_predict(goal, num_return_sequences=5, num_beams=5)
                query = query[0]
                return f'search[{query}]'
        
        # Score actions with BERT
        state_encodings = self.tokenizer(
            self.process(obs), max_length=512, truncation=True, padding='max_length'
        )
        action_encodings = self.tokenizer(
            list(map(self.process, valid_acts)), 
            max_length=512, truncation=True, padding='max_length'
        )
        
        batch = {
            'state_input_ids': state_encodings['input_ids'],
            'state_attention_mask': state_encodings['attention_mask'],
            'action_input_ids': action_encodings['input_ids'],
            'action_attention_mask': action_encodings['attention_mask'],
            'sizes': len(valid_acts),
            'images': info['image_feat'].tolist(),
            'labels': 0
        }
        batch = self.data_collator([batch])
        batch = {k: v.cuda() for k, v in batch.items()}
        
        outputs = self.model(**batch)
        
        if self.use_softmax:
            idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[0].item()
        else:
            idx = outputs.logits[0].argmax(0).item()
        
        return valid_acts[idx]
    
    def get_action_distribution(self, obs: str, info: Dict) -> Dict[str, float]:
        valid_acts = info['valid']
        
        if valid_acts[0].startswith('search['):
            # For search, return uniform over valid actions
            return {a: 1.0 / len(valid_acts) for a in valid_acts}
        
        state_encodings = self.tokenizer(
            self.process(obs), max_length=512, truncation=True, padding='max_length'
        )
        action_encodings = self.tokenizer(
            list(map(self.process, valid_acts)), 
            max_length=512, truncation=True, padding='max_length'
        )
        
        batch = {
            'state_input_ids': state_encodings['input_ids'],
            'state_attention_mask': state_encodings['attention_mask'],
            'action_input_ids': action_encodings['input_ids'],
            'action_attention_mask': action_encodings['attention_mask'],
            'sizes': len(valid_acts),
            'images': info['image_feat'].tolist(),
            'labels': 0
        }
        batch = self.data_collator([batch])
        batch = {k: v.cuda() for k, v in batch.items()}
        
        outputs = self.model(**batch)
        probs = F.softmax(outputs.logits[0], dim=0).cpu().tolist()
        
        return {a: p for a, p in zip(valid_acts, probs)}


# ============================================================================
# LLM Agent (LLaMA, Qwen, Mistral, etc.)
# ============================================================================

class LLMAgent(WebShopAgent):
    """HuggingFace LLM-based WebShop agent."""
    
    # Default system prompt for WebShop (improved version)
    DEFAULT_SYSTEM_PROMPT = """You are an expert online shopping assistant helping users find and purchase products.

You are interacting with a WebShop environment. At each step you'll see:
1. The user's shopping goal/instruction
2. The current webpage observation  
3. A list of VALID ACTIONS you can take

CRITICAL: You must respond with EXACTLY one of the valid actions from the list provided. Copy the action exactly as shown - do not modify it or make up actions.

Common action types in WebShop:
- search[query]: Search for products
- click[item - product name...]: Click on a product (use the EXACT text from valid actions)
- click[Buy Now]: Purchase the current product
- click[Back to Search]: Return to search results
- click[Next >] or click[< Prev]: Navigate pages
- click[size/color/option]: Select product options

Respond with ONLY the action. Copy it exactly from the valid actions list."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        use_flash_attention: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        system_prompt: Optional[str] = None,
        prompt_style: str = "eef",  # "eef" or "verbose"
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.prompt_style = prompt_style
        self.action_history: List[str] = []  # Track action history for EEF style
        
        print(f"Loading LLM from {model_path}...")
        
        # Determine torch dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,
        }
        
        # Try flash attention if requested and available
        if use_flash_attention and torch.cuda.is_available():
            try:
                # Check if flash_attn is installed before requesting it
                import importlib.util
                if importlib.util.find_spec("flash_attn") is not None:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Using Flash Attention 2")
                else:
                    print("Flash Attention not installed, using default attention")
            except Exception as e:
                print(f"Flash Attention check failed ({e}), using default attention")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except ImportError as e:
            # If flash attention was requested but failed, retry without it
            if "flash_attn" in str(e) and "attn_implementation" in model_kwargs:
                print(f"Flash Attention failed to load, retrying with default attention...")
                del model_kwargs["attn_implementation"]
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            else:
                raise
        
        if device != "cuda" or "device_map" not in model_kwargs:
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"LLM loaded successfully on {device}")
    
    def reset_history(self):
        """Reset action history for new episode."""
        self.action_history = []
    
    def _format_prompt(self, obs: str, info: Dict) -> str:
        """Format the prompt for the LLM."""
        goal = info.get('goal', '')
        valid_acts = info['valid']
        
        if self.prompt_style == "eef":
            # EEF paper style - simpler and more direct
            action_history_str = "\n".join(self.action_history[-5:]) if self.action_history else "None"
            
            user_message = f"""Task: {goal}

Current observation:
{obs}

Previous actions:
{action_history_str}

Available actions:
{chr(10).join(valid_acts)}

Choose the next action:"""
        else:
            # Verbose style (original)
            user_message = f"""GOAL: {goal}

CURRENT OBSERVATION:
{obs}

VALID ACTIONS:
{chr(10).join(f'- {act}' for act in valid_acts)}

Select the best action from the valid actions above. Respond with ONLY the action, nothing else."""
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Check if this is a Qwen3 model - disable thinking mode
            is_qwen3 = "qwen3" in self.tokenizer.name_or_path.lower() if hasattr(self.tokenizer, 'name_or_path') else False
            
            try:
                if is_qwen3:
                    # Qwen3 specific: disable thinking with enable_thinking=False
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=False  # Disable thinking for Qwen3
                    )
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
            except TypeError:
                # If enable_thinking parameter is not supported, fall back
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        else:
            # Fallback for tokenizers without chat template
            prompt = f"System: {self.system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        return prompt
    
    def _parse_action(self, response: str, valid_acts: List[str]) -> str:
        """Parse LLM response to extract valid action."""
        import re
        
        response = response.strip()
        
        # Handle Qwen3 thinking tokens - remove <think>...</think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'^<think>.*', '', response, flags=re.DOTALL)
        response = response.strip()
        
        # Remove trailing punctuation that model might add
        response = response.rstrip('.,;:!?')
        
        # Normalize quotes (convert fancy quotes to standard quotes)
        def normalize_quotes(s):
            return s.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'").replace('″', '"').replace('′', "'")
        
        response = normalize_quotes(response)
        valid_acts_normalized = [normalize_quotes(a) for a in valid_acts]
        
        # If response is empty after removing think tokens
        if not response:
            print(f"Warning: Response only contained thinking tokens, using first valid action")
            return valid_acts[0]
        
        # Try exact match first (with normalized quotes)
        if response in valid_acts:
            return response
        for i, norm_act in enumerate(valid_acts_normalized):
            if response == norm_act:
                return valid_acts[i]
        
        # Handle common navigation actions with case-insensitive matching
        # These are frequently output in wrong case by models
        response_lower = response.lower().strip()
        nav_mappings = {
            'click[< prev]': 'click[< Prev]',
            'click[<prev]': 'click[< Prev]',
            'click[prev]': 'click[< Prev]',
            'click[next >]': 'click[Next >]',
            'click[next>]': 'click[Next >]',
            'click[next]': 'click[Next >]',
            'click[back to search]': 'click[Back to Search]',
            'click[buy now]': 'click[Buy Now]',
            'click[buynow]': 'click[Buy Now]',
        }
        
        if response_lower in nav_mappings:
            canonical = nav_mappings[response_lower]
            # Find the actual valid action (might have slight variations)
            for act in valid_acts:
                if act.lower() == canonical.lower():
                    return act
            # If exact canonical not found, return canonical if it's valid
            if canonical in valid_acts:
                return canonical
            # Debug: the navigation action exists but target isn't available
            # This means model wants to go back but there's no Prev button
            # Just silently fall through to other matching
        
        # Case-insensitive exact match for all actions
        for i, act in enumerate(valid_acts):
            if response_lower == act.lower():
                return act
        
        # Try to find valid action in response (response contains the action)
        for i, act in enumerate(valid_acts):
            if act in response:
                return act
            if valid_acts_normalized[i] in response:
                return act
        
        # Try to find if response is contained in a valid action
        for i, act in enumerate(valid_acts):
            if response in act:
                return act
            if response in valid_acts_normalized[i]:
                return act
        
        # Handle ASIN/product ID clicks - model outputs click[B07XYZ] but valid action is click[item - ...]
        if response.startswith('click[B0') or response.startswith('click[b0'):
            asin_match = re.search(r'click\[([Bb]0[A-Za-z0-9]+)\]', response)
            if asin_match:
                asin = asin_match.group(1).upper()
                for act in valid_acts:
                    if asin.lower() in act.lower():
                        return act
                item_acts = [a for a in valid_acts if a.startswith('click[item')]
                if item_acts:
                    return item_acts[0]
        
        # For click actions, try flexible matching
        if response.startswith('click['):
            click_content = response[6:].rstrip(']')
            
            for i, act in enumerate(valid_acts):
                if act.startswith('click['):
                    act_content = act[6:].rstrip(']')
                    act_content_norm = valid_acts_normalized[i][6:].rstrip(']') if valid_acts_normalized[i].startswith('click[') else act_content
                    
                    # Exact content match (including normalized)
                    if click_content == act_content or click_content == act_content_norm:
                        return act
                    
                    # Case-insensitive content match
                    if click_content.lower() == act_content.lower() or click_content.lower() == act_content_norm.lower():
                        return act
                    
                    # Partial content match
                    if len(click_content) > 5:
                        if click_content.lower() in act_content.lower() or click_content.lower() in act_content_norm.lower():
                            return act
                        if act_content.lower() in click_content.lower() or act_content_norm.lower() in click_content.lower():
                            return act
                        # First N chars match
                        min_len = min(len(click_content), len(act_content), 15)
                        if click_content[:min_len].lower() == act_content[:min_len].lower():
                            return act
        
        # Try case-insensitive full match (already done above, but keep for substring matching)
        for i, act in enumerate(valid_acts):
            if act.lower() in response_lower or valid_acts_normalized[i].lower() in response_lower:
                return act
            if response_lower in act.lower() or response_lower in valid_acts_normalized[i].lower():
                return act
        
        # Try to extract search query
        if any(a.startswith('search[') for a in valid_acts):
            search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
            if search_match:
                return f"search[{search_match.group(1)}]"
            if len(response) < 100 and not response.startswith('click'):
                return f"search[{response}]"
        
        # Last resort: find best fuzzy match
        best_match = None
        best_score = 0
        for act in valid_acts:
            response_set = set(response.lower())
            act_set = set(act.lower())
            if len(response_set) > 0:
                score = len(response_set & act_set) / len(response_set | act_set)
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = act
        
        if best_match:
            return best_match
        
        # Fallback: return first valid action
        # Check if this is a navigation action that's just not available
        if response_lower in nav_mappings:
            # Model wanted navigation but it's not available - this is expected behavior
            pass  # Don't print warning for unavailable navigation
        else:
            print(f"Warning: Could not parse action from response: {response[:50]}...")
        return valid_acts[0]
    
    @torch.no_grad()
    def predict(self, obs: str, info: Dict, verbose: bool = False) -> str:
        prompt = self._format_prompt(obs, info)
        
        if verbose:
            print("\n" + "="*60)
            print("PROMPT:")
            print("="*60)
            print(prompt)
            print("="*60)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Build generation kwargs based on temperature
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Only add sampling parameters if temperature > 0
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
        else:
            # Greedy decoding - don't pass temperature/top_p/top_k
            generation_kwargs["do_sample"] = False
        
        outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        if verbose:
            print("\nRAW RESPONSE:")
            print("-"*40)
            print(repr(response))
            print("-"*40)
        
        action = self._parse_action(response, info['valid'])
        
        if verbose:
            print(f"\nPARSED ACTION: {action}")
            print(f"VALID ACTIONS: {info['valid'][:5]}..." if len(info['valid']) > 5 else f"VALID ACTIONS: {info['valid']}")
            print("="*60 + "\n")
        
        # Track action history for EEF style prompts
        self.action_history.append(action)
        
        return action
    
    @torch.no_grad()
    def get_action_distribution(self, obs: str, info: Dict) -> Dict[str, float]:
        """
        Get probability distribution over valid actions by scoring each action.
        """
        prompt = self._format_prompt(obs, info)
        valid_acts = info['valid']
        
        # Score each valid action
        scores = []
        for act in valid_acts:
            full_text = prompt + act
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get log probability of the action tokens
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_inputs['input_ids'].shape[1]
            
            action_logits = logits[0, prompt_len-1:-1, :]
            action_tokens = inputs['input_ids'][0, prompt_len:]
            
            log_probs = F.log_softmax(action_logits, dim=-1)
            action_log_prob = log_probs.gather(1, action_tokens.unsqueeze(1)).sum()
            scores.append(action_log_prob.item())
        
        # Convert to probabilities
        scores_tensor = torch.tensor(scores)
        probs = F.softmax(scores_tensor, dim=0).tolist()
        
        return {a: p for a, p in zip(valid_acts, probs)}


# ============================================================================
# OpenAI API Agent
# ============================================================================

class OpenAIAgent(WebShopAgent):
    """OpenAI API-based WebShop agent."""
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert online shopping assistant helping users find and purchase products.

You are interacting with a WebShop environment. At each step you'll see:
1. The user's shopping goal/instruction
2. The current webpage observation  
3. A list of VALID ACTIONS you can take

CRITICAL: You must respond with EXACTLY one of the valid actions from the list provided. Copy the action exactly as shown - do not modify it or make up actions.

Common action types in WebShop:
- search[query]: Search for products
- click[item - product name...]: Click on a product (use the EXACT text from valid actions)
- click[Buy Now]: Purchase the current product
- click[Back to Search]: Return to search results
- click[Next >] or click[< Prev]: Navigate pages
- click[size/color/option]: Select product options

Respond with ONLY the action. Copy it exactly from the valid actions list."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key argument.")
        
        self.client = OpenAI(api_key=api_key)
        print(f"OpenAI agent initialized with model: {model}")
    
    def _format_user_message(self, obs: str, info: Dict) -> str:
        goal = info.get('goal', '')
        valid_acts = info['valid']
        
        return f"""GOAL: {goal}

CURRENT OBSERVATION:
{obs}

VALID ACTIONS:
{chr(10).join(f'- {act}' for act in valid_acts)}

Select the best action from the valid actions above. Respond with ONLY the action, nothing else."""
    
    def _parse_action(self, response: str, valid_acts: List[str]) -> str:
        """Parse API response to extract valid action."""
        response = response.strip()
        
        # Try exact match first
        if response in valid_acts:
            return response
        
        # Try to find valid action in response
        for act in valid_acts:
            if act in response:
                return act
        
        # Try case-insensitive match
        response_lower = response.lower()
        for act in valid_acts:
            if act.lower() in response_lower:
                return act
        
        # Handle search queries
        if any(a.startswith('search[') for a in valid_acts):
            import re
            search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
            if search_match:
                return f"search[{search_match.group(1)}]"
            if len(response) < 100 and not response.startswith('click'):
                return f"search[{response}]"
        
        print(f"Warning: Could not parse action from response: {response[:100]}...")
        return valid_acts[0]
    
    def predict(self, obs: str, info: Dict, verbose: bool = False) -> str:
        user_message = self._format_user_message(obs, info)
        
        if verbose:
            print(f"User message:\n{user_message}\n")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature,
            max_tokens=256,
        )
        
        response_text = response.choices[0].message.content
        
        if verbose:
            print(f"Response: {response_text}\n")
        
        return self._parse_action(response_text, info['valid'])
    
    def get_action_distribution(self, obs: str, info: Dict) -> Dict[str, float]:
        """
        Get probability distribution over valid actions using logprobs.
        Note: This is an approximation since we can't get exact action probabilities from API.
        """
        # For now, return uniform distribution (API doesn't expose logprobs for completion)
        valid_acts = info['valid']
        return {a: 1.0 / len(valid_acts) for a in valid_acts}


# ============================================================================
# Rule-Based Agent (Baseline)
# ============================================================================

class RuleAgent(WebShopAgent):
    """Simple rule-based baseline agent."""
    
    def __init__(self):
        print("Rule-based agent initialized")
    
    def predict(self, obs: str, info: Dict, verbose: bool = False) -> str:
        valid_acts = info['valid']
        
        # Search: use first valid search action
        if valid_acts[0].startswith('search['):
            return valid_acts[-1]  # Usually the last one has the query
        
        # Click on first item if available
        item_acts = [act for act in valid_acts if act.startswith('click[item - ')]
        if item_acts:
            return item_acts[0]
        
        # Buy if available
        if 'click[buy now]' in valid_acts:
            return 'click[buy now]'
        
        # Otherwise, return first action
        return valid_acts[0]
    
    def get_action_distribution(self, obs: str, info: Dict) -> Dict[str, float]:
        # Rule agent is deterministic
        action = self.predict(obs, info)
        return {a: 1.0 if a == action else 0.0 for a in info['valid']}


# ============================================================================
# Agent Factory
# ============================================================================

def create_agent(args) -> WebShopAgent:
    """Create agent based on command line arguments."""
    
    model_type = args.model_type.lower()
    
    if model_type == "bert_bart":
        return BertBartAgent(
            model_path=args.model_path,
            bart_path=args.bart_path,
            use_bart=args.use_bart,
            use_image=args.use_image,
            use_softmax=args.softmax,
        )
    
    elif model_type in ["llm", "llama", "qwen", "mistral"]:
        return LLMAgent(
            model_path=args.llm_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
            use_flash_attention=args.use_flash_attention,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
            prompt_style=args.prompt_style,
        )
    
    elif model_type == "openai":
        return OpenAIAgent(
            model=args.openai_model,
            temperature=args.temperature,
            api_key=args.openai_api_key,
            system_prompt=args.system_prompt,
        )
    
    elif model_type == "rule":
        return RuleAgent()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Episode Runner
# ============================================================================

def run_episode(
    agent: WebShopAgent,
    env,
    task_idx: Optional[int] = None,
    verbose: bool = False,
    save_trajectory: bool = False,
    save_action_probs: bool = False,
    max_steps: int = 100,
) -> Tuple[float, Optional[Dict]]:
    """
    Run a single episode.
    
    Args:
        agent: WebShop agent
        env: WebShop environment
        task_idx: Task index (None for random)
        verbose: Print actions and observations
        save_trajectory: Save full trajectory
        save_action_probs: Save action probability distributions
        max_steps: Maximum steps per episode
        
    Returns:
        reward: Final reward (0-100 scale)
        trajectory: Trajectory dict if save_trajectory=True, else None
    """
    # Reset agent's action history if it has one (for EEF style prompts)
    if hasattr(agent, 'reset_history'):
        agent.reset_history()
    
    obs, info = env.reset(task_idx)
    
    trajectory = {
        'task_id': task_idx,
        'goal': info.get('goal', ''),
        'steps': [],
        'reward': 0,
        'done': False,
        'success': False
    } if save_trajectory else None
    
    if verbose:
        print("\n" + "#"*70)
        print(f"# EPISODE START - Task {task_idx}")
        print("#"*70)
        print(f"\nGOAL: {info['goal']}")
        print("-"*70)
    
    for step in range(max_steps):
        if verbose:
            print(f"\n>>> STEP {step} <<<")
        
        # Get action (and optionally action distribution)
        action = agent.predict(obs, info, verbose=verbose)
        
        if save_trajectory:
            step_data = {
                'step': step,
                'observation': obs,
                'action_taken': action,
                'valid_actions': info.get('valid', [])
            }
            
            if save_action_probs:
                try:
                    action_probs = agent.get_action_distribution(obs, info)
                    step_data['action_probs'] = action_probs
                except Exception as e:
                    print(f"Warning: Could not get action distribution: {e}")
            
            trajectory['steps'].append(step_data)
        
        if verbose:
            print(f"ACTION TAKEN: {action}")
        
        obs, reward, done, info = env.step(action)
        
        if verbose and done:
            print(f"\n>>> EPISODE DONE <<<")
            print(f"Final reward: {reward * 10:.0f}")
        
        if done:
            # Convert reward from 0-10 to 0-100 range
            reward_100 = reward * 10
            if save_trajectory:
                trajectory['reward'] = reward_100
                trajectory['done'] = True
                trajectory['success'] = (reward_100 == 100.0)
                return reward_100, trajectory
            return reward_100, None
    
    # Episode didn't finish
    if verbose:
        print(f"\n>>> EPISODE TIMEOUT (max_steps={max_steps}) <<<")
    
    if save_trajectory:
        trajectory['reward'] = 0
        trajectory['done'] = False
        trajectory['success'] = False
        return 0, trajectory
    return 0, None


# ============================================================================
# Main Collection Logic
# ============================================================================

def collect_trajectories(args):
    """Main trajectory collection function."""
    
    # Import and setup environment
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split='test')
    print('WebShop environment loaded')
    
    # Configure memory settings
    if args.mem:
        env.env.num_prev_obs = 1
        env.env.num_prev_actions = 5
        print('Memory mode: ON')
    else:
        env.env.num_prev_obs = 0
        env.env.num_prev_actions = 0
        print('Memory mode: OFF')
    
    # Verify deterministic goal generation
    print("\n" + "="*70)
    print("VERIFYING DETERMINISTIC GOAL GENERATION")
    print("="*70)
    test_task = 6
    goals = []
    for i in range(3):
        obs, info = env.reset(test_task)
        goals.append(info['goal'])
    
    if len(set(goals)) == 1:
        print(f"✓ Task {test_task} returns consistent goal across 3 resets")
        print(f"  Goal: {goals[0][:100]}...")
    else:
        print(f"✗ WARNING: Task {test_task} returns DIFFERENT goals!")
        print("  This means goal.py fix is NOT applied correctly")
        for i, g in enumerate(set(goals)):
            print(f"  Variant {i+1}: {g[:100]}...")
        if not args.force:
            print("\nABORTING: Fix goal.py before collecting trajectories!")
            print("Use --force to override this check.")
            sys.exit(1)
    print("="*70 + "\n")
    
    # Create agent
    agent = create_agent(args)
    
    # Also create rule baseline if requested
    rule_agent = RuleAgent() if args.run_rule_baseline else None
    
    print("\n" + "="*70)
    print(f"COLLECTING {args.num_failures} FAILURES")
    print("="*70)
    print(f"  Model type: {args.model_type}")
    print(f"  Target failures: {args.num_failures}")
    print(f"  Max tasks to try: {args.max_tasks}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Save action probs: {args.save_action_probs}")
    print("="*70 + "\n")
    
    # Collection containers
    successes = []
    failures = []
    scores = []
    
    successes_rule = []
    failures_rule = []
    scores_rule = []
    
    task_idx = 0
    
    # Collection loop
    print('idx | reward (model)' + (' | reward (rule)' if args.run_rule_baseline else ''))
    
    while len(failures) < args.num_failures and task_idx < args.max_tasks:
        # Run with main agent
        reward, traj = run_episode(
            agent, env, 
            task_idx=task_idx,
            verbose=args.verbose,
            save_trajectory=args.save_trajectories,
            save_action_probs=args.save_action_probs,
            max_steps=args.max_steps,
        )
        
        scores.append(reward)
        
        if args.save_trajectories:
            if traj['success']:
                successes.append(traj)
            else:
                failures.append(traj)
        else:
            if reward < 100.0:
                failures.append({'task_id': task_idx, 'reward': reward})
        
        # Run rule baseline if requested
        if args.run_rule_baseline:
            reward_rule, traj_rule = run_episode(
                rule_agent, env,
                task_idx=task_idx,
                save_trajectory=args.save_trajectories,
                max_steps=args.max_steps,
            )
            
            scores_rule.append(reward_rule)
            
            if args.save_trajectories:
                if traj_rule['success']:
                    successes_rule.append(traj_rule)
                else:
                    failures_rule.append(traj_rule)
            
            print(f"{task_idx} | {reward:.0f} | {reward_rule:.0f} | "
                  f"Failures: {len(failures)}/{args.num_failures}")
        else:
            print(f"{task_idx} | {reward:.0f} | "
                  f"Failures: {len(failures)}/{args.num_failures}")
        
        task_idx += 1
    
    # Calculate and print statistics
    avg_score = sum(scores) / len(scores)
    success_rate = len([s for s in scores if s == 100.0]) / len(scores) * 100
    
    print("\n" + "="*70)
    print('MODEL RESULTS:')
    print("="*70)
    print(f'  Model type: {args.model_type}')
    print(f'  Tasks run: {task_idx}')
    print(f'  Avg score: {avg_score:.2f}')
    print(f'  Success rate: {success_rate:.1f}%')
    print(f'  Successes: {len(successes)}')
    print(f'  Failures: {len(failures)}')
    print(f'  Target failures: {args.num_failures}')
    
    if len(failures) < args.num_failures:
        print(f'\n  ⚠️  WARNING: Only collected {len(failures)}/{args.num_failures} failures')
        print(f'     Reached max_tasks limit of {args.max_tasks}')
    else:
        print(f'\n  ✓ SUCCESS: Collected {args.num_failures} failures')
    
    # Rule baseline stats
    if args.run_rule_baseline:
        avg_score_rule = sum(scores_rule) / len(scores_rule)
        success_rate_rule = len([s for s in scores_rule if s == 100.0]) / len(scores_rule) * 100
        
        print(f'\nRULE BASELINE RESULTS:')
        print(f'  Avg score: {avg_score_rule:.2f}')
        print(f'  Success rate: {success_rate_rule:.1f}%')
    
    # Save trajectories
    if args.save_trajectories:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = args.model_type
        
        # Save model trajectories
        if successes:
            path = os.path.join(args.output_dir, f'successes_{model_suffix}_{timestamp}.json')
            with open(path, 'w') as f:
                json.dump(successes, f, indent=2)
            print(f'\n✓ Saved {len(successes)} successful trajectories to {path}')
        
        if failures:
            path = os.path.join(args.output_dir, f'failures_{model_suffix}_{timestamp}.json')
            with open(path, 'w') as f:
                json.dump(failures, f, indent=2)
            print(f'✓ Saved {len(failures)} failed trajectories to {path}')
        
        # Save rule baseline trajectories
        if args.run_rule_baseline:
            if successes_rule:
                path = os.path.join(args.output_dir, f'successes_rule_{timestamp}.json')
                with open(path, 'w') as f:
                    json.dump(successes_rule, f, indent=2)
                print(f'✓ Saved {len(successes_rule)} rule successes to {path}')
            
            if failures_rule:
                path = os.path.join(args.output_dir, f'failures_rule_{timestamp}.json')
                with open(path, 'w') as f:
                    json.dump(failures_rule, f, indent=2)
                print(f'✓ Saved {len(failures_rule)} rule failures to {path}')
        
        # Save statistics
        stats = {
            'timestamp': timestamp,
            'model_type': args.model_type,
            'tasks_run': task_idx,
            'target_failures': args.num_failures,
            'config': {
                'temperature': args.temperature,
                'max_steps': args.max_steps,
                'llm_path': getattr(args, 'llm_path', None),
                'openai_model': getattr(args, 'openai_model', None),
            },
            'results': {
                'avg_score': avg_score,
                'success_rate': success_rate,
                'num_successes': len(successes),
                'num_failures': len(failures),
                'scores': scores
            }
        }
        
        if args.run_rule_baseline:
            stats['rule_baseline'] = {
                'avg_score': avg_score_rule,
                'success_rate': success_rate_rule,
                'num_successes': len(successes_rule),
                'num_failures': len(failures_rule),
                'scores': scores_rule
            }
        
        stats_path = os.path.join(args.output_dir, f'statistics_{model_suffix}_{timestamp}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'✓ Saved statistics to {stats_path}')
        
        # Verification
        print("\n" + "="*70)
        print("POST-COLLECTION VERIFICATION")
        print("="*70)
        
        all_trajectories = failures + successes
        mismatches = 0
        checked = 0
        
        for traj in all_trajectories[:min(20, len(all_trajectories))]:
            task_id = traj['task_id']
            saved_goal = traj['goal']
            
            obs, info = env.reset(task_id)
            env_goal = info['goal']
            
            if saved_goal == env_goal:
                print(f"Task {task_id}: ✓ Goal matches")
            else:
                print(f"Task {task_id}: ✗ MISMATCH!")
                print(f"  Saved: {saved_goal[:100]}...")
                print(f"  Env:   {env_goal[:100]}...")
                mismatches += 1
            checked += 1
        
        if mismatches == 0:
            print(f"\n✓ All {checked} sampled trajectories have consistent goals!")
        else:
            print(f"\n✗ WARNING: {mismatches}/{checked} mismatches found!")
        
        print("="*70)
    
    return {
        'successes': successes,
        'failures': failures,
        'scores': scores,
    }


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect WebShop trajectories with multi-model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BERT+BART (original WebShop model)
  python collect_trajectories_multimodel.py --model_type bert_bart --num_failures 500
  
  # LLaMA
  python collect_trajectories_multimodel.py --model_type llama \\
      --llm_path meta-llama/Llama-3.1-8B-Instruct --num_failures 500
  
  # Qwen
  python collect_trajectories_multimodel.py --model_type llm \\
      --llm_path Qwen/Qwen3-8B --num_failures 500
  
  # OpenAI
  python collect_trajectories_multimodel.py --model_type openai \\
      --openai_model gpt-4o-mini --num_failures 500
  
  # Rule baseline only
  python collect_trajectories_multimodel.py --model_type rule --num_failures 500
"""
    )
    
    # Model selection
    parser.add_argument("--model_type", type=str, default="bert_bart",
                       choices=["bert_bart", "llm", "llama", "qwen", "mistral", "openai", "rule"],
                       help="Type of model to use")
    
    # BERT+BART specific
    parser.add_argument("--model_path", type=str, default="./ckpts/web_click/epoch_9/model.pth",
                       help="Path to BERT model checkpoint")
    parser.add_argument("--bart_path", type=str, default="./ckpts/web_search/checkpoint-800",
                       help="Path to BART model checkpoint")
    parser.add_argument("--use_bart", type=bool, default=True,
                       help="Use BART for search queries")
    parser.add_argument("--use_image", type=bool, default=True,
                       help="Use image features in BERT model")
    
    # LLM specific
    parser.add_argument("--llm_path", type=str, default=None,
                       help="HuggingFace model path (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="Torch dtype for LLM")
    parser.add_argument("--use_flash_attention", type=bool, default=True,
                       help="Use flash attention if available")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Max new tokens for LLM generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for LLM (cuda/cpu)")
    parser.add_argument("--prompt_style", type=str, default="eef",
                       choices=["eef", "verbose"],
                       help="Prompt style: 'eef' (simpler, EEF paper style) or 'verbose' (detailed)")
    
    # OpenAI specific
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                       help="OpenAI model name")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    # Common model settings
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0 for greedy)")
    parser.add_argument("--softmax", type=bool, default=True,
                       help="Use softmax sampling for BERT model")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="Custom system prompt for LLM/OpenAI agents")
    
    # Environment settings
    parser.add_argument("--mem", type=int, default=0,
                       help="Use memory mode (0 or 1)")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum steps per episode")
    
    # Collection settings
    parser.add_argument("--save_trajectories", action='store_true',
                       help="Save full trajectories to JSON")
    parser.add_argument("--save_action_probs", action='store_true',
                       help="Save action probability distributions (slower)")
    parser.add_argument("--output_dir", type=str, default="./trajectories",
                       help="Directory to save trajectories")
    parser.add_argument("--num_failures", type=int, default=500,
                       help="Number of FAILURES to collect")
    parser.add_argument("--max_tasks", type=int, default=10000,
                       help="Maximum number of tasks to try")
    parser.add_argument("--run_rule_baseline", action='store_true',
                       help="Also run rule baseline for comparison")
    
    # Debug settings
    parser.add_argument("--verbose", action='store_true',
                       help="Print detailed output during episodes")
    parser.add_argument("--force", action='store_true',
                       help="Force collection even if goal verification fails")
    
    return parser.parse_args()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Validate arguments
    if args.model_type in ["llm", "llama", "qwen", "mistral"] and not args.llm_path:
        print("Error: --llm_path is required for LLM model types")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("TRAJECTORY COLLECTION - MULTI-MODEL")
    print("="*70)
    print(f"Model type: {args.model_type}")
    if args.model_type in ["llm", "llama", "qwen", "mistral"]:
        print(f"LLM path: {args.llm_path}")
    elif args.model_type == "openai":
        print(f"OpenAI model: {args.openai_model}")
    elif args.model_type == "bert_bart":
        print(f"BERT path: {args.model_path}")
        print(f"BART path: {args.bart_path}")
    print("="*70 + "\n")
    
    collect_trajectories(args)