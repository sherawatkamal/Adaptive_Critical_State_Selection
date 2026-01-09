# """
# Standalone LATS for WebShop - No custom imports needed

# Usage:
# 1. Set WEBSHOP_PATH environment variable to your WebShop installation
# 2. Run: python lats_standalone.py

# Or manually edit WEBSHOP_PATH below.
# """

# import os
# import sys
# import json
# import math
# import random
# import time
# from typing import List, Dict, Optional, Tuple
# from dataclasses import dataclass, field
# import numpy as np

# # Load .env file if it exists
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
#     print("✓ Loaded .env file")
# except ImportError:
#     # If python-dotenv not installed, try manual parsing
#     env_path = os.path.join(os.path.dirname(__file__), '.env')
#     if os.path.exists(env_path):
#         with open(env_path) as f:
#             for line in f:
#                 line = line.strip()
#                 if line and not line.startswith('#') and '=' in line:
#                     key, value = line.split('=', 1)
#                     # Remove quotes if present
#                     value = value.strip().strip('"').strip("'")
#                     os.environ[key.strip()] = value
#         print("✓ Loaded .env file (manual parsing)")
#     pass

# # ========== CONFIGURE THIS ==========
# # Set this to your WebShop-master directory
# WEBSHOP_PATH = os.environ.get('WEBSHOP_PATH', os.path.expanduser('~/webshop/WebShop-master'))

# if not os.path.exists(WEBSHOP_PATH):
#     print(f"ERROR: WebShop not found at: {WEBSHOP_PATH}")
#     print("\nPlease set WEBSHOP_PATH:")
#     print("  export WEBSHOP_PATH=/path/to/WebShop-master")
#     print("  python lats_standalone.py")
#     sys.exit(1)

# # Add to path
# sys.path.insert(0, WEBSHOP_PATH)

# # Patch WebShop to add state save/restore if needed
# import pickle

# def patch_webshop():
#     """Add get_state/set_state to WebAgentTextEnv if missing."""
#     from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    
#     if hasattr(WebAgentTextEnv, 'get_state'):
#         return  # Already has it
    
#     def get_state(self):
#         # Get session data
#         if hasattr(self, 'sessions') and hasattr(self, 'session'):
#             session_data = self.sessions.get(self.session, {})
#         else:
#             session_data = {}
        
#         state = {
#             'session': session_data.copy() if isinstance(session_data, dict) else session_data,
#             'steps': getattr(self, 'steps', 0),
#             'session_id': getattr(self, 'session', None)
#         }
#         return pickle.dumps(state)
    
#     def set_state(self, state_bytes):
#         data = pickle.loads(state_bytes)
        
#         # Initialize sessions dict if needed
#         if not hasattr(self, 'sessions'):
#             self.sessions = {}
        
#         session_id = data['session_id']
#         if session_id is not None:
#             if session_id not in self.sessions:
#                 self.sessions[session_id] = {}
#             self.sessions[session_id] = data['session']
#             self.session = session_id
        
#         self.steps = data['steps']
    
#     WebAgentTextEnv.get_state = get_state
#     WebAgentTextEnv.set_state = set_state
#     print("✓ WebShop patched with state save/restore")

# # Now import WebShop
# try:
#     from web_agent_site.envs import WebAgentTextEnv
#     patch_webshop()  # Apply patch if needed
#     print(f"✓ WebShop loaded from: {WEBSHOP_PATH}")
# except ImportError as e:
#     print(f"ERROR: Could not import WebShop: {e}")
#     print(f"  Path tried: {WEBSHOP_PATH}")
#     sys.exit(1)
# # ====================================

# # OpenAI
# try:
#     from openai import OpenAI
# except ImportError:
#     print("ERROR: OpenAI package not found. Install with: pip install openai")
#     sys.exit(1)


# @dataclass
# class TreeNode:
#     """Node in the MCTS tree."""
#     state_id: int
#     observation: str
#     goal: str
#     parent: Optional['TreeNode'] = None
#     action: Optional[str] = None
#     children: List['TreeNode'] = field(default_factory=list)
    
#     visits: int = 0
#     value: float = 0.0
#     is_terminal: bool = False
#     reward: float = 0.0
    
#     valid_actions: List[str] = field(default_factory=list)
#     explored_states: set = field(default_factory=set)  # Track which states we've visited
#     reflection: Optional[str] = None
#     max_expansions: int = 5  # Max times to expand from this node
    
#     def is_fully_expanded(self) -> bool:
#         # Node is fully expanded if we've tried max_expansions times
#         return len(self.children) >= self.max_expansions
    
#     def best_child(self, c_param: float = 1.4) -> 'TreeNode':
#         if not self.children:
#             return self
        
#         choices_weights = []
#         for child in self.children:
#             if child.visits == 0:
#                 # Unvisited child gets infinite priority
#                 choices_weights.append(float('inf'))
#             else:
#                 # UCT formula
#                 exploitation = child.value / child.visits
#                 exploration = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
#                 choices_weights.append(exploitation + exploration)
        
#         return self.children[np.argmax(choices_weights)]
    
#     def update(self, reward: float):
#         self.visits += 1
#         self.value += reward


# class LATS:
#     """Language Agent Tree Search for WebShop."""
    
#     SYSTEM_PROMPT = """You are an expert online shopping assistant using tree search.

# RULES:
# 1. Search with specific keywords from the goal
# 2. Browse multiple products (use Next)
# 3. Verify ALL requirements before buying
# 4. If no results, simplify keywords
# 5. Maximum 20 steps

# Respond with ONLY the action: action[argument]"""

#     def __init__(self, model="gpt-4o", temperature=0.7, max_iterations=5, 
#                  max_depth=20, exploration_constant=1.4, num_simulations=2, verbose=False):
#         self.model = model
#         self.temperature = temperature
#         self.max_iterations = max_iterations
#         self.max_depth = max_depth
#         self.exploration_constant = exploration_constant
#         self.num_simulations = num_simulations
#         self.verbose = verbose
        
#         self.total_api_calls = 0
#         self.state_counter = 0
#         self.state_cache = {}
        
#         # Determine API type and initialize client
#         if model.startswith('gemini'):
#             try:
#                 import google.generativeai as genai
#                 api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
#                 if not api_key:
#                     raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
#                 genai.configure(api_key=api_key)
#                 self.client = genai.GenerativeModel(model)
#                 self.api_type = 'gemini'
#                 print(f"✓ Using Gemini: {model} (FREE!)")
#             except ImportError:
#                 raise ValueError("google-generativeai not installed. Run: pip install google-generativeai")
#         elif model.startswith('gpt'):
#             # OpenAI
#             from openai import OpenAI
#             self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#             self.api_type = 'openai'
#             if self.verbose:
#                 print(f"✓ Using OpenAI: {model}")
#         else:
#             # Local model (HuggingFace)
#             try:
#                 from transformers import AutoModelForCausalLM, AutoTokenizer
#                 import torch
                
#                 print(f"Loading local model: {model}")
#                 self.tokenizer = AutoTokenizer.from_pretrained(model)
#                 self.client = AutoModelForCausalLM.from_pretrained(
#                     model,
#                     torch_dtype=torch.float16,
#                     device_map="auto"
#                 )
#                 self.api_type = 'local'
#                 print(f"✓ Using local model: {model}")
#             except ImportError:
#                 raise ValueError("transformers not installed. Run: pip install transformers torch")
    
#     def get_state_id(self, observation: str) -> int:
#         if observation not in self.state_cache:
#             self.state_cache[observation] = self.state_counter
#             self.state_counter += 1
#         return self.state_cache[observation]
    
#     def format_valid_actions(self, observation: str, valid_actions: List[str]) -> tuple:
#         """Format valid actions to match the correct trajectory format.
        
#         Returns:
#             (formatted_actions, action_mapping)
#             - formatted_actions: Actions shown to LLM with descriptions
#             - action_mapping: Maps descriptive actions back to environment actions
        
#         For product listings, extract product descriptions and format as:
#         click[item - <description>]
        
#         For product pages with options, extract and format options.
        
#         For other actions, format as click[action_name]
#         """
#         formatted = []
#         action_mapping = {}  # Maps LLM action → environment action
        
#         # Check if "buy now" is in valid_actions - if not, there are likely options
#         has_buy_now = any('buy now' in action.lower() for action in valid_actions)
        
#         # Check if we're on a product page (has price, rating, or features/description buttons)
#         is_product_page = any(indicator in observation.lower() for indicator in ['price:', 'rating:', 'features', 'description', 'reviews'])
        
#         if is_product_page and not has_buy_now:
#             # Product page WITHOUT buy now = options need to be selected
#             # Look for options in valid_actions that aren't navigation buttons
#             for action in valid_actions:
#                 action_lower = action.lower()
#                 # Skip navigation actions
#                 if action_lower not in ['back to search', 'next >', '< prev', 'description', 'features', 'reviews', 'search']:
#                     # This is likely an option (color, size, etc.)
#                     formatted_action = f"click[{action_lower}]"
#                     formatted.append(formatted_action)
#                     action_mapping[formatted_action] = f"click[{action_lower}]"
        
#         # Check if we're on a search results page (has product IDs)
#         # Format: "B0125HSS72 [SEP] OneDor 20\" Curly Synthetic..."
#         elif 'B0' in observation and '[SEP]' in observation and 'page' in observation.lower():
#             # Split by [SEP] and look for patterns like "B0XXXXX [SEP] Description"
#             parts = observation.split('[SEP]')
            
#             i = 0
#             while i < len(parts):
#                 part = parts[i].strip()
                
#                 # Check if this part contains a product ID (starts with B0 followed by alphanumeric)
#                 import re
#                 prod_match = re.search(r'\b(B0[A-Z0-9]{8,10})\b', part)
                
#                 if prod_match and i + 1 < len(parts):
#                     prod_id = prod_match.group(1)
#                     # Next part is the product description
#                     description = parts[i + 1].strip()
                    
#                     # Take first line/sentence as description (truncate if too long)
#                     desc_lines = description.split('\n')
#                     desc = desc_lines[0] if desc_lines else description
                    
#                     # Truncate and lowercase
#                     desc_lower = desc.lower()
#                     if len(desc_lower) > 100:
#                         desc_lower = desc_lower[:97] + '...'
                    
#                     if desc_lower and len(desc_lower) > 5:  # Avoid empty/short descriptions
#                         formatted_action = f"click[item - {desc_lower}]"
#                         formatted.append(formatted_action)
#                         # Map back to environment action (lowercase product ID)
#                         action_mapping[formatted_action] = f"click[{prod_id.lower()}]"
                
#                 i += 1
        
#         # Add standard navigation actions
#         for action in valid_actions:
#             action_lower = action.lower()
#             if action_lower in ['back to search', 'next >', '< prev', 'buy now', 
#                                'description', 'features', 'reviews', 'search']:
#                 if action_lower == 'search':
#                     formatted_action = 'search[keywords from goal]'
#                     formatted.append(formatted_action)
#                     action_mapping[formatted_action] = 'search'
#                 else:
#                     formatted_action = f"click[{action_lower}]"
#                     formatted.append(formatted_action)
#                     action_mapping[formatted_action] = f"click[{action_lower}]"
        
#         # If no product actions found, return navigation actions only
#         if not formatted:
#             formatted = ['search[keywords from goal]']
#             action_mapping['search[keywords from goal]'] = 'search'
        
#         return formatted, action_mapping
    
#     def get_action(self, observation: str, goal: str, valid_actions: List[str], 
#                    trajectory: List[str] = None) -> str:
#         self.total_api_calls += 1
        
#         # Format valid actions to match correct trajectory format
#         formatted_actions, action_mapping = self.format_valid_actions(observation, valid_actions)
        
#         # Build trajectory history
#         history = ""
#         if trajectory:
#             history = "Previous actions:\n" + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(trajectory[-3:])) + "\n\n"
        
#         # Use EXACT format from original LATS paper
#         system_instruction = """Webshop 
# Instruction:  
# i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
# [Search]  

# Action: search[3 ounce bright citrus deodorant sensitive skin]
# Observation: 
# [Back to Search] 
# Page 1 (Total results: 50) 
# [Next >] 
# [B078GWRC1J] 
# Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
# $10.99 
# [B078GTKVXY] 
# Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
# $10.99

# Action: click[B078GWRC1J]
# Observation: 
# [Back to Search] 
# [< Prev] 
# scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
# size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
# Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
# Price: $10.99 
# Rating: N.A. 
# [Description] 
# [Features] 
# [Reviews] 
# [Buy Now]  

# Action: click[bright citrus]
# Observation: You have clicked bright citrus. 

# Action: click[3 ounce (pack of 1)]
# Observation: You have clicked 3 ounce (pack of 1). 

# Action: click[Buy Now]

# CRITICAL RULES:
# 1. ALWAYS check the price - it must be LOWER than the budget
# 2. ALWAYS verify ALL attributes match the goal (size, color, features, etc.)
# 3. If a product page has options, SELECT the correct options BEFORE buying
# 4. If unsure, click [Features] or [Description] to verify
# 5. If product doesn't match ALL requirements, click [Back to Search] and try another
# """
        
#         # Format observation to match their style - add brackets around actions
#         obs_formatted = observation[:2000]  # Increase from 1500 to see Buy Now button
#         for action in valid_actions[:10]:
#             if action.lower() not in ['search']:
#                 # Add brackets like original format: [Back to Search], [B078GWRC1J]
#                 obs_formatted = obs_formatted.replace(action, f'[{action}]')
        
#         prompt = f"""{system_instruction}

# {goal}
# {history}Observation:
# {obs_formatted}

# Action:"""

#         if self.verbose:
#             print(f"\n    DEBUG at depth {len(trajectory)}:")
#             print(f"    Observation (first 300 chars): {obs_formatted[:300]}")
#             print(f"    Formatted actions: {formatted_actions[:5]}")
        
#         try:
#             if self.api_type == 'gemini':
#                 # Gemini API
#                 response = self.client.generate_content(
#                     prompt,
#                     generation_config={
#                         'temperature': self.temperature,
#                         'max_output_tokens': 100,
#                     }
#                 )
#                 action = response.text.strip()
#             elif self.api_type == 'openai':
#                 # OpenAI API
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=self.temperature,
#                     max_tokens=100
#                 )
#                 action = response.choices[0].message.content.strip()
#             else:
#                 # Local model (HuggingFace)
#                 inputs = self.tokenizer(prompt, return_tensors="pt").to(self.client.device)
#                 outputs = self.client.generate(
#                     **inputs,
#                     max_new_tokens=100,
#                     temperature=self.temperature,
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
#                 response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#                 action = response_text.strip()
            
#             # Clean up response
#             action = action.replace('Action:', '').replace('action:', '').replace('**', '').strip()
            
#             # Fix double brackets: click[[B078...]] → click[B078...]
#             import re
#             action = re.sub(r'\[\[([^\]]+)\]\]', r'[\1]', action)
            
#             # Extract just the action line if there's extra text
#             lines = action.split('\n')
#             for line in lines:
#                 if line.strip().startswith('search[') or line.strip().startswith('click['):
#                     action = line.strip()
#                     break
#                 # Skip think actions - extract what comes after
#                 if line.strip().startswith('think['):
#                     # Look for mention of click or search in the think
#                     think_content = line.strip()
#                     # Try to extract product ID or action from think
#                     import re
#                     # Look for "click on B0XXX" pattern
#                     click_match = re.search(r'click.*?(B0[A-Z0-9]{8,10})', think_content, re.IGNORECASE)
#                     if click_match:
#                         prod_id = click_match.group(1).lower()
#                         action = f"click[{prod_id}]"
#                         if self.verbose:
#                             print(f"    Extracted from think: {action}")
#                         break
            
#             # If still a think action, reject it and pick first formatted action
#             if action.startswith('think['):
#                 if self.verbose:
#                     print(f"    Think action detected, choosing first formatted action")
#                 # Pick first product from formatted actions if available
#                 for fmt_act in formatted_actions:
#                     if fmt_act.startswith('click[item'):
#                         action = fmt_act
#                         if self.verbose:
#                             print(f"    Using: {action}")
#                         break
            
#             # Validate format - must be search[...] or click[...]
#             if not (action.startswith('search[') or action.startswith('click[')):
#                 # Try to fix common mistakes
#                 if 'search' in action.lower() and '[' not in action:
#                     # Extract query after "search"
#                     query = action.lower().split('search')[-1].strip().strip(':').strip()
#                     if query:
#                         action = f"search[{query}]"
#                     else:
#                         # Fallback: use goal keywords
#                         keywords = ' '.join(goal.split(':')[-1].strip().split()[:5])
#                         action = f"search[{keywords}]"
#                 elif 'click' in action.lower():
#                     target = action.lower().split('click')[-1].strip().strip(':').strip()
#                     action = f"click[{target}]"
#                 else:
#                     # Last resort: search with goal
#                     keywords = ' '.join(goal.split(':')[-1].strip().split()[:5])
#                     action = f"search[{keywords}]"
            
#             if self.verbose:
#                 print(f"    LLM generated: {action}")
            
#             # Map the LLM action back to environment action
#             # Try exact match first
#             if action in action_mapping:
#                 env_action = action_mapping[action]
#                 if self.verbose:
#                     print(f"    Mapped to environment action: {env_action}")
#                 return env_action
            
#             # Try fuzzy matching for click[item - ...] actions
#             if action.startswith('click[item -'):
#                 # Find best matching action in mapping
#                 for llm_act, env_act in action_mapping.items():
#                     if llm_act.startswith('click[item -') and action[:50] in llm_act:
#                         if self.verbose:
#                             print(f"    Fuzzy matched to: {env_act}")
#                         return env_act
            
#             # If no mapping found, return as-is (fallback)
#             if self.verbose:
#                 print(f"    No mapping found, using as-is")
            
#             return action
            
#         except Exception as e:
#             print(f"    API error: {e}")
#             # Fallback: generate search from goal
#             keywords = ' '.join(goal.split(':')[-1].strip().split()[:5])
#             return f"search[{keywords}]"
    
#     def selection(self, node: TreeNode) -> TreeNode:
#         """Select node to expand using UCT - traverse to a leaf."""
#         path = []
#         original_node = node
        
#         while not node.is_terminal:
#             if self.verbose and node == original_node:
#                 print(f"    Selection starting at root: {len(node.children)} children, {node.visits} visits")
            
#             # If node has no children, expand it
#             if len(node.children) == 0:
#                 if self.verbose:
#                     print(f"    → Returning leaf node for expansion (depth={len(path)})")
#                 return node
            
#             # Has children - check if any are unvisited  
#             unvisited = [c for c in node.children if c.visits == 0]
#             if unvisited:
#                 if self.verbose:
#                     print(f"    → Found unvisited child at depth={len(path)}")
#                 return unvisited[0]
            
#             # All children visited - descend to best child
#             best = node.best_child(self.exploration_constant)
#             if self.verbose:
#                 print(f"    → Descending to best child (visits={best.visits}, value={best.value:.1f})")
#             node = best
#             path.append(node.action[:40] if node.action else "?")
        
#         if self.verbose:
#             print(f"    → Reached terminal node at depth={len(path)}")
        
#         return node
    
#     def expansion(self, node: TreeNode, env) -> TreeNode:
#         """Expand node by generating a new action via LLM."""
#         if node.is_terminal or node.is_fully_expanded():
#             return node
        
#         # Build trajectory to this node
#         trajectory = []
#         current = node
#         while current.parent is not None:
#             trajectory.insert(0, current.action)
#             current = current.parent
        
#         # Generate action via LLM
#         action = self.get_action(node.observation, node.goal, node.valid_actions, trajectory)
        
#         if self.verbose:
#             print(f"    LLM action: {action[:60]}")
        
#         # Execute action
#         result = env.step(action)
        
#         # Handle different return formats
#         if isinstance(result, tuple) and len(result) == 4:
#             obs, reward, done, info = result
#             if info is not None and isinstance(info, dict):
#                 valid_actions_raw = info.get('valid', {})
#             else:
#                 valid_actions_raw = getattr(env, 'get_available_actions', lambda: {})()
#         else:
#             obs = str(result[0]) if isinstance(result, tuple) else str(result)
#             reward = 0
#             done = False
#             valid_actions_raw = getattr(env, 'get_available_actions', lambda: {})()
        
#         # Extract clickables
#         if isinstance(valid_actions_raw, dict):
#             valid_actions = valid_actions_raw.get('clickables', [])
#         else:
#             valid_actions = valid_actions_raw if isinstance(valid_actions_raw, list) else []
        
#         # Create child node
#         state_id = self.get_state_id(obs)
#         child = TreeNode(
#             state_id=state_id,
#             observation=obs,
#             goal=node.goal,
#             parent=node,
#             action=action,
#             valid_actions=valid_actions,
#             is_terminal=done,
#             reward=reward * 10
#         )
        
#         if self.verbose and len(node.children) < 2:  # Only show for first few expansions
#             print(f"      → New state (reward={reward*10:.1f}, done={done})")
#             print(f"      → Valid actions now: {valid_actions[:3]}{'...' if len(valid_actions) > 3 else ''}")
        
#         node.children.append(child)
#         node.explored_states.add(state_id)
        
#         return child
    
#     def simulation(self, node: TreeNode, env, max_steps: int = 15) -> float:
#         if node.is_terminal:
#             return node.reward
        
#         trajectory = []
#         current = node
#         while current.parent is not None:
#             trajectory.insert(0, current.action)
#             current = current.parent
        
#         step = 0
#         done = False
#         total_reward = 0.0
#         current_obs = node.observation
#         current_valid = node.valid_actions
        
#         while not done and step < max_steps:
#             action = self.get_action(current_obs, node.goal, current_valid, trajectory)
            
#             result = env.step(action)
            
#             # Handle different return formats
#             if isinstance(result, tuple) and len(result) == 4:
#                 obs, reward, done, info = result
#                 if info is not None and isinstance(info, dict):
#                     valid_actions_raw = info.get('valid', {})
#                 else:
#                     valid_actions_raw = getattr(env, 'get_available_actions', lambda: {})()
#             else:
#                 obs = str(result[0]) if isinstance(result, tuple) else str(result)
#                 reward = 0
#                 done = False
#                 valid_actions_raw = getattr(env, 'get_available_actions', lambda: {})()
            
#             # Extract clickables from WebShop format
#             if isinstance(valid_actions_raw, dict):
#                 current_valid = valid_actions_raw.get('clickables', [])
#             else:
#                 current_valid = valid_actions_raw if isinstance(valid_actions_raw, list) else []
            
#             total_reward = reward * 10
#             trajectory.append(action)
#             current_obs = obs
#             step += 1
        
#         return total_reward
    
#     def backpropagation(self, node: TreeNode, reward: float):
#         while node is not None:
#             node.update(reward)
#             node = node.parent
    
#     def search(self, env, task_idx: int) -> Dict:
#         """Run LATS to solve a task."""
#         # Track all reasoning data
#         reasoning_data = {
#             'mcts_iterations': [],
#             'tree_stats': {},
#             'all_rollouts': []
#         }
        
#         # Reset environment - handle different return formats
#         result = env.reset(task_idx)
        
#         # Debug: see what reset returns
#         if self.verbose:
#             print(f"DEBUG: reset() returned type: {type(result)}")
#             if isinstance(result, tuple):
#                 print(f"DEBUG: tuple length: {len(result)}")
        
#         # Handle different WebShop versions
#         obs = None
#         info = None
        
#         if result is None:
#             # Environment doesn't return anything from reset
#             obs = getattr(env, 'observation', '')
#             info = None
#         elif isinstance(result, tuple) and len(result) == 2:
#             # New format: (obs, info)
#             obs, info = result
#         elif isinstance(result, str):
#             # Old format: just obs string
#             obs = result
#             info = None
#         else:
#             # Fallback
#             obs = str(result)
#             info = None
        
#         # Extract goal and valid actions
#         if info is not None and isinstance(info, dict):
#             goal = info.get('goal', 'Unknown goal')
#             valid_actions_raw = info.get('valid', [])
#         else:
#             # Get from environment attributes
#             goal = getattr(env, 'instruction_text', 'Unknown goal')
#             valid_actions_raw = getattr(env, 'get_available_actions', lambda: {})()
        
#         # Extract actual actions from WebShop format
#         if isinstance(valid_actions_raw, dict):
#             # WebShop returns {'has_search_bar': True, 'clickables': ['action1', 'action2']}
#             valid_actions = valid_actions_raw.get('clickables', [])
#         elif isinstance(valid_actions_raw, list):
#             valid_actions = valid_actions_raw
#         else:
#             valid_actions = []
        
#         if not valid_actions:
#             valid_actions = ['search[product]']  # Fallback
        
#         if self.verbose:
#             print(f"\n=== LATS Task {task_idx} ===")
#             print(f"Goal: {goal}")
#             print(f"Valid actions: {len(valid_actions)} available")
#             if len(valid_actions) < 10:  # Only print if not too many
#                 print(f"Actions: {valid_actions}")
#             print(f"DEBUG: First action type: {type(valid_actions[0]) if valid_actions else 'N/A'}")
#             print(f"DEBUG: First action: {valid_actions[0][:50] if valid_actions else 'N/A'}")
        
#         root = TreeNode(
#             state_id=self.get_state_id(obs),
#             observation=obs,
#             goal=goal,
#             valid_actions=valid_actions
#         )
        
#         # MCTS iterations
#         for iteration in range(self.max_iterations):
#             if self.verbose:
#                 print(f"  Iteration {iteration + 1}/{self.max_iterations}")
#                 # Show tree stats
#                 print(f"    Root: {len(root.children)} children, {root.visits} visits, value={root.value:.1f}")
#                 if len(root.children) > 0:
#                     for i, child in enumerate(root.children[:3]):  # Show first 3
#                         print(f"      Child {i}: visits={child.visits}, value={child.value:.1f}, action={child.action[:40]}")
            
#             # Track this iteration
#             iteration_data = {
#                 'iteration': iteration + 1,
#                 'selected_path': [],
#                 'expanded_action': None,
#                 'simulation_rewards': [],
#                 'nodes_visited': 0
#             }
            
#             # Save state
#             env_state = env.get_state()
            
#             # Selection
#             node = self.selection(root)
            
#             # Track selection path
#             path_node = node
#             while path_node.parent is not None:
#                 iteration_data['selected_path'].insert(0, {
#                     'action': path_node.action,
#                     'visits': path_node.visits,
#                     'value': path_node.value
#                 })
#                 path_node = path_node.parent
            
#             # Expansion
#             if not node.is_terminal:
#                 node = self.expansion(node, env)
#                 iteration_data['expanded_action'] = node.action
            
#             # Simulation
#             total_reward = 0.0
#             for sim_idx in range(self.num_simulations):
#                 env.set_state(env_state)
#                 reward = self.simulation(node, env)
#                 total_reward += reward
#                 iteration_data['simulation_rewards'].append(reward)
                
#                 # Store rollout details
#                 reasoning_data['all_rollouts'].append({
#                     'iteration': iteration + 1,
#                     'simulation': sim_idx + 1,
#                     'reward': reward,
#                     'from_node_visits': node.visits
#                 })
            
#             avg_reward = total_reward / self.num_simulations
#             iteration_data['avg_reward'] = avg_reward
            
#             # Backpropagation
#             self.backpropagation(node, avg_reward)
            
#             # Restore
#             env.set_state(env_state)
            
#             # Save iteration data
#             reasoning_data['mcts_iterations'].append(iteration_data)
        
#         # Extract best path
#         trajectory = []
#         node = root
#         env.set_state(env.get_state())
        
#         # Collect tree statistics
#         def collect_tree_stats(node, depth=0):
#             stats = {
#                 'depth': depth,
#                 'visits': node.visits,
#                 'value': node.value,
#                 'num_children': len(node.children),
#                 'is_terminal': node.is_terminal,
#                 'action': node.action,
#                 'children': []
#             }
#             for child in node.children:
#                 stats['children'].append(collect_tree_stats(child, depth + 1))
#             return stats
        
#         reasoning_data['tree_stats'] = collect_tree_stats(root)
        
#         total_reward = 0.0
#         for step in range(self.max_depth):
#             if node.is_terminal or len(node.children) == 0:
#                 break
            
#             node = max(node.children, key=lambda c: c.visits)
            
#             result = env.step(node.action)
            
#             # Handle different return formats
#             if isinstance(result, tuple) and len(result) == 4:
#                 obs, reward, done, info = result
#             else:
#                 obs = str(result[0]) if isinstance(result, tuple) else str(result)
#                 reward = 0
#                 done = False
            
#             trajectory.append({
#                 'step': step,
#                 'action': node.action,
#                 'observation': obs[:200],
#                 'reward': reward * 10
#             })
            
#             total_reward = reward * 10
#             if done:
#                 break
        
#         result = {
#             'task_idx': task_idx,
#             'goal': goal,
#             'trajectory': trajectory,
#             'final_reward': total_reward,
#             'success': total_reward >= 100.0,
#             'num_steps': len(trajectory),
#             'api_calls': self.total_api_calls,
#             'iterations': self.max_iterations,
#             'reasoning_data': reasoning_data  # <-- Raw LATS reasoning data
#         }
        
#         return result


# def main():
#     """Test LATS on WebShop."""
#     import argparse
    
#     parser = argparse.ArgumentParser(description='LATS for WebShop')
#     parser.add_argument('--model', type=str, default='gemini-2.0-flash-exp',
#                         help='Model to use: gemini-2.0-flash-exp, gpt-4o-mini, gpt-4o, or local model path')
#     parser.add_argument('--num_tasks', type=int, default=5,
#                         help='Number of tasks to test (default: 5)')
#     parser.add_argument('--start_task', type=int, default=0,
#                         help='Starting task index (default: 0)')
#     parser.add_argument('--iterations', type=int, default=5,
#                         help='LATS iterations per task (default: 5)')
#     parser.add_argument('--simulations', type=int, default=2,
#                         help='Simulations per expansion (default: 2)')
#     parser.add_argument('--verbose', action='store_true', default=True,
#                         help='Print detailed output')
#     args = parser.parse_args()
    
#     print("=" * 80)
#     print("LATS WebShop Standalone Test")
#     print("=" * 80)
#     print(f"Model: {args.model}")
#     print(f"Tasks: {args.num_tasks}")
#     print(f"Iterations: {args.iterations}, Simulations: {args.simulations}")
#     print("=" * 80)
    
#     model = args.model
    
#     # Check API key based on model type
#     if model.startswith('gemini'):
#         if not (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')):
#             print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set")
#             print("  Get free key from: https://aistudio.google.com/app/apikey")
#             print("  export GEMINI_API_KEY='your-key-here'")
#             sys.exit(1)
#     elif model.startswith('gpt'):
#         if not os.getenv('OPENAI_API_KEY'):
#             print("ERROR: OPENAI_API_KEY not set")
#             print("  export OPENAI_API_KEY='your-key-here'")
#             sys.exit(1)
#     else:
#         # Local model - no API key needed
#         print(f"Using local model: {model}")
    
#     # Initialize environment
#     print("Initializing WebShop...")
#     env = WebAgentTextEnv(observation_mode='text', human_goals=True)
    
#     # Test state save/restore (must reset first to initialize state)
#     print("Testing state save/restore...")
#     env.reset(0)  # Initialize environment state
#     try:
#         state = env.get_state()
#         env.set_state(state)
#         print("✓ State save/restore works")
#     except Exception as e:
#         print(f"✗ State save/restore failed: {e}")
#         print("\nLATS requires get_state() and set_state() methods.")
#         print("Your WebShop version may not support this.")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
    
#     # Initialize LATS
#     print("Initializing LATS...")
#     lats = LATS(
#         model=model,
#         temperature=0.7,
#         max_iterations=args.iterations,
#         num_simulations=args.simulations,
#         verbose=args.verbose
#     )
    
#     # Test on specified number of tasks
#     num_tasks = args.num_tasks
#     results = []
    
#     print(f"\nTesting on {num_tasks} tasks (starting from task {args.start_task})...")
#     print("=" * 80)
    
#     for i in range(args.start_task, args.start_task + num_tasks):
#         try:
#             result = lats.search(env, i)
#             results.append(result)
#             print(f"\nTask {i}: {'✓ SUCCESS' if result['success'] else '✗ FAILURE'}")
#             print(f"  Reward: {result['final_reward']}")
#             print(f"  Steps: {result['num_steps']}")
#             print(f"  API calls: {result['api_calls']}")
#         except Exception as e:
#             print(f"\nTask {i}: ERROR - {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Summary
#     successes = sum(1 for r in results if r['success'])
#     total_calls = sum(r['api_calls'] for r in results)
    
#     print("\n" + "=" * 80)
#     print("SUMMARY")
#     print("=" * 80)
#     if len(results) > 0:
#         print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
#         print(f"Total API calls: {total_calls}")
#         print(f"Avg API calls: {total_calls/len(results):.1f}")
        
#         # Cost estimate
#         cost = total_calls * 0.002  # Rough estimate
#         print(f"Estimated cost: ${cost:.2f}")
#         print(f"Projected for 3000 tasks: ${cost * 3000 / len(results):.2f}")
#     else:
#         print("No results collected (all tasks failed)")
#     print("=" * 80)
    
#     # Save results
#     with open('lats_test_results.json', 'w') as f:
#         json.dump(results, f, indent=2)
#     print("\nResults saved to: lats_test_results.json")
    
#     # Also save reasoning data separately for easier analysis
#     reasoning_only = [
#         {
#             'task_idx': r['task_idx'],
#             'goal': r['goal'],
#             'success': r['success'],
#             'reasoning_data': r['reasoning_data']
#         }
#         for r in results
#     ]
#     with open('lats_reasoning_data.json', 'w') as f:
#         json.dump(reasoning_only, f, indent=2)
#     print("Reasoning data saved to: lats_reasoning_data.json")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
LATS (Language Agent Tree Search) for WebShop
Clean implementation that uses WebShop's native valid_actions format
"""

import os
import sys
import json
import math
import argparse
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# WebShop imports
sys.path.insert(0, os.path.expanduser('~/SageWebShop2'))
from web_agent_site.envs import WebAgentTextEnv

# API imports
import google.generativeai as genai
from openai import OpenAI

class TreeNode:
    """Node in the MCTS tree"""
    def __init__(self, state: Dict, parent=None):
        self.state = state  # {'observation': str, 'action': str, 'reward': float}
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0.0
    
    def uct(self, exploration_constant: float = 1.4) -> float:
        """UCT formula for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.4):
        """Select best child using UCT"""
        return max(self.children, key=lambda c: c.uct(exploration_constant))


class LATSAgent:
    """LATS agent for WebShop"""
    
    def __init__(self, model: str, temperature: float = 0.7, verbose: bool = True):
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.total_api_calls = 0
        
        # Initialize API client
        if model.startswith('gemini'):
            self.api_type = 'gemini'
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.client = genai.GenerativeModel(model)
        elif model.startswith('gpt'):
            self.api_type = 'openai'
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            # Local model
            self.api_type = 'local'
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.client = AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype='auto',
                device_map="auto"
            )
    
    def get_action(self, observation: str, goal: str, valid_actions: List[str], 
                   trajectory: List[str] = None) -> str:
        """
        Get action from LLM using WebShop's native valid_actions.
        
        Args:
            observation: Current observation from WebShop
            goal: Shopping goal/instruction
            valid_actions: List of valid actions from WebShop (ALREADY FORMATTED!)
            trajectory: Previous actions for context
            
        Returns:
            Selected action (must be from valid_actions)
        """
        self.total_api_calls += 1
        
        # Build history from trajectory
        history = ""
        if trajectory and len(trajectory) > 0:
            history = "Previous steps:\n" + "\n".join(trajectory[-5:]) + "\n\n"
        
        # System instruction with example
        system_instruction = """You are an expert online shopping assistant helping users find and purchase products on WebShop.

Example task:
Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

Step 1 - Search:
Available actions: [search[3 ounce bright citrus deodorant], search[bright citrus deodorant sensitive skin], ...]
Action: search[bright citrus deodorant sensitive skin]

Step 2 - Select product:
Available actions: [click[item - bright citrus deodorant by earth mama...], click[item - ginger fresh deodorant...], click[next >], ...]
Action: click[item - bright citrus deodorant by earth mama | natural and safe for sensitive skin, pregnancy and breastfeeding, contains organic calendula 3-ounce]

Step 3 - Select scent option:
Available actions: [click[bright citrus], click[calming lavender], click[3 ounce (pack of 1)], click[buy now], ...]
Action: click[bright citrus]

Step 4 - Select size option:
Available actions: [click[3 ounce (pack of 1)], click[3-ounce (2-pack)], click[buy now], ...]
Action: click[3 ounce (pack of 1)]

Step 5 - Purchase:
Available actions: [click[buy now], click[back to search], ...]
Action: click[buy now]

CRITICAL RULES - READ CAREFULLY:
1. PRICE CHECK: Product price MUST be lower than the budget stated in the goal
2. OPTIONS FIRST: If you see actions like click[1], click[2], click[red], click[large], etc. - these are PRODUCT OPTIONS
   - You MUST select the correct option(s) BEFORE clicking [buy now]
   - Common options: quantity (click[1], click[2]), color, size, style, scent
   - Example: If goal says "60 capsules" and you see click[1], click[2], click[3] - pick click[1] for single bottle
3. VERIFY REQUIREMENTS: Product must match ALL criteria (brand, features, size, price, etc.)
4. EXPLORE: Use [next >] to see more products if current ones don't match
5. EXACT MATCH: Choose ONE action from the "Available actions" list - type it EXACTLY as shown
6. NO BUY WITHOUT OPTIONS: If options are present (numbers, colors, sizes), NEVER click [buy now] until you've selected them!

REMEMBER: Options appear as simple clicks like click[1], click[red], click[small] - select these BEFORE [buy now]!
"""
        
        # Prioritize options over navigation - show product options FIRST
        navigation_actions = ['click[back to search]', 'click[< prev]', 'click[next >]', 
                            'click[description]', 'click[features]', 'click[reviews]']
        
        # Separate options from navigation
        options = [a for a in valid_actions if a not in navigation_actions]
        navigation = [a for a in valid_actions if a in navigation_actions]
        
        # Prioritized list: options first, then navigation
        prioritized_actions = options + navigation
        
        # Show up to 30 actions (more than before to capture all options)
        actions_list = "\n".join(f"{i+1}. {action}" for i, action in enumerate(prioritized_actions[:30]))
        
        # Create prompt
        prompt = f"""{system_instruction}

Current task:
{goal}

{history}Current observation:
{observation[:1500]}

Available actions (YOU MUST CHOOSE ONE OF THESE):
{actions_list}

Choose the BEST action by typing it EXACTLY as shown above.
Your action:"""
        
        if self.verbose:
            print(f"\n    LLM call (depth {len(trajectory) if trajectory else 0}):")
            print(f"    First 5 valid actions: {valid_actions[:5]}")
        
        try:
            # Get LLM response
            if self.api_type == 'gemini':
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': 150
                    }
                )
                action = response.text.strip()
            elif self.api_type == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=150
                )
                action = response.choices[0].message.content.strip()
            else:
                # Local model
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.client.device)
                outputs = self.client.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                action = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            
            # Clean up response
            action = action.replace('Action:', '').replace('action:', '').strip()
            action = action.replace('**', '').strip()
            
            # Remove numbering if present (e.g., "1. search[...]" -> "search[...]")
            import re
            action = re.sub(r'^\d+\.\s*', '', action)
            
            # Extract first line if multi-line
            if '\n' in action:
                action = action.split('\n')[0].strip()
            
            # Try to find exact match in valid_actions
            if action in valid_actions:
                if self.verbose:
                    print(f"    ✓ Selected: {action[:80]}")
                return action
            
            # Try case-insensitive match
            action_lower = action.lower()
            for valid_action in valid_actions:
                if valid_action.lower() == action_lower:
                    if self.verbose:
                        print(f"    ✓ Case-matched: {valid_action[:80]}")
                    return valid_action
            
            # Try partial match for long action strings
            for valid_action in valid_actions:
                if len(action) > 20 and action[:20].lower() in valid_action.lower():
                    if self.verbose:
                        print(f"    ~ Partial match: {valid_action[:80]}")
                    return valid_action
                if len(valid_action) > 20 and valid_action[:20].lower() in action.lower():
                    if self.verbose:
                        print(f"    ~ Partial match: {valid_action[:80]}")
                    return valid_action
            
            # Fallback: use first valid action (safer than failing)
            if self.verbose:
                print(f"    ✗ No match for '{action[:80]}', using first valid action")
            return valid_actions[0]
            
        except Exception as e:
            print(f"    ERROR in get_action: {e}")
            # Fallback: return first valid action
            return valid_actions[0] if valid_actions else 'search[product]'
    
    def collect_episode(self, env, idx, max_steps=20, verbose=False):
        """
        Collect one trajectory using LATS.
        
        Returns trajectory in same format as collect_trajectories_gpt_sampled.py
        """
        obs, info = env.reset(idx)
        goal = info['goal']
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"LATS Task {idx}")
            print(f"Goal: {goal}")
            print(f"{'='*80}")
        
        # Initialize trajectory in EXACT format from collect_trajectories_gpt_sampled.py
        trajectory = {
            'idx': idx,
            'goal': goal,
            'steps': [],
            'final_reward': 0,
            'success': False,
            'num_steps': 0,
            'model': self.model
        }
        
        # Run LATS search to get best action sequence
        result = self.search(env, idx, iterations=10, simulations=2, exploration_constant=1.4)
        
        # Extract steps from LATS result
        if result['trajectory']:
            for i, step_data in enumerate(result['trajectory']):
                trajectory['steps'].append({
                    'step': i,
                    'observation': step_data['observation'],
                    'valid_actions': step_data.get('valid_actions', []),
                    'action': step_data['action'],
                    'raw_response': step_data['action']  # LATS doesn't have separate raw response
                })
        
        trajectory['final_reward'] = result['final_reward']
        trajectory['success'] = result['success']
        trajectory['num_steps'] = result['num_steps']
        
        if verbose:
            status = '✓ SUCCESS' if trajectory['success'] else '✗ FAILURE'
            print(f"\nTask {idx}: {status}")
            print(f"  Reward: {trajectory['final_reward']}")
            print(f"  Steps: {trajectory['num_steps']}")
        
        return trajectory
    
    def search(self, env, task_idx: int, iterations: int = 10, simulations: int = 2,
               exploration_constant: float = 1.4) -> Dict:
        """
        Run LATS search for a WebShop task.
        
        Returns:
            Dictionary with trajectory data compatible with collect_trajectories_gpt_sampled.py format
        """
        # Reset environment - WebEnv returns (obs, info)
        obs, info = env.reset(task_idx)
        goal = info['goal']
        valid_actions = info['valid']  # WebEnv stores valid actions in info['valid']
        
        if self.verbose:
            print(f"  Starting LATS with {iterations} iterations, {simulations} simulations")
            print(f"  Initial valid actions: {len(valid_actions)}")
        
        # Initialize root node
        root = TreeNode(state={
            'observation': obs,
            'action': None,
            'reward': 0.0,
            'valid_actions': valid_actions
        })
        
        # MCTS iterations
        for iteration in range(iterations):
            if self.verbose and iteration % 3 == 0:
                print(f"  Iteration {iteration + 1}/{iterations}")
            
            # 1. Selection
            node = self._select(root, exploration_constant)
            
            # 2. Expansion
            if not node.is_terminal:
                self._expand(node, env, task_idx, goal, simulations)
            
            # 3. Backpropagation
            self._backpropagate(node, node.value)
            
            # Check if we found a successful trajectory
            if self._has_successful_child(root):
                if self.verbose:
                    print(f"  ✓ Found successful trajectory at iteration {iteration + 1}")
                break
        
        # Extract best trajectory
        best_node = self._get_best_terminal_node(root)
        trajectory_steps = self._collect_trajectory(best_node)
        
        # Format result in compatible structure
        result = {
            'task_idx': task_idx,
            'goal': goal,
            'success': best_node.reward >= 1.0,
            'final_reward': best_node.reward * 100,  # Scale to 0-100 like GPT collector
            'num_steps': len(trajectory_steps),
            'trajectory': trajectory_steps,
            'api_calls': self.total_api_calls
        }
        
        return result
    
    def _select(self, node: TreeNode, exploration_constant: float) -> TreeNode:
        """Select a leaf node using UCT"""
        while node.children:
            # If all children are terminal, return current node
            if all(child.is_terminal for child in node.children):
                return node
            
            # Select best non-terminal child
            non_terminal = [c for c in node.children if not c.is_terminal]
            if non_terminal:
                # Prefer unvisited children
                unvisited = [c for c in non_terminal if c.visits == 0]
                if unvisited:
                    node = random.choice(unvisited)
                else:
                    node = max(non_terminal, key=lambda c: c.uct(exploration_constant))
            else:
                return node
        
        return node
    
    def _expand(self, node: TreeNode, env, task_idx: int, goal: str, simulations: int):
        """Expand node by generating child states"""
        # Reconstruct state by replaying actions
        env.reset(task_idx)
        trajectory = []
        current = node
        path = []
        
        while current.parent is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        
        # Replay actions to reach this state
        last_info = None
        for n in path:
            if n.state['action']:
                trajectory.append(f"Action: {n.state['action']}")
                step_result = env.step(n.state['action'])
                if isinstance(step_result, tuple):
                    obs_str, reward, done, last_info = step_result if len(step_result) == 4 else (step_result[0], 0, False, {})
                trajectory.append(f"Observation: {obs_str[:200]}")
        
        # Get current state
        current_obs = node.state['observation']
        
        # Get valid actions from last step info
        valid_actions = last_info.get('valid', []) if last_info else node.state.get('valid_actions', [])
        
        if not valid_actions:
            # No valid actions, mark as terminal
            node.is_terminal = True
            return
        
        # Generate multiple actions (simulations)
        for _ in range(simulations):
            action = self.get_action(current_obs, goal, valid_actions, trajectory)
            
            # Take action in environment
            step_result = env.step(action)
            
            # WebEnv step returns: observation, reward, done, info
            if isinstance(step_result, tuple):
                if len(step_result) == 4:
                    obs_str, reward, done, info = step_result
                elif len(step_result) == 3:
                    obs_str, reward, done = step_result
                    info = {}
                else:
                    obs_str = step_result[0]
                    reward = 0.0
                    done = False
                    info = {}
            else:
                obs_str = step_result
                reward = 0.0
                done = False
                info = {}
            
            # Get valid actions for next state
            next_valid_actions = info.get('valid', [])
            
            # Create child node
            child = TreeNode(
                state={
                    'observation': obs_str,
                    'action': action,
                    'reward': reward,
                    'valid_actions': next_valid_actions
                },
                parent=node
            )
            
            child.is_terminal = done or reward >= 1.0
            child.reward = reward
            child.value = reward * 10  # Scale reward for value
            
            node.children.append(child)
            
            # Reset for next simulation
            if _ < simulations - 1:
                env.reset(task_idx)
                for n in path:
                    if n.state['action']:
                        env.step(n.state['action'])
    
    def _backpropagate(self, node: TreeNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _has_successful_child(self, node: TreeNode) -> bool:
        """Check if any descendant has reward >= 1.0"""
        if node.reward >= 1.0:
            return True
        for child in node.children:
            if self._has_successful_child(child):
                return True
        return False
    
    def _get_best_terminal_node(self, root: TreeNode) -> TreeNode:
        """Get best terminal node from tree"""
        terminals = []
        self._collect_terminals(root, terminals)
        
        if not terminals:
            return root
        
        # Prefer successful terminals
        successful = [n for n in terminals if n.reward >= 1.0]
        if successful:
            return max(successful, key=lambda n: n.reward)
        
        # Otherwise, return highest reward
        return max(terminals, key=lambda n: n.reward)
    
    def _collect_terminals(self, node: TreeNode, terminals: List[TreeNode]):
        """Collect all terminal nodes"""
        if node.is_terminal or not node.children:
            terminals.append(node)
        else:
            for child in node.children:
                self._collect_terminals(child, terminals)
    
    def _collect_trajectory(self, node: TreeNode) -> List[Dict]:
        """Collect trajectory from root to node in format compatible with GPT collector"""
        trajectory = []
        path = []
        
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        
        for i, n in enumerate(path[1:]):  # Skip root
            trajectory.append({
                'action': n.state['action'],
                'observation': n.state['observation'][:1000],  # Truncate like GPT collector
                'valid_actions': n.state.get('valid_actions', []),
                'reward': n.state['reward']
            })
        
        return trajectory


def main():
    parser = argparse.ArgumentParser(description='LATS for WebShop')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model: gpt-4o, gpt-4o-mini, gemini-2.0-flash-exp, or local path')
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='Number of tasks')
    parser.add_argument('--start_task', type=int, default=0,
                        help='Starting task index')
    parser.add_argument('--iterations', type=int, default=10,
                        help='LATS iterations per task')
    parser.add_argument('--simulations', type=int, default=2,
                        help='Simulations per expansion')
    parser.add_argument('--output', type=str, default='./lats_trajectories.json',
                        help='Output file path')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 80)
    print("LATS WebShop - GPT Collector Compatible")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.num_tasks} (starting from {args.start_task})")
    print(f"Iterations: {args.iterations}, Simulations: {args.simulations}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Initialize environment using WebEnv (same as GPT collector)
    print("\nInitializing WebShop...")
    from train_rl import parse_args as webenv_args
    from env import WebEnv
    
    env_args = webenv_args()[0]
    env = WebEnv(env_args, split='train')
    env.env.num_prev_obs = 0
    env.env.num_prev_actions = 0
    print("✓ WebShop ready")
    
    # Initialize agent
    print(f"\nInitializing LATS agent with {args.model}...")
    agent = LATSAgent(model=args.model, verbose=args.verbose)
    print("✓ Agent ready")
    
    # Collect trajectories
    print(f"\nCollecting {args.num_tasks} trajectories...")
    successful_trajectories = []
    all_trajectories = []
    
    for i in range(args.start_task, args.start_task + args.num_tasks):
        try:
            trajectory = agent.collect_episode(env, i, max_steps=20, verbose=args.verbose)
            all_trajectories.append(trajectory)
            
            if trajectory['success']:
                successful_trajectories.append(trajectory)
                
        except Exception as e:
            print(f"\nTask {i}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary (same format as GPT collector)
    expert_successes = successful_trajectories
    total_tasks = len(all_trajectories)
    
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    if total_tasks > 0:
        print(f"Total tasks: {total_tasks}")
        print(f"Expert success: {len(expert_successes)} ({len(expert_successes)/total_tasks*100:.1f}%)")
        print(f"Expert failure: {total_tasks - len(expert_successes)} ({(total_tasks - len(expert_successes))/total_tasks*100:.1f}%)")
        
        if expert_successes:
            avg_reward = sum(t['final_reward'] for t in expert_successes) / len(expert_successes)
            avg_steps = sum(t['num_steps'] for t in expert_successes) / len(expert_successes)
            print(f"\n--- Successful Trajectories Stats ---")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.1f}")
        
        total_calls = agent.total_api_calls
        print(f"\n--- Performance ---")
        print(f"Total API calls: {total_calls}")
        print(f"Avg API calls per task: {total_calls/total_tasks:.1f}")
        
        # Estimate cost
        if args.model == 'gpt-4o':
            cost_per_call = 0.006
        elif args.model == 'gpt-4o-mini':
            cost_per_call = 0.0002
        else:
            cost_per_call = 0.0
        
        total_cost = total_calls * cost_per_call
        print(f"\n--- Cost ---")
        print(f"Estimated cost: ${total_cost:.4f}")
        print(f"Cost per task: ${total_cost/total_tasks:.4f}")
        print(f"Projected for 3000 tasks: ${total_cost * 3000 / total_tasks:.2f}")
    else:
        print("No tasks completed")
    
    print("=" * 80)
    
    # Save in same format as collect_trajectories_gpt_sampled.py
    from datetime import datetime
    output_data = {
        'metadata': {
            'model': args.model,
            'method': 'LATS',
            'collection_date': datetime.now().isoformat(),
            'num_tasks': total_tasks,
            'iterations': args.iterations,
            'simulations': args.simulations,
        },
        'summary': {
            'total_tasks': total_tasks,
            'expert_successes': len(expert_successes),
            'expert_failures': total_tasks - len(expert_successes),
            'expert_success_rate': (len(expert_successes) / total_tasks * 100) if total_tasks > 0 else 0,
        },
        'successful_trajectories': successful_trajectories,
        'all_trajectories': all_trajectories,
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    if expert_successes:
        print(f"\n✓ {len(expert_successes)} VALID EXPERT TRAJECTORIES COLLECTED")
        print(f"\nCompatible with collect_trajectories_gpt_sampled.py format!")


if __name__ == '__main__':
    main()