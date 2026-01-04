"""
Student rollout for failure discovery.

This module handles rolling out a student model in the environment
to discover failure cases - the most valuable teachable moments.

Key capabilities:
- Run student model on tasks
- Detect and classify failures (stuck, wrong action, timeout, etc.)
- Collect failure trajectories with rich metadata
- Compare against teacher performance for gap analysis
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Callable, Tuple
from enum import Enum
import time

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of student failures."""
    
    STUCK_LOOP = "stuck_loop"           # Repeating same actions
    WRONG_ACTION = "wrong_action"       # Action that leads away from goal
    TIMEOUT = "timeout"                 # Exceeded max steps without success
    EARLY_TERMINATION = "early_term"    # Terminated prematurely
    SUBOPTIMAL = "suboptimal"          # Succeeded but inefficiently
    CONFUSION = "confusion"             # High uncertainty, random-looking actions
    NONE = "none"                       # No failure (success)


@dataclass
class FailureEvent:
    """A detected failure during student rollout."""
    
    trajectory_id: str
    step_idx: int
    failure_type: FailureType
    state: str
    action_taken: str
    valid_actions: list[str]
    
    # Context
    recent_actions: list[str] = field(default_factory=list)
    recent_states: list[str] = field(default_factory=list)
    
    # Metrics at failure point
    cumulative_reward: float = 0.0
    steps_remaining: int = 0
    
    # Model outputs (if available)
    action_probs: Optional[dict[str, float]] = None
    model_confidence: Optional[float] = None
    
    # Teacher comparison (if available)
    teacher_action: Optional[str] = None
    teacher_would_succeed: Optional[bool] = None
    
    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "step_idx": self.step_idx,
            "failure_type": self.failure_type.value,
            "state": self.state,
            "action_taken": self.action_taken,
            "valid_actions": self.valid_actions,
            "recent_actions": self.recent_actions,
            "recent_states": self.recent_states,
            "cumulative_reward": self.cumulative_reward,
            "steps_remaining": self.steps_remaining,
            "action_probs": self.action_probs,
            "model_confidence": self.model_confidence,
            "teacher_action": self.teacher_action,
            "teacher_would_succeed": self.teacher_would_succeed,
        }


@dataclass
class StudentRolloutConfig:
    """Configuration for student rollout."""
    
    # Model settings
    model_name: str = "student_base"
    model_path: Optional[str] = None
    
    # Rollout settings
    max_steps: int = 30
    n_tasks: int = 100
    task_ids: Optional[list[str]] = None  # Specific tasks, or None for random
    
    # Failure detection
    loop_detection_window: int = 6
    loop_threshold: int = 2              # Repeat pattern N times = stuck
    confidence_threshold: float = 0.3    # Below this = confusion
    suboptimal_factor: float = 1.5       # >1.5x optimal steps = suboptimal
    
    # Collection settings
    collect_all: bool = False            # If True, collect successes too
    save_model_outputs: bool = True      # Save action distributions
    
    # Environment
    env_type: str = "webshop"
    mock_env: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "StudentRolloutConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("student_rollout", {}))


@dataclass
class RolloutState:
    """State at a single step during rollout (for snapshot creation)."""
    
    step_idx: int
    observation: str
    valid_actions: list[str]
    action_taken: str
    reward: float
    done: bool
    
    # Model outputs
    action_probs: Optional[dict[str, float]] = None
    confidence: Optional[float] = None
    raw_output: Optional[str] = None
    
    # Restorable env state (base64-encoded bytes from env.get_state())
    env_state_b64: Optional[str] = None
    
    # Task context (useful for prompting + debugging)
    instruction_text: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "observation": self.observation,
            "valid_actions": self.valid_actions,
            "action_taken": self.action_taken,
            "reward": self.reward,
            "done": self.done,
            "action_probs": self.action_probs,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
            "env_state_b64": self.env_state_b64,
            "instruction_text": self.instruction_text,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RolloutState":
        return cls(
            step_idx=data["step_idx"],
            observation=data["observation"],
            valid_actions=data.get("valid_actions", []),
            action_taken=data["action_taken"],
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            action_probs=data.get("action_probs"),
            confidence=data.get("confidence"),
            raw_output=data.get("raw_output"),
            env_state_b64=data.get("env_state_b64"),
            instruction_text=data.get("instruction_text"),
        )



@dataclass
class RolloutResult:
    """Result of a single rollout."""
    
    trajectory_id: str
    task_id: str
    success: bool
    total_reward: float
    n_steps: int
    
    # Full trajectory
    states: list[str]
    actions: list[str]
    rewards: list[float]
    
    # Detected failures
    failures: list[FailureEvent] = field(default_factory=list)
    
    # NEW: Full rollout states with env snapshots
    rollout_states: list[RolloutState] = field(default_factory=list)
    
    # Task context
    instruction_text: str = ""
    
    # Timing
    duration_seconds: float = 0.0
    
    # Model metadata
    model_name: str = ""
    
    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "success": self.success,
            "total_reward": self.total_reward,
            "n_steps": self.n_steps,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "failures": [f.to_dict() for f in self.failures],
            "rollout_states": [s.to_dict() for s in self.rollout_states],
            "instruction_text": self.instruction_text,
            "duration_seconds": self.duration_seconds,
            "model_name": self.model_name,
        }



class StudentRollout:
    """
    Executes student model rollouts and detects failures.
    
    This is the primary mechanism for discovering teachable moments:
    run the student, see where it fails, and collect those cases
    for targeted training.
    """
    
    def __init__(
        self,
        config: StudentRolloutConfig,
        model: Optional[Any] = None,
        env: Optional[Any] = None,
        model_factory: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.env = env
        self.model_factory = model_factory
        self.tokenizer = None
        self._trajectory_counter = 0
        
    def _load_model(self):
        """Load student model if not provided.
        
        Uses ModelFactory for consistent action decoding and
        action-probability estimation across rollouts/labels.
        """
        if self.model is not None:
            return
        
        # Try to use ModelFactory for consistent loading
        if self.model_factory is None and self.config.model_path:
            try:
                from ..utils.model_factory import ModelFactory, ModelConfig
                
                logger.info(f"Loading model via ModelFactory from {self.config.model_path}")
                model_config = ModelConfig.from_checkpoint(self.config.model_path)
                self.model_factory = ModelFactory(model_config)
                self.model, self.tokenizer = self.model_factory.load()
                
                # Set pad_token if missing (common issue)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                return
            except ImportError:
                logger.warning("ModelFactory not available, falling back to direct loading")
            except Exception as e:
                logger.warning(f"ModelFactory loading failed: {e}, falling back to direct loading")
        
        # Fallback: direct transformers loading
        if self.config.model_path:
            logger.info(f"Loading model directly from {self.config.model_path}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError("Either model or model_path must be provided")
    
    def _load_env(self):
        """Load environment if not provided."""
        if self.env is not None:
            return
            
        from ..data.webshop_env import create_env, WebShopConfig
        env_config = WebShopConfig(max_steps=self.config.max_steps)
        self.env = create_env(env_config, mock=self.config.mock_env)
    
    def _get_action(
        self,
        state: str,
        valid_actions: list[str],
        instruction_text: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Get action from student model.
        
        Uses ModelFactory (masked scoring) to produce BOTH the chosen action
        and a meaningful action distribution for UQ metrics.
        
        Args:
            state: Current observation text
            valid_actions: List of valid action strings
            instruction_text: Optional task description for context
            
        Returns:
            action: Selected action string
            metadata: Dict with action_probs, confidence, raw_output, etc.
        """
        # Preferred path: use the shared model_factory
        if self.model_factory is not None:
            action, action_probs, raw_output = self.model_factory.decode_action(
                observation=state,
                valid_actions=valid_actions,
                task_description=instruction_text,
            )
            confidence = max(action_probs.values()) if action_probs else None
            return action, {
                'action_probs': action_probs,
                'confidence': confidence,
                'raw_output': raw_output,
            }
        
        # Fallback: direct model generation
        prompt = self._format_prompt(state, valid_actions)
        
        if hasattr(self.model, "generate") and self.tokenizer:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated = self.tokenizer.decode(
                outputs.sequences[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            action = self._extract_action(generated, valid_actions)
            
            confidence = self._compute_confidence(outputs) if outputs.scores else None
            return action, {
                "action_probs": None,
                "confidence": confidence,
                "raw_output": generated,
            }
        else:
            # Mock/simple models
            action = valid_actions[0] if valid_actions else "none"
            return action, {
                "action_probs": None,
                "confidence": 0.5,
                "raw_output": None,
            }

    
    def _format_prompt(self, state: str, valid_actions: list[str]) -> str:
        """Format state into model prompt."""
        actions_str = ", ".join(valid_actions[:10])  # Limit for context
        return f"State: {state}\nValid actions: {actions_str}\nAction:"
    
    def _extract_action(
        self,
        generated_text: str,
        valid_actions: list[str],
    ) -> str:
        """
        Extract action from model output with multiple fallback strategies.
        
        Handles:
        1. Exact matches
        2. Partial matches
        3. WebShop action format (search[x], click[x])
        4. JSON responses
        """
        import re
        import json as json_module
        
        generated_lower = generated_text.lower().strip()
        
        # Strategy 1: Exact match
        for action in valid_actions:
            if action.lower() == generated_lower:
                return action
        
        # Strategy 2: Try to parse as JSON
        try:
            # Handle {"action": "..."} format
            json_match = re.search(r'\{[^}]+\}', generated_text)
            if json_match:
                data = json_module.loads(json_match.group())
                if "action" in data:
                    action_text = data["action"].lower()
                    for action in valid_actions:
                        if action.lower() == action_text:
                            return action
        except (json_module.JSONDecodeError, KeyError):
            pass
        
        # Strategy 3: WebShop action format parsing
        # search[query]
        search_match = re.search(r'search\s*\[\s*([^\]]+)\s*\]', generated_text, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip().lower()
            # Find matching search action
            for action in valid_actions:
                if action.lower().startswith("search[") and query in action.lower():
                    return action
            # Create search action if valid
            search_action = f"search[{query}]"
            if any(a.lower().startswith("search") for a in valid_actions):
                return search_action
        
        # click[element]
        click_match = re.search(r'click\s*\[\s*([^\]]+)\s*\]', generated_text, re.IGNORECASE)
        if click_match:
            element = click_match.group(1).strip().lower()
            for action in valid_actions:
                if element in action.lower():
                    return action
        
        # Strategy 4: Substring matching
        for action in valid_actions:
            if action.lower() in generated_lower:
                return action
        
        # Strategy 5: Reverse substring (action contains generated)
        for action in valid_actions:
            if generated_lower in action.lower() and len(generated_lower) > 2:
                return action
        
        # Strategy 6: Word overlap scoring
        generated_words = set(generated_lower.split())
        best_action = None
        best_score = 0
        
        for action in valid_actions:
            action_words = set(action.lower().split())
            overlap = len(generated_words & action_words)
            if overlap > best_score:
                best_score = overlap
                best_action = action
        
        if best_action and best_score > 0:
            return best_action
        
        # Fallback: Log warning and return first action
        logger.warning(
            f"Could not extract action from: '{generated_text[:100]}'. "
            f"Using fallback: '{valid_actions[0]}'"
        )
        return valid_actions[0] if valid_actions else ""

    def _generate_with_constraints(
        self,
        prompt: str,
        valid_actions: list[str],
    ) -> str:
        """
        Generate action with soft constraints toward valid actions.
        
        Uses action scoring to select best valid action.
        """
        import torch
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Score each valid action
        action_scores = {}
        
        for action in valid_actions:
            # Compute likelihood of action given prompt
            full_text = f"{prompt}\n{action}"
            full_inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
            full_inputs = {k: v.to(self.model.device) for k, v in full_inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**full_inputs, labels=full_inputs["input_ids"])
                # Negative log likelihood -> lower is better
                action_scores[action] = -outputs.loss.item()
        
        # Return highest scoring action
        return max(action_scores, key=action_scores.get)
    
    def _compute_confidence(self, outputs) -> float:
        """Compute model confidence from generation outputs."""
        # Simplified: use mean probability of generated tokens
        if not outputs.scores:
            return 0.5
        
        import torch
        probs = [torch.softmax(s, dim=-1).max().item() for s in outputs.scores]
        return sum(probs) / len(probs) if probs else 0.5
    
    def _detect_loop(self, actions: list[str]) -> bool:
        """Detect if student is stuck in action loop."""
        if len(actions) < self.config.loop_detection_window:
            return False
        
        window = actions[-self.config.loop_detection_window:]
        
        # Check for repeated single action
        if len(set(window)) == 1:
            return True
        
        # Check for repeated pattern (length 2 or 3)
        for pattern_len in [2, 3]:
            if len(window) >= pattern_len * self.config.loop_threshold:
                pattern = tuple(window[:pattern_len])
                repeats = 0
                for i in range(0, len(window) - pattern_len + 1, pattern_len):
                    if tuple(window[i:i+pattern_len]) == pattern:
                        repeats += 1
                if repeats >= self.config.loop_threshold:
                    return True
        
        return False
    
    def _classify_failure(
        self,
        success: bool,
        n_steps: int,
        actions: list[str],
        final_reward: float,
        confidences: list[float],
        optimal_steps: Optional[int] = None,
    ) -> FailureType:
        """Classify the type of failure (or success)."""
        
        if success:
            # Check if suboptimal
            if optimal_steps and n_steps > optimal_steps * self.config.suboptimal_factor:
                return FailureType.SUBOPTIMAL
            return FailureType.NONE
        
        # Failed - classify why
        if self._detect_loop(actions):
            return FailureType.STUCK_LOOP
        
        if n_steps >= self.config.max_steps:
            return FailureType.TIMEOUT
        
        # Check for confusion (low confidence throughout)
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence < self.config.confidence_threshold:
                return FailureType.CONFUSION
        
        # Default to wrong action (made bad decisions)
        return FailureType.WRONG_ACTION
    
    def rollout_single(self, task_id: Optional[str] = None) -> RolloutResult:
        """
        Execute single rollout and detect failures.
        
        Args:
            task_id: Specific task, or None for random
            
        Returns:
            RolloutResult with trajectory and detected failures
        """
        self._load_model()
        self._load_env()
        
        # Generate trajectory ID
        self._trajectory_counter += 1
        traj_id = f"student_{self.config.model_name}_{self._trajectory_counter:06d}"
        
        # Reset environment
        obs = self.env.reset(task_id)
        actual_task_id = obs.get("task_id", task_id or "unknown")
        instruction_text = obs.get("instruction_text", "")
        
        # Collect trajectory
        states = [obs["observation"]]
        actions = []
        rewards = []
        confidences = []
        failures = []
        rollout_states = []
        
        start_time = time.time()
        done = False
        total_reward = 0.0
        step_idx = 0
        
        while not done:
            state = obs["observation"]
            valid_actions = obs.get("valid_actions", [])
            
            if not valid_actions:
                break
            
            # Capture environment state BEFORE taking action (for snapshot restoration)
            env_state_b64 = None
            if hasattr(self.env, 'get_state'):
                try:
                    env_state_bytes = self.env.get_state()
                    if env_state_bytes:
                        env_state_b64 = base64.b64encode(env_state_bytes).decode("ascii")
                except Exception as e:
                    logger.debug(f"Could not capture env state: {e}")
            
            # Get student action (with instruction context)
            action, metadata = self._get_action(state, valid_actions, instruction_text)
            
            confidence = metadata.get("confidence")
            if confidence is not None:
                confidences.append(confidence)
            
            # Check for loop before taking action
            if self._detect_loop(actions + [action]):
                # Record failure event
                failures.append(FailureEvent(
                    trajectory_id=traj_id,
                    step_idx=step_idx,
                    failure_type=FailureType.STUCK_LOOP,
                    state=state,
                    action_taken=action,
                    valid_actions=valid_actions,
                    recent_actions=actions[-5:] if actions else [],
                    recent_states=states[-3:] if states else [],
                    cumulative_reward=total_reward,
                    steps_remaining=self.config.max_steps - step_idx,
                    action_probs=metadata.get("action_probs"),
                    model_confidence=confidence,
                ))
            
            # Take action
            obs, reward, done, info = self.env.step(action)
            
            # Build RolloutState for this step (captures state BEFORE action was taken)
            rollout_state = RolloutState(
                step_idx=step_idx,
                observation=state,
                valid_actions=valid_actions,
                action_taken=action,
                reward=reward,
                done=done,
                action_probs=metadata.get("action_probs"),
                confidence=confidence,
                raw_output=metadata.get("raw_output"),
                env_state_b64=env_state_b64,
                instruction_text=instruction_text,
            )
            rollout_states.append(rollout_state)
            
            actions.append(action)
            rewards.append(reward)
            states.append(obs["observation"])
            total_reward += reward
            step_idx += 1
        
        duration = time.time() - start_time
        success = self.env.is_success(total_reward)
        
        # Classify overall failure type
        overall_failure = self._classify_failure(
            success=success,
            n_steps=len(actions),
            actions=actions,
            final_reward=total_reward,
            confidences=confidences,
        )
        
        # If failed and no specific failure detected, add one at end
        if overall_failure != FailureType.NONE and not failures:
            # Find the likely failure point (step with lowest confidence or reward)
            failure_step = len(actions) - 1
            if confidences:
                failure_step = confidences.index(min(confidences))
            
            failures.append(FailureEvent(
                trajectory_id=traj_id,
                step_idx=failure_step,
                failure_type=overall_failure,
                state=states[failure_step] if failure_step < len(states) else states[-1],
                action_taken=actions[failure_step] if failure_step < len(actions) else "",
                valid_actions=[],  # Not available retrospectively
                recent_actions=actions[max(0, failure_step-5):failure_step],
                recent_states=states[max(0, failure_step-3):failure_step],
                cumulative_reward=sum(rewards[:failure_step+1]),
                steps_remaining=self.config.max_steps - failure_step,
                model_confidence=confidences[failure_step] if failure_step < len(confidences) else None,
            ))
        
        return RolloutResult(
            trajectory_id=traj_id,
            task_id=actual_task_id,
            success=success,
            total_reward=total_reward,
            n_steps=len(actions),
            states=states,
            actions=actions,
            rewards=rewards,
            failures=failures,
            rollout_states=rollout_states,
            instruction_text=instruction_text,
            duration_seconds=duration,
            model_name=self.config.model_name,
        )
    
    def rollout_batch(
        self,
        task_ids: Optional[list[str]] = None,
        n_tasks: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[RolloutResult]:
        """
        Execute batch of rollouts.
        
        Args:
            task_ids: Specific tasks, or None to use config
            n_tasks: Number of tasks (if task_ids not specified)
            progress_callback: Called with (completed, total)
            
        Returns:
            List of RolloutResult
        """
        task_ids = task_ids or self.config.task_ids
        n = n_tasks or self.config.n_tasks
        
        results = []
        total = len(task_ids) if task_ids else n
        
        for i in range(total):
            task_id = task_ids[i] if task_ids else None
            
            try:
                result = self.rollout_single(task_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Rollout {i} failed: {e}")
                continue
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def collect_failures(
        self,
        results: Optional[list[RolloutResult]] = None,
        n_tasks: Optional[int] = None,
    ) -> list[FailureEvent]:
        """
        Collect all failure events from rollouts.
        
        Args:
            results: Pre-computed results, or None to run new rollouts
            n_tasks: Number of tasks if running new rollouts
            
        Returns:
            List of FailureEvent
        """
        if results is None:
            results = self.rollout_batch(n_tasks=n_tasks)
        
        failures = []
        for result in results:
            failures.extend(result.failures)
        
        return failures
    
    def get_statistics(self, results: list[RolloutResult]) -> dict:
        """Compute statistics over rollout results."""
        if not results:
            return {}
        
        successes = sum(1 for r in results if r.success)
        total_failures = sum(len(r.failures) for r in results)
        
        # Failure type breakdown
        failure_types = {}
        for r in results:
            for f in r.failures:
                ft = f.failure_type.value
                failure_types[ft] = failure_types.get(ft, 0) + 1
        
        return {
            "n_rollouts": len(results),
            "n_successes": successes,
            "n_failures": len(results) - successes,
            "success_rate": successes / len(results),
            "total_failure_events": total_failures,
            "failure_type_breakdown": failure_types,
            "avg_steps": sum(r.n_steps for r in results) / len(results),
            "avg_reward": sum(r.total_reward for r in results) / len(results),
            "avg_duration": sum(r.duration_seconds for r in results) / len(results),
        }
    
    def save_results(
        self,
        results: list[RolloutResult],
        output_path: str,
        include_full_trajectories: bool = True,
    ):
        """Save rollout results to JSON."""
        data = {
            "config": {
                "model_name": self.config.model_name,
                "n_tasks": len(results),
                "max_steps": self.config.max_steps,
            },
            "statistics": self.get_statistics(results),
            "results": [r.to_dict() for r in results] if include_full_trajectories else None,
            "failures": [
                f.to_dict()
                for r in results
                for f in r.failures
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
