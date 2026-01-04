"""
Teacher rollout for oracle trajectory collection.

This module handles rolling out a teacher model (e.g., GPT-4o) to:
1. Collect expert demonstrations
2. Compare against student failures
3. Identify teachable gaps (where student fails but teacher succeeds)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Callable
import time

logger = logging.getLogger(__name__)


@dataclass
class TeacherRolloutConfig:
    """Configuration for teacher rollout."""
    
    # Model settings
    model_name: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    
    # Rollout settings
    max_steps: int = 30
    n_tasks: int = 100
    task_ids: Optional[list[str]] = None
    
    # Generation settings
    temperature: float = 0.0           # Deterministic for demonstrations
    max_tokens: int = 100
    
    # Prompting
    system_prompt: str = """You are an expert web shopping agent. Given a task and current page state, 
select the best action to complete the shopping task efficiently.

Output only the action in the format: action[parameter]
Examples: search[laptop], click[Add to Cart], click[Buy Now]"""
    
    # Environment
    env_type: str = "webshop"
    mock_env: bool = False
    
    # Comparison mode
    compare_to_student: bool = False    # If True, run on same tasks as student
    student_results_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "TeacherRolloutConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("teacher_rollout", {}))


@dataclass
class TeacherRolloutResult:
    """Result of a single teacher rollout."""
    
    trajectory_id: str
    task_id: str
    success: bool
    total_reward: float
    n_steps: int
    
    # Full trajectory
    states: list[str]
    actions: list[str]
    rewards: list[float]
    
    # Teacher reasoning (if available)
    reasoning: list[str] = field(default_factory=list)
    
    # Timing
    duration_seconds: float = 0.0
    api_calls: int = 0
    
    # Comparison metadata
    student_succeeded: Optional[bool] = None
    student_steps: Optional[int] = None
    
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
            "reasoning": self.reasoning,
            "duration_seconds": self.duration_seconds,
            "api_calls": self.api_calls,
            "student_succeeded": self.student_succeeded,
            "student_steps": self.student_steps,
        }


class TeacherRollout:
    """
    Executes teacher model rollouts for expert demonstrations.
    
    Primary use cases:
    1. Collect oracle trajectories for demo supervision
    2. Compare with student to find teachable gaps
    3. Generate optimal action sequences for evaluation
    """
    
    def __init__(
        self,
        config: TeacherRolloutConfig,
        client: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        self.config = config
        self.client = client
        self.env = env
        self._trajectory_counter = 0
        
    def _load_client(self):
        """Load teacher API client if not provided."""
        if self.client is not None:
            return
            
        import os
        from src.teacher.client import TeacherClient, TeacherConfig
        
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in {self.config.api_key_env}")
        
        teacher_config = TeacherConfig(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.client = TeacherClient(teacher_config)
    
    def _load_env(self):
        """Load environment if not provided."""
        if self.env is not None:
            return
            
        from src.data.webshop_env import create_env, WebShopConfig
        env_config = WebShopConfig(max_steps=self.config.max_steps)
        self.env = create_env(env_config, mock=self.config.mock_env)
    
    def _get_action(self, state: str, valid_actions: list[str], task: str) -> tuple[str, str]:
        """
        Get action from teacher model.
        
        Returns:
            action: Selected action string
            reasoning: Teacher's reasoning (if available)
        """
        prompt = self._format_prompt(state, valid_actions, task)
        
        response = self.client.complete(
            prompt=prompt,
            system=self.config.system_prompt,
        )
        
        # Parse action from response
        action = self._parse_action(response, valid_actions)
        reasoning = response  # Full response as reasoning
        
        return action, reasoning
    
    def _format_prompt(self, state: str, valid_actions: list[str], task: str) -> str:
        """Format state into model prompt."""
        actions_str = "\n".join(f"- {a}" for a in valid_actions[:20])
        return f"""Task: {task}

Current page:
{state}

Available actions:
{actions_str}

Select the best action to progress toward completing the task."""
    
    def _parse_action(self, response: str, valid_actions: list[str]) -> str:
        """Parse teacher response to valid action."""
        import re

        if not valid_actions:
            return ""

        response = response or ""
        response_lower = response.lower()

        # Exact (case-insensitive) match of a full action string.
        for action in valid_actions:
            if action.lower() in response_lower:
                return action

        # Try to match verb + parameter even if brackets are omitted (e.g. "Click on Buy Now").
        for action in valid_actions:
            m = re.match(r"^(?P<verb>\w+)\[(?P<param>.*)\]$", action)
            if not m:
                continue
            verb = (m.group("verb") or "").lower()
            param = (m.group("param") or "").strip().lower()
            if verb and verb in response_lower and param and param in response_lower:
                return action

        # Regex extract an action-like span and match it to valid actions.
        m = re.search(r"(?P<act>(click|search)\[[^\]]+\])", response, flags=re.IGNORECASE)
        if m:
            cand = m.group("act")
            cand_lower = cand.lower()
            for action in valid_actions:
                if action.lower() == cand_lower:
                    return action

        return valid_actions[0]
    
    def rollout_single(
        self,
        task_id: Optional[str] = None,
        student_result: Optional[dict] = None,
    ) -> TeacherRolloutResult:
        """
        Execute single teacher rollout.
        
        Args:
            task_id: Specific task, or None for random
            student_result: Student result for same task (for comparison)
            
        Returns:
            TeacherRolloutResult with trajectory
        """
        self._load_client()
        self._load_env()
        
        # Generate trajectory ID
        self._trajectory_counter += 1
        traj_id = f"teacher_{self.config.model_name}_{self._trajectory_counter:06d}"
        
        # Reset environment
        obs = self.env.reset(task_id)
        actual_task_id = obs.get("task_id", task_id or "unknown")
        task_description = self._get_task_description(obs)
        
        # Collect trajectory
        states = [obs["observation"]]
        actions = []
        rewards = []
        reasoning = []
        
        start_time = time.time()
        done = False
        total_reward = 0.0
        api_calls = 0
        
        while not done:
            state = obs["observation"]
            valid_actions = obs.get("valid_actions", [])
            
            if not valid_actions:
                break
            
            # Get teacher action
            action, reason = self._get_action(state, valid_actions, task_description)
            api_calls += 1
            
            # Take action
            obs, reward, done, info = self.env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            states.append(obs["observation"])
            reasoning.append(reason)
            total_reward += reward
        
        duration = time.time() - start_time
        success = self.env.is_success(total_reward)
        
        return TeacherRolloutResult(
            trajectory_id=traj_id,
            task_id=actual_task_id,
            success=success,
            total_reward=total_reward,
            n_steps=len(actions),
            states=states,
            actions=actions,
            rewards=rewards,
            reasoning=reasoning,
            duration_seconds=duration,
            api_calls=api_calls,
            student_succeeded=student_result.get("success") if student_result else None,
            student_steps=student_result.get("n_steps") if student_result else None,
        )
    
    def _get_task_description(self, obs: dict) -> str:
        """Extract task description from observation."""
        # Preferred: direct instruction text if available (from new env wrapper)
        if obs.get("instruction_text"):
            return obs["instruction_text"]

        # Fallback: WebShop specific - task is usually in first observation
        observation = obs.get("observation", "")
        
        # Look for instruction pattern
        import re
        match = re.search(r'Instruction:\s*\[([^\]]+)\]', observation)
        if match:
            return match.group(1)
        
        # Final fallback
        return observation[:200]
    
    def rollout_batch(
        self,
        task_ids: Optional[list[str]] = None,
        n_tasks: Optional[int] = None,
        student_results: Optional[list[dict]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[TeacherRolloutResult]:
        """
        Execute batch of teacher rollouts.
        
        Args:
            task_ids: Specific tasks, or None to use config
            n_tasks: Number of tasks (if task_ids not specified)
            student_results: Student results for comparison (by task_id)
            progress_callback: Called with (completed, total)
            
        Returns:
            List of TeacherRolloutResult
        """
        task_ids = task_ids or self.config.task_ids
        n = n_tasks or self.config.n_tasks
        
        # Index student results by task_id
        student_by_task = {}
        if student_results:
            for r in student_results:
                student_by_task[r.get("task_id")] = r
        
        results = []
        total = len(task_ids) if task_ids else n
        
        for i in range(total):
            task_id = task_ids[i] if task_ids else None
            student_result = student_by_task.get(task_id) if task_id else None
            
            try:
                result = self.rollout_single(task_id, student_result)
                results.append(result)
            except Exception as e:
                logger.error(f"Teacher rollout {i} failed: {e}")
                continue
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def compare_with_student(
        self,
        teacher_results: list[TeacherRolloutResult],
        student_results: list[dict],
    ) -> dict:
        """
        Compare teacher and student performance.
        
        Returns statistics on:
        - Where teacher succeeds and student fails (teachable gaps)
        - Where both fail (hard tasks)
        - Efficiency differences
        """
        # Index by task_id
        teacher_by_task = {r.task_id: r for r in teacher_results}
        student_by_task = {r.get("task_id"): r for r in student_results}
        
        common_tasks = set(teacher_by_task.keys()) & set(student_by_task.keys())
        
        teachable_gaps = []  # Teacher succeeds, student fails
        both_succeed = []
        both_fail = []
        student_better = []  # Student succeeds, teacher fails (rare)
        
        for task_id in common_tasks:
            t = teacher_by_task[task_id]
            s = student_by_task[task_id]
            
            t_success = t.success
            s_success = s.get("success", False)
            
            comparison = {
                "task_id": task_id,
                "teacher_success": t_success,
                "student_success": s_success,
                "teacher_steps": t.n_steps,
                "student_steps": s.get("n_steps", 0),
                "teacher_reward": t.total_reward,
                "student_reward": s.get("total_reward", 0),
            }
            
            if t_success and not s_success:
                teachable_gaps.append(comparison)
            elif t_success and s_success:
                both_succeed.append(comparison)
            elif not t_success and not s_success:
                both_fail.append(comparison)
            else:
                student_better.append(comparison)
        
        return {
            "n_common_tasks": len(common_tasks),
            "teachable_gaps": len(teachable_gaps),
            "both_succeed": len(both_succeed),
            "both_fail": len(both_fail),
            "student_better": len(student_better),
            "teachable_gap_rate": len(teachable_gaps) / len(common_tasks) if common_tasks else 0,
            "teacher_success_rate": (len(teachable_gaps) + len(both_succeed)) / len(common_tasks) if common_tasks else 0,
            "student_success_rate": (len(both_succeed) + len(student_better)) / len(common_tasks) if common_tasks else 0,
            "gap_details": teachable_gaps,
            "hard_tasks": both_fail,
        }
    
    def get_statistics(self, results: list[TeacherRolloutResult]) -> dict:
        """Compute statistics over teacher rollout results."""
        if not results:
            return {}
        
        successes = sum(1 for r in results if r.success)
        
        return {
            "n_rollouts": len(results),
            "n_successes": successes,
            "success_rate": successes / len(results),
            "avg_steps": sum(r.n_steps for r in results) / len(results),
            "avg_reward": sum(r.total_reward for r in results) / len(results),
            "avg_duration": sum(r.duration_seconds for r in results) / len(results),
            "total_api_calls": sum(r.api_calls for r in results),
        }
    
    def save_results(
        self,
        results: list[TeacherRolloutResult],
        output_path: str,
        include_reasoning: bool = True,
    ):
        """Save teacher rollout results to JSON."""
        data = {
            "config": {
                "model_name": self.config.model_name,
                "n_tasks": len(results),
                "max_steps": self.config.max_steps,
            },
            "statistics": self.get_statistics(results),
            "results": [r.to_dict() for r in results],
        }
        
        if not include_reasoning:
            for r in data["results"]:
                r["reasoning"] = []
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(results)} teacher results to {output_path}")
