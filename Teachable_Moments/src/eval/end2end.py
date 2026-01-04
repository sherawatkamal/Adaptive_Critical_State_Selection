"""End-to-end evaluation metrics for trained agents.

Computes overall success rates, reward distributions, and task completion
metrics across evaluation episodes.
"""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional
import json
import numpy as np
from pathlib import Path


class PolicyProtocol(Protocol):
    """Protocol for policy inference."""
    
    def get_action(self, observation: str) -> str:
        """Return action given observation."""
        ...
    
    def get_action_probs(self, observation: str) -> Dict[str, float]:
        """Return action probabilities given observation."""
        ...


class EnvironmentProtocol(Protocol):
    """Protocol for environment interaction."""
    
    def reset(self, task_id: Optional[str] = None) -> str:
        """Reset environment and return initial observation."""
        ...
    
    def step(self, action: str) -> tuple:
        """Take action, return (obs, reward, done, info)."""
        ...


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""
    task_id: str
    success: bool
    total_reward: float
    num_steps: int
    final_observation: str
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results across episodes."""
    success_rate: float
    mean_reward: float
    std_reward: float
    median_reward: float
    mean_steps: int
    completion_rate: float  # Episodes that terminated normally
    num_episodes: int
    episode_results: List[EpisodeResult] = field(default_factory=list)
    
    # Per-task breakdown
    per_task_success: Dict[str, bool] = field(default_factory=dict)
    
    # Confidence intervals (bootstrap)
    success_rate_ci: tuple = (0.0, 0.0)
    reward_ci: tuple = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "median_reward": self.median_reward,
            "mean_steps": self.mean_steps,
            "completion_rate": self.completion_rate,
            "num_episodes": self.num_episodes,
            "success_rate_ci": list(self.success_rate_ci),
            "reward_ci": list(self.reward_ci),
            "per_task_success": self.per_task_success,
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EvaluationResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            success_rate=data["success_rate"],
            mean_reward=data["mean_reward"],
            std_reward=data["std_reward"],
            median_reward=data["median_reward"],
            mean_steps=data["mean_steps"],
            completion_rate=data["completion_rate"],
            num_episodes=data["num_episodes"],
            success_rate_ci=tuple(data.get("success_rate_ci", [0.0, 0.0])),
            reward_ci=tuple(data.get("reward_ci", [0.0, 0.0])),
            per_task_success=data.get("per_task_success", {}),
        )


def run_episode(
    policy: PolicyProtocol,
    env: EnvironmentProtocol,
    task_id: Optional[str] = None,
    max_steps: int = 50,
    record_trajectory: bool = False,
) -> EpisodeResult:
    """Run a single evaluation episode.
    
    Args:
        policy: Policy to evaluate
        env: Environment to run in
        task_id: Optional task identifier
        max_steps: Maximum steps before truncation
        record_trajectory: Whether to record full trajectory
        
    Returns:
        EpisodeResult with episode outcomes
    """
    obs = env.reset(task_id)
    trajectory = []
    total_reward = 0.0
    done = False
    step = 0
    
    while not done and step < max_steps:
        action = policy.get_action(obs)
        
        if record_trajectory:
            trajectory.append({
                "step": step,
                "observation": obs,
                "action": action,
            })
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        if record_trajectory:
            trajectory[-1]["reward"] = reward
            trajectory[-1]["done"] = done
    
    # Determine success (reward >= 1.0 in WebShop means success)
    success = total_reward >= 1.0
    
    return EpisodeResult(
        task_id=task_id or "unknown",
        success=success,
        total_reward=total_reward,
        num_steps=step,
        final_observation=obs,
        trajectory=trajectory,
        metadata={"truncated": step >= max_steps and not done},
    )


def compute_success_rate(results: List[EpisodeResult]) -> float:
    """Compute success rate from episode results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def compute_reward_metrics(results: List[EpisodeResult]) -> Dict[str, float]:
    """Compute reward statistics from episode results."""
    if not results:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    
    rewards = [r.total_reward for r in results]
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "median": float(np.median(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }


def bootstrap_confidence_interval(
    values: List[float],
    stat_fn: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple:
    """Compute bootstrap confidence interval.
    
    Args:
        values: Sample values
        stat_fn: Statistic to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 95%)
        
    Returns:
        (lower, upper) confidence interval bounds
    """
    if len(values) < 2:
        return (0.0, 0.0)
    
    rng = np.random.default_rng(42)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_stats.append(stat_fn(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


def evaluate_end2end(
    policy: PolicyProtocol,
    env: EnvironmentProtocol,
    task_ids: List[str],
    max_steps: int = 50,
    record_trajectories: bool = False,
    compute_ci: bool = True,
) -> EvaluationResult:
    """Run end-to-end evaluation across multiple tasks.
    
    Args:
        policy: Policy to evaluate
        env: Environment to run in
        task_ids: List of task identifiers to evaluate on
        max_steps: Maximum steps per episode
        record_trajectories: Whether to record full trajectories
        compute_ci: Whether to compute confidence intervals
        
    Returns:
        EvaluationResult with aggregated metrics
    """
    episode_results = []
    
    for task_id in task_ids:
        result = run_episode(
            policy=policy,
            env=env,
            task_id=task_id,
            max_steps=max_steps,
            record_trajectory=record_trajectories,
        )
        episode_results.append(result)
    
    # Compute aggregated metrics
    success_rate = compute_success_rate(episode_results)
    reward_metrics = compute_reward_metrics(episode_results)
    
    mean_steps = int(np.mean([r.num_steps for r in episode_results]))
    completion_rate = sum(1 for r in episode_results if not r.metadata.get("truncated", False)) / len(episode_results)
    
    per_task_success = {r.task_id: r.success for r in episode_results}
    
    # Compute confidence intervals
    if compute_ci and len(episode_results) >= 10:
        successes = [1.0 if r.success else 0.0 for r in episode_results]
        rewards = [r.total_reward for r in episode_results]
        success_rate_ci = bootstrap_confidence_interval(successes)
        reward_ci = bootstrap_confidence_interval(rewards)
    else:
        success_rate_ci = (0.0, 0.0)
        reward_ci = (0.0, 0.0)
    
    return EvaluationResult(
        success_rate=success_rate,
        mean_reward=reward_metrics["mean"],
        std_reward=reward_metrics["std"],
        median_reward=reward_metrics["median"],
        mean_steps=mean_steps,
        completion_rate=completion_rate,
        num_episodes=len(episode_results),
        episode_results=episode_results if record_trajectories else [],
        per_task_success=per_task_success,
        success_rate_ci=success_rate_ci,
        reward_ci=reward_ci,
    )


def compare_models(
    results: Dict[str, EvaluationResult],
) -> Dict[str, Dict[str, float]]:
    """Compare evaluation results across multiple models.
    
    Args:
        results: Dictionary mapping model name to EvaluationResult
        
    Returns:
        Comparison statistics including deltas and significance
    """
    if "baseline" not in results:
        # Use first model as baseline if not specified
        baseline_name = list(results.keys())[0]
    else:
        baseline_name = "baseline"
    
    baseline = results[baseline_name]
    comparison = {}
    
    for name, result in results.items():
        delta_success = result.success_rate - baseline.success_rate
        delta_reward = result.mean_reward - baseline.mean_reward
        
        # Check if improvement is significant (non-overlapping CIs)
        sig_success = (
            result.success_rate_ci[0] > baseline.success_rate_ci[1] or
            result.success_rate_ci[1] < baseline.success_rate_ci[0]
        )
        sig_reward = (
            result.reward_ci[0] > baseline.reward_ci[1] or
            result.reward_ci[1] < baseline.reward_ci[0]
        )
        
        comparison[name] = {
            "success_rate": result.success_rate,
            "delta_success": delta_success,
            "significant_success": sig_success,
            "mean_reward": result.mean_reward,
            "delta_reward": delta_reward,
            "significant_reward": sig_reward,
        }
    
    return comparison
