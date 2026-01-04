"""
Two-estimator leverage for actionability measurement.

Implements:
- Estimator A (L_local): Single-step expert leverage
- Estimator B (L_upper): Expert upper bound (full control)
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..data.snapshot import Snapshot, LeverageLabels, TeacherHint


class PolicyProtocol(Protocol):
    """Protocol for policy objects."""
    
    def get_action(self, observation: str, valid_actions: list[str]) -> str:
        """Get action for given observation."""
        ...
    
    def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
        """Get action probability distribution."""
        ...


class EnvironmentProtocol(Protocol):
    """Protocol for environment objects."""
    
    def set_state(self, state_bytes: bytes) -> dict:
        """Restore environment state."""
        ...
    
    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """Take action in environment."""
        ...
    
    def is_success(self, reward: float) -> bool:
        """Check if reward indicates success."""
        ...


@dataclass
class LeverageConfig:
    """Configuration for leverage estimation."""
    
    # Estimator A: Single-step expert (p_force)
    n_force_rollouts: int = 7
    
    # Estimator B: Expert upper bound (p_expert)
    n_expert_rollouts: int = 2
    
    # Rollout settings
    max_steps: int = 30
    success_threshold: float = 0.8
    
    # Parallelization
    n_parallel: int = 4
    
    @classmethod
    def from_yaml(cls, path: str) -> "LeverageConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("leverage", {}))


def run_rollouts(
    snapshot: Snapshot,
    env: EnvironmentProtocol,
    policy: PolicyProtocol,
    n: int,
    max_steps: int = 30,
) -> float:
    """
    Run multiple rollouts from snapshot state using given policy.
    
    Args:
        snapshot: Starting state snapshot
        env: Environment instance
        policy: Policy to use for actions
        n: Number of rollouts
        max_steps: Maximum steps per rollout
        
    Returns:
        Success rate (fraction of successful rollouts)
    """
    successes = 0
    
    for _ in range(n):
        # Restore state
        obs_dict = env.set_state(snapshot.env_state_bytes)
        observation = obs_dict["observation"]
        valid_actions = obs_dict["valid_actions"]
        
        done = False
        steps = 0
        final_reward = 0.0
        
        while not done and steps < max_steps:
            action = policy.get_action(observation, valid_actions)
            obs_dict, reward, done, info = env.step(action)
            
            observation = obs_dict["observation"]
            valid_actions = obs_dict.get("valid_actions", [])
            final_reward = reward
            steps += 1
        
        if env.is_success(final_reward):
            successes += 1
    
    return successes / n if n > 0 else 0.0


def run_forced_rollouts(
    snapshot: Snapshot,
    env: EnvironmentProtocol,
    forced_action: str,
    continuation_policy: PolicyProtocol,
    n: int,
    max_steps: int = 30,
) -> float:
    """
    Run rollouts with forced first action, then continuation policy.
    
    This implements Estimator A (single-step expert leverage):
    Force the expert's suggested action at step 0, then let student continue.
    
    Args:
        snapshot: Starting state snapshot
        env: Environment instance
        forced_action: Action to force at first step
        continuation_policy: Policy to use after forced action
        n: Number of rollouts
        max_steps: Maximum steps per rollout
        
    Returns:
        Success rate after forcing expert action
    """
    successes = 0
    
    for _ in range(n):
        # Restore state
        obs_dict = env.set_state(snapshot.env_state_bytes)
        
        # Force first action
        obs_dict, reward, done, info = env.step(forced_action)
        observation = obs_dict["observation"]
        valid_actions = obs_dict.get("valid_actions", [])
        final_reward = reward
        steps = 1
        
        # Continue with student policy
        while not done and steps < max_steps:
            action = continuation_policy.get_action(observation, valid_actions)
            obs_dict, reward, done, info = env.step(action)
            
            observation = obs_dict["observation"]
            valid_actions = obs_dict.get("valid_actions", [])
            final_reward = reward
            steps += 1
        
        if env.is_success(final_reward):
            successes += 1
    
    return successes / n if n > 0 else 0.0


def estimate_leverage(
    snapshot: Snapshot,
    env: EnvironmentProtocol,
    student_policy: PolicyProtocol,
    expert_policy: PolicyProtocol,
    teacher_hint: TeacherHint,
    config: LeverageConfig,
) -> LeverageLabels:
    """
    Compute two-estimator leverage with aligned naming.
    
    - Estimator A (L_local): Single-step expert leverage
      p_force = P(success | force expert action, then student)
      L_local = p_force - p_policy
      
    - Estimator B (L_upper): Expert upper bound
      p_expert = P(success | expert takes over completely)
      L_upper = p_expert - p_policy
    
    Total budget: ~9 episodes per snapshot.
    
    Args:
        snapshot: State snapshot to evaluate
        env: Environment instance
        student_policy: Student policy for baseline and continuation
        expert_policy: Expert policy for upper bound
        teacher_hint: Contains suggested expert action
        config: Leverage estimation configuration
        
    Returns:
        LeverageLabels with all computed values
    """
    # Baseline: student policy success rate
    p_policy = run_rollouts(
        snapshot, env, student_policy,
        n=config.n_force_rollouts,
        max_steps=config.max_steps,
    )
    
    # Estimator A: Single-step expert (force expert action, then student)
    p_force = run_forced_rollouts(
        snapshot, env,
        forced_action=teacher_hint.suggested_action,
        continuation_policy=student_policy,
        n=config.n_force_rollouts,
        max_steps=config.max_steps,
    )
    L_local = p_force - p_policy
    
    # Estimator B: Expert upper bound (expert takes over completely)
    p_expert = run_rollouts(
        snapshot, env, expert_policy,
        n=config.n_expert_rollouts,
        max_steps=config.max_steps,
    )
    L_upper = p_expert - p_policy
    
    return LeverageLabels(
        p_policy=p_policy,
        p_force=p_force,
        L_local=L_local,
        n_force_rollouts=config.n_force_rollouts,
        p_expert=p_expert,
        L_upper=L_upper,
        n_expert_rollouts=config.n_expert_rollouts,
    )


def estimate_leverage_batch(
    snapshots: list[Snapshot],
    env_factory: Callable[[], EnvironmentProtocol],
    student_policy: PolicyProtocol,
    expert_policy: PolicyProtocol,
    teacher_hints: dict[str, TeacherHint],
    config: LeverageConfig,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> dict[str, LeverageLabels]:
    """
    Estimate leverage for a batch of snapshots in parallel.
    
    Args:
        snapshots: List of snapshots to process
        env_factory: Factory function to create environment instances
        student_policy: Student policy
        expert_policy: Expert policy
        teacher_hints: Dict mapping snapshot IDs to teacher hints
        config: Leverage configuration
        progress_callback: Optional callback called with completed count
        
    Returns:
        Dict mapping snapshot IDs to LeverageLabels
    """
    results = {}
    
    def process_snapshot(snapshot: Snapshot) -> tuple[str, LeverageLabels]:
        env = env_factory()
        try:
            hint = teacher_hints.get(snapshot.id)
            if hint is None:
                raise ValueError(f"No teacher hint for snapshot {snapshot.id}")
            
            leverage = estimate_leverage(
                snapshot, env, student_policy, expert_policy, hint, config
            )
            return snapshot.id, leverage
        finally:
            if hasattr(env, "close"):
                env.close()
    
    with ThreadPoolExecutor(max_workers=config.n_parallel) as executor:
        futures = {
            executor.submit(process_snapshot, snap): snap.id
            for snap in snapshots
        }
        
        completed = 0
        for future in as_completed(futures):
            snapshot_id, leverage = future.result()
            results[snapshot_id] = leverage
            
            completed += 1
            if progress_callback:
                progress_callback(completed)
    
    return results


def get_leverage_profile(L_local: float, L_upper: float) -> str:
    """
    Classify snapshot by leverage profile for analysis.
    
    Profiles:
    - A_bottleneck: Single action is the bottleneck (high L_local, small gap)
    - B_trajectory: Needs trajectory-level help (low L_local, large gap)
    - C_deadend: Even expert can't recover (low L_upper)
    
    Args:
        L_local: Single-step leverage (Estimator A)
        L_upper: Expert upper bound (Estimator B)
        
    Returns:
        Profile classification string
    """
    L_gap = L_upper - L_local
    
    if L_upper < 0.3:
        return "C_deadend"
    elif L_local > 0.3 and L_gap < 0.2:
        return "A_bottleneck"
    elif L_local < 0.2 and L_gap > 0.2:
        return "B_trajectory"
    else:
        return "mixed"


def compute_leverage_statistics(
    leverage_labels: list[LeverageLabels],
) -> dict:
    """
    Compute summary statistics for leverage labels.
    
    Args:
        leverage_labels: List of LeverageLabels objects
        
    Returns:
        Dictionary with statistics
    """
    if not leverage_labels:
        return {}
    
    p_policy = [l.p_policy for l in leverage_labels]
    L_local = [l.L_local for l in leverage_labels]
    L_upper = [l.L_upper for l in leverage_labels]
    leverage_gap = [l.leverage_gap for l in leverage_labels]
    
    profiles = [get_leverage_profile(l.L_local, l.L_upper) for l in leverage_labels]
    profile_counts = {p: profiles.count(p) for p in set(profiles)}
    
    return {
        "p_policy": {
            "mean": float(np.mean(p_policy)),
            "std": float(np.std(p_policy)),
            "min": float(np.min(p_policy)),
            "max": float(np.max(p_policy)),
        },
        "L_local": {
            "mean": float(np.mean(L_local)),
            "std": float(np.std(L_local)),
            "min": float(np.min(L_local)),
            "max": float(np.max(L_local)),
        },
        "L_upper": {
            "mean": float(np.mean(L_upper)),
            "std": float(np.std(L_upper)),
            "min": float(np.min(L_upper)),
            "max": float(np.max(L_upper)),
        },
        "leverage_gap": {
            "mean": float(np.mean(leverage_gap)),
            "std": float(np.std(leverage_gap)),
        },
        "profile_distribution": profile_counts,
        "n_snapshots": len(leverage_labels),
    }


# Alias for script compatibility
compute_leverage = estimate_leverage

