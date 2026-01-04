"""
Contextual Patch Test (CPT) for learning payoff estimation.

CPT tests whether in-context teaching signals help the agent recover from failures.
This provides a proxy for learning payoff (ELP) without running full fine-tuning.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Callable, Tuple, List, Any
import re

from ..data.snapshot import Snapshot, TeacherHint, CPTLabels


# Patch templates as specified in research blueprint §3.3.1

PLACEBO_TEMPLATE = """[System note: Review the current situation carefully before selecting your next action.]"""

DEMO_TEMPLATE = """[Example from a similar situation]
In a situation like this, where the observation shows: "{brief_observation}"
The correct action was: {teacher_action}
Reason: {rationale}
[End of example]

Now continue with your task:"""

CONTRAST_TEMPLATE = """[Feedback on action choice]
In a situation like this, where the observation shows: "{brief_observation}"
- Avoid: {bad_action} - This fails because: {why_bad}
- Instead: {teacher_action} - This works because: {why_good}
[End of feedback]

Now continue with your task:"""

HINT_TEMPLATE = """[Observation from a similar situation]
When facing: "{brief_observation}"
A key insight was: {diagnosis}
[End of observation]

Now continue with your task:"""


class PolicyProtocol(Protocol):
    """Protocol for policy objects."""
    
    def get_action(self, observation: str, valid_actions: list[str]) -> str:
        """Get action for given observation."""
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
class CPTConfig:
    """Configuration for Contextual Patch Test."""
    
    patch_types: list[str] = None  # Default: ["base", "placebo", "demo", "contrast", "hint"]
    n_per_condition: int = 2       # Single-stage allocation (v8)
    max_steps: int = 30
    
    # Observation summarization
    max_observation_tokens: int = 50
    
    def __post_init__(self):
        if self.patch_types is None:
            self.patch_types = ["base", "placebo", "demo", "contrast", "hint"]
    
    @classmethod
    def from_yaml(cls, path: str) -> "CPTConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("cpt", {}))


def summarize_observation(observation: str, max_tokens: int = 50) -> str:
    """
    Summarize observation to fit in patch templates.
    
    Args:
        observation: Full observation string
        max_tokens: Maximum approximate token count
        
    Returns:
        Summarized observation
    """
    # Simple word-based truncation (approximating tokens)
    words = observation.split()
    
    if len(words) <= max_tokens:
        return observation
    
    # Keep first and last parts for context
    n_keep = max_tokens // 2
    return " ".join(words[:n_keep]) + " ... " + " ".join(words[-n_keep:])


def infer_why_bad(snapshot: Snapshot, teacher_hint: TeacherHint) -> str:
    """
    Infer why the agent's action was bad based on teacher hint.
    
    Args:
        snapshot: State snapshot with agent's action
        teacher_hint: Teacher's analysis
        
    Returns:
        Explanation of why the action was suboptimal
    """
    error_type = teacher_hint.error_type.value
    
    explanations = {
        "affordance_miss": "The action doesn't address the available option.",
        "attribute_confusion": "The action targets the wrong attribute or item.",
        "planning_error": "The action is out of sequence for the goal.",
        "exploration_failure": "More exploration was needed before this action.",
    }
    
    return explanations.get(error_type, "This action doesn't advance the task effectively.")


def generate_diagnosis(teacher_hint: TeacherHint) -> str:
    """
    Generate diagnostic insight from teacher hint for HINT patch.
    
    Args:
        teacher_hint: Teacher's analysis
        
    Returns:
        Diagnostic insight string
    """
    # Extract key insight from rationale
    rationale = teacher_hint.rationale
    
    # Try to extract the core insight
    if "because" in rationale.lower():
        parts = re.split(r"\bbecause\b", rationale, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[1].strip()
    
    return rationale


def generate_patch_text(
    patch_type: str,
    snapshot: Snapshot,
    teacher_hint: TeacherHint,
    max_observation_tokens: int = 50,
) -> str:
    """
    Generate patch text for CPT based on patch type.
    
    Args:
        patch_type: One of "placebo", "demo", "contrast", "hint"
        snapshot: State snapshot
        teacher_hint: Teacher's hint with suggested action
        max_observation_tokens: Max tokens for observation summary
        
    Returns:
        Formatted patch text
    """
    brief_obs = summarize_observation(snapshot.observation, max_observation_tokens)
    
    if patch_type == "placebo":
        return PLACEBO_TEMPLATE
    
    elif patch_type == "demo":
        return DEMO_TEMPLATE.format(
            brief_observation=brief_obs,
            teacher_action=teacher_hint.suggested_action,
            rationale=teacher_hint.rationale,
        )
    
    elif patch_type == "contrast":
        return CONTRAST_TEMPLATE.format(
            brief_observation=brief_obs,
            bad_action=snapshot.last_action or "previous action",
            why_bad=infer_why_bad(snapshot, teacher_hint),
            teacher_action=teacher_hint.suggested_action,
            why_good=teacher_hint.rationale,
        )
    
    elif patch_type == "hint":
        return HINT_TEMPLATE.format(
            brief_observation=brief_obs,
            diagnosis=generate_diagnosis(teacher_hint),
        )
    
    else:
        raise ValueError(f"Unknown patch type: {patch_type}")


def _apply_patch(patch_text: Optional[str], observation: str) -> str:
    """Apply patch text to observation.
    
    Args:
        patch_text: Patch to prepend (can be None or empty)
        observation: Current observation
        
    Returns:
        Patched observation
    """
    if not patch_text:
        return observation
    return f"{patch_text}\n\n{observation}"


def _rollout_from_state(
    env: EnvironmentProtocol,
    policy: PolicyProtocol,
    start_state_bytes: bytes,
    patch_text: Optional[str],
    max_steps: int = 50,
) -> Tuple[bool, float, int]:
    """Run a policy from a restorable state.
    
    Args:
        env: Environment instance
        policy: Policy for action selection
        start_state_bytes: Serialized state to start from
        patch_text: Optional patch to apply at EVERY step
        max_steps: Maximum steps per rollout
        
    Returns:
        Tuple of (success, total_reward, steps)
    """
    obs_dict = env.set_state(start_state_bytes)
    observation = obs_dict.get("observation", "")
    valid_actions = obs_dict.get("valid_actions", [])
    
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        steps += 1
        # Apply patch at EVERY step (not just step 0)
        patched_obs = _apply_patch(patch_text, observation)
        
        action = policy.get_action(patched_obs, valid_actions)
        obs_dict, reward, done, _info = env.step(action)
        
        observation = obs_dict.get("observation", "")
        valid_actions = obs_dict.get("valid_actions", [])
        total_reward += float(reward)
    
    return env.is_success(total_reward), total_reward, steps


def run_patched_rollouts(
    snapshot: Snapshot,
    env: EnvironmentProtocol,
    policy: PolicyProtocol,
    patch_text: str,
    n: int,
    max_steps: int = 30,
) -> float:
    """
    Run rollouts with patched agent at EVERY decision step.
    
    Args:
        snapshot: Starting state snapshot
        env: Environment instance
        policy: Policy to use (should accept prefixed observations)
        patch_text: Text to prepend to agent context at each step
        n: Number of rollouts
        max_steps: Maximum steps per rollout
        
    Returns:
        Success rate
    """
    if snapshot.env_state_bytes is None:
        raise ValueError(
            f"Snapshot {snapshot.id} has no env_state_bytes. "
            "Ensure snapshots are saved with state restoration enabled."
        )
    
    successes = 0
    
    for _ in range(n):
        success, _, _ = _rollout_from_state(
            env=env,
            policy=policy,
            start_state_bytes=snapshot.env_state_bytes,
            patch_text=patch_text,
            max_steps=max_steps,
        )
        if success:
            successes += 1
    
    return successes / n if n > 0 else 0.0


def run_cpt(
    snapshot: Snapshot,
    env: EnvironmentProtocol,
    student_policy: PolicyProtocol,
    teacher_hint: TeacherHint,
    config: Optional[CPTConfig] = None,
    retention_task_ids: Optional[List[int]] = None,
    retention_env_factory: Optional[Callable[[], EnvironmentProtocol]] = None,
) -> CPTLabels:
    """
    Run single-stage CPT with all patch types.
    
    This implements the Contextual Patch Test as specified in research blueprint §3.3.
    Total: 5 conditions × n_per_condition rollouts = 10 episodes (default).
    
    Args:
        snapshot: State snapshot to test
        env: Environment instance
        student_policy: Student policy for rollouts
        teacher_hint: Teacher's hint for patch generation
        config: CPT configuration
        retention_task_ids: Optional list of task IDs for retention evaluation
        retention_env_factory: Factory for retention environments (required if retention_task_ids provided)
        
    Returns:
        CPTLabels with all results
    """
    if config is None:
        config = CPTConfig()
    
    results = {}
    
    for patch_type in config.patch_types:
        if patch_type == "base":
            # No patch, just run policy
            patch_text = ""
        else:
            patch_text = generate_patch_text(
                patch_type, snapshot, teacher_hint,
                max_observation_tokens=config.max_observation_tokens,
            )
        
        results[patch_type] = run_patched_rollouts(
            snapshot, env, student_policy, patch_text,
            n=config.n_per_condition,
            max_steps=config.max_steps,
        )
    
    # Compute gains
    p_base = results.get("base", 0.0)
    p_placebo = results.get("placebo", 0.0)
    
    patch_gain_raw = {}
    patch_gain_net = {}
    
    for patch_type in ["demo", "contrast", "hint"]:
        if patch_type in results:
            patch_gain_raw[patch_type] = results[patch_type] - p_base
            patch_gain_net[patch_type] = results[patch_type] - p_placebo
    
    # Retention evaluation (optional)
    p_base_retention = None
    p_placebo_retention = None
    patch_retention = {}
    
    if retention_task_ids and retention_env_factory:
        # Evaluate on new tasks to check if intervention effect generalizes
        retention_env = retention_env_factory()
        try:
            for patch_type in ["base", "placebo", "demo", "contrast", "hint"]:
                if patch_type not in results:
                    continue
                
                if patch_type == "base":
                    patch_text = ""
                else:
                    patch_text = generate_patch_text(
                        patch_type, snapshot, teacher_hint,
                        max_observation_tokens=config.max_observation_tokens,
                    )
                
                # Run on retention tasks
                retention_successes = 0
                for task_id in retention_task_ids:
                    obs_dict = retention_env.reset(task_id)
                    observation = obs_dict.get("observation", "")
                    valid_actions = obs_dict.get("valid_actions", [])
                    
                    done = False
                    total_reward = 0.0
                    for _ in range(config.max_steps):
                        patched_obs = _apply_patch(patch_text, observation)
                        action = student_policy.get_action(patched_obs, valid_actions)
                        obs_dict, reward, done, _ = retention_env.step(action)
                        observation = obs_dict.get("observation", "")
                        valid_actions = obs_dict.get("valid_actions", [])
                        total_reward += float(reward)
                        if done:
                            break
                    
                    if retention_env.is_success(total_reward):
                        retention_successes += 1
                
                rate = retention_successes / len(retention_task_ids) if retention_task_ids else 0.0
                

                if patch_type == "base":
                    p_base_retention = rate
                elif patch_type == "placebo":
                    p_placebo_retention = rate
                else:
                    patch_retention[patch_type] = rate
                    
            # Apply retention penalty to net gain
            # Net Gain = (Success_patch - Success_placebo) - max(0, Success_base_retention - Success_patch_retention)
            if p_base_retention is not None:
                for patch_type in ["demo", "contrast", "hint"]:
                    if patch_type in patch_gain_net and patch_type in patch_retention:
                        # Penalty: if patch causes retention drop compared to base
                        penalty = max(0.0, p_base_retention - patch_retention[patch_type])
                        patch_gain_net[patch_type] -= penalty
                        
        finally:
            if hasattr(retention_env, "close"):
                retention_env.close()
    
    return CPTLabels(
        p_base=p_base,
        p_placebo=p_placebo,
        n_per_condition=config.n_per_condition,
        patch_gain_raw=patch_gain_raw,
        patch_gain_net=patch_gain_net,
        p_base_retention=p_base_retention,
        p_placebo_retention=p_placebo_retention,
        patch_retention=patch_retention,
        total_episodes=len(config.patch_types) * config.n_per_condition,
    )


def run_cpt_batch(
    snapshots: list[Snapshot],
    env_factory: Callable[[], EnvironmentProtocol],
    student_policy: PolicyProtocol,
    teacher_hints: dict[str, TeacherHint],
    config: Optional[CPTConfig] = None,
    n_parallel: int = 4,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> dict[str, CPTLabels]:
    """
    Run CPT for a batch of snapshots.
    
    Args:
        snapshots: List of snapshots to process
        env_factory: Factory function to create environment instances
        student_policy: Student policy
        teacher_hints: Dict mapping snapshot IDs to teacher hints
        config: CPT configuration
        n_parallel: Number of parallel workers
        progress_callback: Optional callback called with completed count
        
    Returns:
        Dict mapping snapshot IDs to CPTLabels
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if config is None:
        config = CPTConfig()
    
    results = {}
    
    def process_snapshot(snapshot: Snapshot) -> tuple[str, CPTLabels]:
        env = env_factory()
        try:
            hint = teacher_hints.get(snapshot.id)
            if hint is None:
                raise ValueError(f"No teacher hint for snapshot {snapshot.id}")
            
            cpt = run_cpt(snapshot, env, student_policy, hint, config)
            return snapshot.id, cpt
        finally:
            if hasattr(env, "close"):
                env.close()
    
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {
            executor.submit(process_snapshot, snap): snap.id
            for snap in snapshots
        }
        
        completed = 0
        for future in as_completed(futures):
            snapshot_id, cpt = future.result()
            results[snapshot_id] = cpt
            
            completed += 1
            if progress_callback:
                progress_callback(completed)
    
    return results


def compute_cpt_statistics(cpt_labels: list[CPTLabels]) -> dict:
    """
    Compute summary statistics for CPT labels.
    
    Args:
        cpt_labels: List of CPTLabels objects
        
    Returns:
        Dictionary with statistics
    """
    import numpy as np
    
    if not cpt_labels:
        return {}
    
    elp_net = [l.ELP_net for l in cpt_labels]
    elp_raw = [l.ELP_raw for l in cpt_labels]
    
    routes = [l.route_net for l in cpt_labels]
    route_counts = {r: routes.count(r) for r in set(routes) if r}
    
    return {
        "ELP_net": {
            "mean": float(np.mean(elp_net)),
            "std": float(np.std(elp_net)),
            "min": float(np.min(elp_net)),
            "max": float(np.max(elp_net)),
            "median": float(np.median(elp_net)),
        },
        "ELP_raw": {
            "mean": float(np.mean(elp_raw)),
            "std": float(np.std(elp_raw)),
        },
        "route_distribution": route_counts,
        "placebo_effect": {
            "mean": float(np.mean([l.p_placebo - l.p_base for l in cpt_labels])),
            "std": float(np.std([l.p_placebo - l.p_base for l in cpt_labels])),
        },
        "n_snapshots": len(cpt_labels),
    }
