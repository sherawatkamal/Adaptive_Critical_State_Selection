"""
Tier 1: Structural features for teachability prediction.

These are fast, interpretable features computed directly from
snapshot data without requiring model inference.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class StructuralFeatures:
    """Tier 1 structural features for a snapshot."""
    
    # Trajectory position
    step_index: int
    trajectory_length: int
    relative_position: float  # step_index / trajectory_length
    
    # Action space
    n_available_actions: int
    action_space_entropy: float
    
    # State characteristics
    state_length: int
    n_numeric_tokens: int
    n_product_mentions: int
    
    # Task characteristics
    task_complexity: float  # estimated from description
    n_constraints: int  # number of attribute constraints
    
    # Historical context
    n_prior_failures: int
    steps_since_last_success: int
    
    # Label-derived (if available)
    uncertainty_bin: Optional[str] = None  # low, medium, high
    leverage_bin: Optional[str] = None
    quadrant: Optional[str] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.step_index,
            self.trajectory_length,
            self.relative_position,
            self.n_available_actions,
            self.action_space_entropy,
            self.state_length,
            self.n_numeric_tokens,
            self.n_product_mentions,
            self.task_complexity,
            self.n_constraints,
            self.n_prior_failures,
            self.steps_since_last_success,
        ])
    
    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "trajectory_length": self.trajectory_length,
            "relative_position": self.relative_position,
            "n_available_actions": self.n_available_actions,
            "action_space_entropy": self.action_space_entropy,
            "state_length": self.state_length,
            "n_numeric_tokens": self.n_numeric_tokens,
            "n_product_mentions": self.n_product_mentions,
            "task_complexity": self.task_complexity,
            "n_constraints": self.n_constraints,
            "n_prior_failures": self.n_prior_failures,
            "steps_since_last_success": self.steps_since_last_success,
            "uncertainty_bin": self.uncertainty_bin,
            "leverage_bin": self.leverage_bin,
            "quadrant": self.quadrant,
        }


def compute_action_space_entropy(action_probs: dict[str, float]) -> float:
    """
    Compute entropy of action probability distribution.
    
    Args:
        action_probs: Dict mapping actions to probabilities
        
    Returns:
        Entropy value
    """
    probs = np.array(list(action_probs.values()))
    probs = probs[probs > 0]  # Filter zero probs
    
    if len(probs) == 0:
        return 0.0
    
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def count_numeric_tokens(text: str) -> int:
    """Count tokens that are numbers or contain digits."""
    tokens = text.split()
    return sum(1 for t in tokens if any(c.isdigit() for c in t))


def count_product_mentions(text: str) -> int:
    """Count product-related mentions in text."""
    product_keywords = [
        "product", "item", "buy", "price", "$", 
        "add to cart", "purchase", "order"
    ]
    
    text_lower = text.lower()
    return sum(1 for kw in product_keywords if kw in text_lower)


def estimate_task_complexity(task_description: str) -> float:
    """
    Estimate task complexity from description.
    
    Returns value between 0 and 1.
    """
    if not task_description:
        return 0.5
    
    # Complexity indicators
    complexity_terms = [
        "and", "or", "with", "without", "between", "under", "over",
        "specific", "exact", "must", "should", "need"
    ]
    
    desc_lower = task_description.lower()
    term_count = sum(1 for term in complexity_terms if term in desc_lower)
    
    # Word count also indicates complexity
    word_count = len(task_description.split())
    
    # Normalize to 0-1
    complexity = min(1.0, (term_count * 0.1) + (word_count * 0.02))
    
    return complexity


def count_constraints(task_description: str) -> int:
    """Count attribute constraints in task description."""
    constraint_patterns = [
        "color", "size", "price", "brand", "material", "style",
        "rating", "review", "under $", "over $", "between"
    ]
    
    desc_lower = task_description.lower()
    return sum(1 for p in constraint_patterns if p in desc_lower)


def extract_structural_features(
    snapshot: dict,
    trajectory_context: Optional[dict] = None,
) -> StructuralFeatures:
    """Extract tier 1 structural features from a snapshot.
    
    This function accepts BOTH the old schema (state/available_actions/task...) and
    the newer schema used by `Snapshot`/`LabeledSnapshot` (observation/valid_actions/...).
    
    Args:
        snapshot: Snapshot dictionary
        trajectory_context: Optional context about full trajectory
        
    Returns:
        StructuralFeatures instance
    """
    # Trajectory position - handle both field names
    step_index = snapshot.get("step_index", snapshot.get("step_idx", 0))
    trajectory_length = snapshot.get("trajectory_length", 1)
    if trajectory_context:
        trajectory_length = trajectory_context.get("length", trajectory_length)
    
    relative_position = step_index / max(1, trajectory_length)
    
    # Action space - handle both field names
    available_actions = snapshot.get("available_actions", snapshot.get("valid_actions", [])) or []
    n_available_actions = len(available_actions)
    
    # Action probs - check multiple locations
    action_probs = snapshot.get("action_probs")
    if action_probs is None:
        action_probs = snapshot.get("policy_outputs", {}).get("action_probs")
    if action_probs is None:
        action_probs = snapshot.get("uncertainty_features", {}).get("action_probs")
    if action_probs is None:
        action_probs = {}
    
    action_space_entropy = compute_action_space_entropy(action_probs)
    
    # State text - handle both field names
    state_text = snapshot.get("state")
    if state_text is None:
        state_text = snapshot.get("observation", "")
    
    state_length = len(state_text)
    n_numeric_tokens = count_numeric_tokens(state_text)
    n_product_mentions = count_product_mentions(state_text)
    
    # Task description - check multiple locations
    task_description = (
        snapshot.get("instruction_text")
        or snapshot.get("task_description")
        or snapshot.get("agent_prefix")
        or snapshot.get("task", {}).get("description", "")
        or ""
    )
    task_complexity = estimate_task_complexity(task_description)
    n_constraints = count_constraints(task_description)
    
    # Historical context
    n_prior_failures = 0
    steps_since_last_success = step_index
    
    if trajectory_context:
        n_prior_failures = trajectory_context.get("prior_failures", 0)
        steps_since_last_success = trajectory_context.get("steps_since_success", step_index)
    
    # Label-derived features - handle both nested and flat structures
    uncertainty = snapshot.get("uncertainty", snapshot.get("uncertainty_features", {}))
    leverage = snapshot.get("leverage", {})
    
    # If leverage is a LeverageLabels dict, extract value
    if isinstance(leverage, dict) is False and hasattr(leverage, "to_dict"):
        leverage = leverage.to_dict()
    if isinstance(leverage, dict) is False:
        leverage = {}
    
    uncertainty_bin = None
    entropy = uncertainty.get("entropy")
    if entropy is None:
        entropy = snapshot.get("U")  # LabeledSnapshot uses U for uncertainty
    if entropy is not None:
        uncertainty_bin = "high" if entropy > 1.5 else ("low" if entropy < 0.5 else "medium")
    
    leverage_bin = None
    l_local = leverage.get("L_local") if isinstance(leverage, dict) else None
    if l_local is not None:
        leverage_bin = "high" if l_local > 0.3 else ("low" if l_local < 0.1 else "medium")
    
    quadrant = snapshot.get("quadrant")
    
    return StructuralFeatures(
        step_index=step_index,
        trajectory_length=trajectory_length,
        relative_position=relative_position,
        n_available_actions=n_available_actions,
        action_space_entropy=action_space_entropy,
        state_length=state_length,
        n_numeric_tokens=n_numeric_tokens,
        n_product_mentions=n_product_mentions,
        task_complexity=task_complexity,
        n_constraints=n_constraints,
        n_prior_failures=n_prior_failures,
        steps_since_last_success=steps_since_last_success,
        uncertainty_bin=uncertainty_bin,
        leverage_bin=leverage_bin,
        quadrant=quadrant,
    )


def extract_batch(
    snapshots: list[dict],
    trajectory_contexts: Optional[dict[str, dict]] = None,
) -> list[StructuralFeatures]:
    """
    Extract structural features for a batch of snapshots.
    
    Args:
        snapshots: List of snapshot dicts
        trajectory_contexts: Optional dict mapping trajectory IDs to contexts
        
    Returns:
        List of StructuralFeatures
    """
    results = []
    
    for snap in snapshots:
        traj_id = snap.get("trajectory_id")
        context = None
        if trajectory_contexts and traj_id:
            context = trajectory_contexts.get(traj_id)
        
        features = extract_structural_features(snap, context)
        results.append(features)
    
    return results


def get_feature_names() -> list[str]:
    """Get ordered list of feature names for vector representation."""
    return [
        "step_index",
        "trajectory_length",
        "relative_position",
        "n_available_actions",
        "action_space_entropy",
        "state_length",
        "n_numeric_tokens",
        "n_product_mentions",
        "task_complexity",
        "n_constraints",
        "n_prior_failures",
        "steps_since_last_success",
    ]
