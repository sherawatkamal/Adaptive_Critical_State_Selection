"""
Uncertainty estimation for teachability characterization.

Implements multiple uncertainty estimators based on the agent's action distribution:
- Entropy: H(π(·|s))
- Margin: difference between top-2 action probabilities
- Top-k spread: difference between top-1 and top-k probabilities
- Effective actions: perplexity of action distribution
"""

import math
from typing import Protocol, Optional, Any
from dataclasses import dataclass

import numpy as np


class PolicyProtocol(Protocol):
    """Protocol for policy objects that provide action distributions."""
    
    def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
        """
        Get action probability distribution for given observation.
        
        Args:
            observation: Current observation string
            valid_actions: List of valid action strings
            
        Returns:
            Dictionary mapping action strings to probabilities
        """
        ...


class UncertaintyEstimator(Protocol):
    """Protocol for uncertainty estimation functions."""
    
    def __call__(
        self,
        policy: PolicyProtocol,
        observation: str,
        valid_actions: list[str],
    ) -> float:
        """
        Compute uncertainty for given state.
        
        Args:
            policy: Policy object with get_action_distribution method
            observation: Current observation string
            valid_actions: List of valid action strings
            
        Returns:
            Uncertainty value (higher = more uncertain)
        """
        ...


def compute_entropy(
    policy: PolicyProtocol,
    observation: str,
    valid_actions: list[str],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute entropy of action distribution: H(π(·|s)) = -Σ_a π(a|s) log π(a|s)
    
    Higher entropy indicates more uncertainty in action selection.
    
    Args:
        policy: Policy object
        observation: Current observation
        valid_actions: List of valid actions
        epsilon: Small value to avoid log(0)
        
    Returns:
        Entropy value (bits if using log2, nats if using log)
    """
    action_probs = policy.get_action_distribution(observation, valid_actions)
    probs = list(action_probs.values())
    
    entropy = 0.0
    for p in probs:
        if p > epsilon:
            entropy -= p * math.log(p + epsilon)
    
    return entropy


def compute_margin(
    policy: PolicyProtocol,
    observation: str,
    valid_actions: list[str],
) -> float:
    """
    Compute margin between top-2 actions: π(a_1|s) - π(a_2|s)
    
    Lower margin indicates more uncertainty (close competition between actions).
    We return 1 - margin so higher value = more uncertainty.
    
    Args:
        policy: Policy object
        observation: Current observation
        valid_actions: List of valid actions
        
    Returns:
        1 - margin value (higher = more uncertain)
    """
    action_probs = policy.get_action_distribution(observation, valid_actions)
    probs = sorted(action_probs.values(), reverse=True)
    
    if len(probs) < 2:
        return 0.0  # Only one action, no uncertainty
    
    margin = probs[0] - probs[1]
    return 1.0 - margin  # Invert so higher = more uncertain


def compute_topk_spread(
    policy: PolicyProtocol,
    observation: str,
    valid_actions: list[str],
    k: int = 5,
) -> float:
    """
    Compute spread between top-1 and top-k action: π(a_1|s) - π(a_k|s)
    
    Lower spread indicates probability is distributed across more actions.
    We return 1 - spread so higher value = more uncertainty.
    
    Args:
        policy: Policy object
        observation: Current observation
        valid_actions: List of valid actions
        k: The k-th action to compare with top action
        
    Returns:
        1 - spread value (higher = more uncertain)
    """
    action_probs = policy.get_action_distribution(observation, valid_actions)
    probs = sorted(action_probs.values(), reverse=True)
    
    if len(probs) < k:
        k = len(probs)
    
    if k < 1:
        return 0.0
    
    spread = probs[0] - probs[k - 1]
    return 1.0 - spread  # Invert so higher = more uncertain


def compute_effective_actions(
    policy: PolicyProtocol,
    observation: str,
    valid_actions: list[str],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute effective number of actions: exp(H) = perplexity
    
    Higher perplexity indicates the policy is considering more actions.
    
    Args:
        policy: Policy object
        observation: Current observation
        valid_actions: List of valid actions
        epsilon: Small value to avoid log(0)
        
    Returns:
        Effective number of actions (perplexity)
    """
    entropy = compute_entropy(policy, observation, valid_actions, epsilon)
    return math.exp(entropy)


def compute_all_uncertainty(
    policy: PolicyProtocol,
    observation: str,
    valid_actions: list[str],
) -> dict[str, float]:
    """
    Compute all uncertainty estimators for a given state.
    
    Args:
        policy: Policy object
        observation: Current observation
        valid_actions: List of valid actions
        
    Returns:
        Dictionary with all uncertainty measures
    """
    return {
        "entropy": compute_entropy(policy, observation, valid_actions),
        "margin": compute_margin(policy, observation, valid_actions),
        "topk_spread": compute_topk_spread(policy, observation, valid_actions),
        "effective_actions": compute_effective_actions(policy, observation, valid_actions),
    }


def compute_uncertainty_from_probs(probs: list[float]) -> dict[str, float]:
    """
    Compute uncertainty from raw probability list.
    
    Useful when probabilities are already computed.
    
    Args:
        probs: List of action probabilities (should sum to ~1)
        
    Returns:
        Dictionary with all uncertainty measures
    """
    epsilon = 1e-10
    probs = sorted(probs, reverse=True)
    
    # Entropy
    entropy = 0.0
    for p in probs:
        if p > epsilon:
            entropy -= p * math.log(p + epsilon)
    
    # Margin
    margin = probs[0] - probs[1] if len(probs) > 1 else 1.0
    
    # Top-k spread
    k = min(5, len(probs))
    spread = probs[0] - probs[k - 1] if k > 0 else 1.0
    
    # Effective actions
    effective = math.exp(entropy)
    
    return {
        "entropy": entropy,
        "margin": 1.0 - margin,
        "topk_spread": 1.0 - spread,
        "effective_actions": effective,
    }


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation."""
    
    primary_estimator: str = "entropy"  # Which estimator to use as U
    topk_k: int = 5
    epsilon: float = 1e-10
    
    @classmethod
    def from_yaml(cls, path: str) -> "UncertaintyConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("uncertainty", {}))


def verify_warm_start(
    policy: PolicyProtocol,
    sample_observations: list[tuple[str, list[str]]],
    n_samples: int = 50,
) -> dict:
    """
    Verify the policy has meaningful structure (not collapsed or uniform).
    
    This should be run before trajectory collection to ensure the policy
    produces informative action distributions.
    
    Args:
        policy: Policy to verify
        sample_observations: List of (observation, valid_actions) tuples
        n_samples: Number of samples to check
        
    Returns:
        Dictionary with verification results
    """
    entropies = []
    max_entropies = []
    
    for obs, valid_actions in sample_observations[:n_samples]:
        entropy = compute_entropy(policy, obs, valid_actions)
        max_possible = math.log(len(valid_actions)) if valid_actions else 0
        
        entropies.append(entropy)
        max_entropies.append(max_possible)
    
    mean_entropy = np.mean(entropies)
    mean_max = np.mean(max_entropies)
    
    if mean_max > 0:
        entropy_ratio = mean_entropy / mean_max
    else:
        entropy_ratio = 0.0
    
    return {
        "mean_entropy": float(mean_entropy),
        "mean_max_entropy": float(mean_max),
        "entropy_ratio": float(entropy_ratio),
        "is_valid": 0.1 < entropy_ratio < 0.9,
        "n_samples": n_samples,
        "diagnosis": _diagnose_warm_start(entropy_ratio),
    }


def _diagnose_warm_start(entropy_ratio: float) -> str:
    """Provide diagnostic message for warm start verification."""
    if entropy_ratio < 0.1:
        return "Policy appears collapsed (very low entropy). Model may not be properly initialized."
    elif entropy_ratio > 0.9:
        return "Policy appears nearly uniform (very high entropy). Model may not have learned meaningful preferences."
    else:
        return "Policy has meaningful structure. Ready for trajectory collection."
