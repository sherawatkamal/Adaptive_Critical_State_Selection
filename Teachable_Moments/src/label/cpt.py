"""
CPT (Contextual Patch Test) module for teachability validation.

This module provides the interface for running CPT experiments.
The actual implementation is in patch_gain.py, this is a facade module
for compatibility with scripts that import from src.label.cpt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .patch_gain import (
    CPTConfig,
    run_cpt,
    generate_patch_text,
    PLACEBO_TEMPLATE,
    DEMO_TEMPLATE,
    CONTRAST_TEMPLATE,
    HINT_TEMPLATE,
)
from ..data.snapshot import Snapshot, CPTLabels


@dataclass
class CPTResult:
    """Result of a CPT experiment on a single snapshot."""
    
    treatment_rewards: List[float]
    placebo_rewards: List[float]
    treatment_mean: float
    placebo_mean: float
    effect_size: float
    is_significant: bool
    p_value: float
    
    @classmethod
    def from_cpt_labels(cls, labels: CPTLabels) -> "CPTResult":
        """Create CPTResult from CPTLabels."""
        # Calculate effect size from patch gains
        max_gain = max(labels.patch_gain_net.values()) if labels.patch_gain_net else 0.0
        
        return cls(
            treatment_rewards=[labels.p_base + max_gain],
            placebo_rewards=[labels.p_placebo],
            treatment_mean=labels.p_base + max_gain,
            placebo_mean=labels.p_placebo,
            effect_size=labels.ELP_net,
            is_significant=labels.ELP_net > 0.1,  # Threshold for significance
            p_value=0.05 if labels.ELP_net > 0.1 else 0.5,  # Placeholder
        )


def run_cpt_experiment(
    snapshot: Snapshot,
    config: CPTConfig,
) -> CPTResult:
    """
    Run a CPT experiment for a single snapshot.
    
    This is a wrapper around run_cpt for script compatibility.
    
    Args:
        snapshot: The snapshot to evaluate
        config: CPT configuration
        
    Returns:
        CPTResult with treatment and placebo outcomes
    """
    # Run the underlying CPT implementation
    cpt_labels = run_cpt(snapshot, config)
    
    # Convert to CPTResult format expected by scripts
    return CPTResult.from_cpt_labels(cpt_labels)


# Re-export for convenience
__all__ = [
    "CPTConfig",
    "CPTResult",
    "run_cpt",
    "run_cpt_experiment",
    "generate_patch_text",
    "PLACEBO_TEMPLATE",
    "DEMO_TEMPLATE",
    "CONTRAST_TEMPLATE",
    "HINT_TEMPLATE",
]
