"""
Labeling modules for teachability characterization.
"""

from .uncertainty import (
    compute_entropy,
    compute_margin,
    compute_topk_spread,
    compute_effective_actions,
    compute_all_uncertainty,
    UncertaintyEstimator,
)
from .leverage import (
    LeverageConfig,
    estimate_leverage,
    run_rollouts,
    run_forced_rollouts,
)
from .patch_gain import (
    CPTConfig,
    generate_patch_text,
    run_cpt,
    PLACEBO_TEMPLATE,
    DEMO_TEMPLATE,
    CONTRAST_TEMPLATE,
    HINT_TEMPLATE,
)
from .depth import (
    compute_depth_from_leverage,
)
from .quadrant import (
    QuadrantConfig,
    compute_thresholds,
    assign_quadrant,
    partition_by_quadrant,
)

__all__ = [
    # Uncertainty
    "compute_entropy",
    "compute_margin",
    "compute_topk_spread",
    "compute_effective_actions",
    "compute_all_uncertainty",
    "UncertaintyEstimator",
    # Leverage
    "LeverageConfig",
    "estimate_leverage",
    "run_rollouts",
    "run_forced_rollouts",
    # CPT
    "CPTConfig",
    "generate_patch_text",
    "run_cpt",
    "PLACEBO_TEMPLATE",
    "DEMO_TEMPLATE",
    "CONTRAST_TEMPLATE",
    "HINT_TEMPLATE",
    # Depth
    "compute_depth_from_leverage",
    # Quadrant
    "QuadrantConfig",
    "compute_thresholds",
    "assign_quadrant",
    "partition_by_quadrant",
]
