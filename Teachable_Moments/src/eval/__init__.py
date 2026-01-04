"""Evaluation modules for teachable moments experiments."""

from .end2end import (
    EpisodeResult,
    EvaluationResult,
    run_episode,
    evaluate_end2end,
    compute_success_rate,
    compute_reward_metrics,
    bootstrap_confidence_interval,
    compare_models,
)
from .mechanistic import (
    QuadrantMetrics,
    QuadrantEvaluationResult,
    compute_pre_post_success,
)
from .stuckness import (
    StucknessMetrics,
    StuckPattern,
    EpisodeStuckness,
    detect_action_loops,
    detect_state_loops,
    detect_no_progress,
    detect_stuck_episodes,
)
from .transfer_matrix import (
    TransferResult,
    TransferMatrix,
    compute_transfer_matrix,
)
from .retention import (
    RetentionCheckpoint,
    RetentionCurve,
    RetentionAnalysis,
    detect_catastrophic_forgetting,
)
from .drift_panel import (
    PanelTask,
    EvaluationPanel,
    PanelResult,
    create_evaluation_panel,
    detect_drift,
)

__all__ = [
    # End-to-end
    "EpisodeResult",
    "EvaluationResult",
    "run_episode",
    "evaluate_end2end",
    "compute_success_rate",
    "compute_reward_metrics",
    "bootstrap_confidence_interval",
    "compare_models",
    # Mechanistic
    "QuadrantMetrics",
    "QuadrantEvaluationResult",
    "compute_pre_post_success",
    # Stuckness
    "StucknessMetrics",
    "StuckPattern",
    "EpisodeStuckness",
    "detect_action_loops",
    "detect_state_loops",
    "detect_no_progress",
    "detect_stuck_episodes",
    # Transfer matrix
    "TransferResult",
    "TransferMatrix",
    "compute_transfer_matrix",
    # Retention
    "RetentionCheckpoint",
    "RetentionCurve",
    "RetentionAnalysis",
    "detect_catastrophic_forgetting",
    # Drift panel
    "PanelTask",
    "EvaluationPanel",
    "PanelResult",
    "create_evaluation_panel",
    "detect_drift",
]
