"""
Teachability predictor module.

Predicts teachability metrics from features, enabling cheap
identification of teachable moments without full labeling.
"""

from .multitask_model import (
    PredictorConfig,
    PredictionResult,
    TeachabilityPredictor,
    MockPredictor,
)

from .training import (
    TrainingConfig,
    TrainingResult,
    prepare_labels,
    train_predictor,
    evaluate_predictor as evaluate_during_training,
    run_predictor_training,
)

from .evaluation import (
    EvaluationMetrics,
    compute_regression_metrics,
    compute_classification_metrics,
    evaluate_predictor,
    analyze_prediction_errors,
    run_predictor_evaluation,
)

__all__ = [
    # Model
    "PredictorConfig",
    "PredictionResult",
    "TeachabilityPredictor",
    "MockPredictor",
    # Training
    "TrainingConfig",
    "TrainingResult",
    "prepare_labels",
    "train_predictor",
    "evaluate_during_training",
    "run_predictor_training",
    # Evaluation
    "EvaluationMetrics",
    "compute_regression_metrics",
    "compute_classification_metrics",
    "evaluate_predictor",
    "analyze_prediction_errors",
    "run_predictor_evaluation",
]
