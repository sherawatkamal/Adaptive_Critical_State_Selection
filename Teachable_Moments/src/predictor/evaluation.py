"""
Evaluation for teachability predictor.

Measures prediction quality and analyzes where the predictor
succeeds and fails.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for predictor evaluation."""
    
    # Overall metrics
    n_samples: int
    
    # Regression metrics (uncertainty, leverage, ELP)
    regression_metrics: dict[str, dict]
    
    # Classification metrics (quadrant)
    classification_metrics: dict
    
    # Per-quadrant breakdown
    per_quadrant_metrics: dict[str, dict]
    
    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "regression_metrics": self.regression_metrics,
            "classification_metrics": self.classification_metrics,
            "per_quadrant_metrics": self.per_quadrant_metrics,
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_regression_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute regression metrics.
    
    Args:
        predictions: Predicted values
        labels: True values
        
    Returns:
        Dict of metrics
    """
    mse = float(np.mean((predictions - labels) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predictions - labels)))
    
    # R-squared
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Correlation
    if np.std(predictions) > 0 and np.std(labels) > 0:
        correlation = float(np.corrcoef(predictions, labels)[0, 1])
    else:
        correlation = 0.0
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": correlation,
    }




def _rankdata_simple(x: np.ndarray) -> np.ndarray:
    """Return ranks 0..n-1 (ties broken arbitrarily)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman_corr_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via Pearson corr of ranks.

    Note: this is a simple implementation and does not handle ties optimally.
    For our use (ranking snapshots), it is sufficient.
    """
    if len(a) < 2:
        return 0.0
    ra = _rankdata_simple(a)
    rb = _rankdata_simple(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def precision_pos_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int, threshold: float = 0.0) -> float:
    """Among the top-k predicted, what fraction have y_true > threshold?"""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = int(min(max(k, 1), n))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_idx] > threshold))


def topk_overlap(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Overlap@k between predicted top-k and true top-k (normalized by k)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = int(min(max(k, 1), n))
    top_pred = set(np.argsort(y_score)[::-1][:k].tolist())
    top_true = set(np.argsort(y_true)[::-1][:k].tolist())
    return float(len(top_pred & top_true) / k)


def ndcg_at_k(relevance: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k for a single ranking.

    We use linear gain (gain = relevance) and log2 discount.
    """
    n = len(relevance)
    if n == 0:
        return 0.0
    k = int(min(max(k, 1), n))

    order = np.argsort(y_score)[::-1][:k]
    rel = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel * discounts))

    ideal_order = np.argsort(relevance)[::-1][:k]
    ideal_rel = relevance[ideal_order]
    idcg = float(np.sum(ideal_rel * discounts))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def compute_elp_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, ks: tuple[int, ...] = (10, 50)) -> dict:
    """Ranking-centric metrics for ELP.

    We treat positive ELP as "teachable"; negative ELP is clipped to 0 for NDCG relevance.
    """
    metrics: dict[str, float] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics["spearman"] = spearman_corr_simple(y_true, y_pred)

    relevance = np.maximum(y_true, 0.0)
    for k in ks:
        metrics[f"precision_pos_at_{k}"] = precision_pos_at_k(y_true, y_pred, k=k, threshold=0.0)
        metrics[f"topk_overlap_at_{k}"] = topk_overlap(y_true, y_pred, k=k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(relevance, y_pred, k=k)

    return metrics
def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 4,
) -> dict:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: True class indices
        n_classes: Number of classes
        
    Returns:
        Dict of metrics
    """
    # Accuracy
    accuracy = float(np.mean(predictions == labels))
    
    # Per-class metrics
    per_class = {}
    for c in range(n_classes):
        mask = labels == c
        if np.sum(mask) > 0:
            class_acc = float(np.mean(predictions[mask] == labels[mask]))
            per_class[f"class_{c}"] = {
                "accuracy": class_acc,
                "support": int(np.sum(mask)),
            }
    
    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(predictions, labels):
        confusion[int(true), int(pred)] += 1
    
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_predictor(
    predictor,
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    quadrant_labels: list[str],
) -> EvaluationMetrics:
    """
    Comprehensive evaluation of predictor.
    
    Args:
        predictor: Trained TeachabilityPredictor
        features: (N, feature_dim) array
        labels: Dict of label arrays
        quadrant_labels: List of quadrant label names
        
    Returns:
        EvaluationMetrics with detailed results
    """
    n = features.shape[0]
    
    # Get predictions
    predictions = predictor.predict_batch(features)
    
    # Extract prediction values
    pred_uncertainty = np.array([p.uncertainty or 0 for p in predictions])
    pred_leverage = np.array([p.leverage or 0 for p in predictions])
    pred_elp = np.array([p.elp or 0 for p in predictions])
    pred_quadrant = np.array([
        quadrant_labels.index(p.quadrant) if p.quadrant in quadrant_labels else 0
        for p in predictions
    ])
    
    # Regression metrics
    regression_metrics = {
        "uncertainty": compute_regression_metrics(pred_uncertainty, labels["uncertainty"]),
        "leverage": compute_regression_metrics(pred_leverage, labels["leverage"]),
        "elp": compute_regression_metrics(pred_elp, labels["elp"]),
    }
    
    # Ranking metrics (ELP): how well do we identify the most teachable moments?
    try:
        rank_m = compute_elp_ranking_metrics(labels["elp"], pred_elp, ks=(10, 50))
        regression_metrics["elp"].update({
            "spearman": rank_m.get("spearman", 0.0),
            "precision_pos_at_10": rank_m.get("precision_pos_at_10", 0.0),
            "precision_pos_at_50": rank_m.get("precision_pos_at_50", 0.0),
            "topk_overlap_at_10": rank_m.get("topk_overlap_at_10", 0.0),
            "topk_overlap_at_50": rank_m.get("topk_overlap_at_50", 0.0),
            "ndcg_at_10": rank_m.get("ndcg_at_10", 0.0),
            "ndcg_at_50": rank_m.get("ndcg_at_50", 0.0),
        })
    except Exception as e:
        logger.warning(f"Failed to compute ranking metrics: {e}")

    # Classification metrics
    classification_metrics = compute_classification_metrics(
        pred_quadrant, labels["quadrant"], n_classes=4
    )
    
    # Per-quadrant breakdown
    per_quadrant_metrics = {}
    for i, q_label in enumerate(quadrant_labels):
        mask = labels["quadrant"] == i
        if np.sum(mask) == 0:
            continue
        
        per_quadrant_metrics[q_label] = {
            "n_samples": int(np.sum(mask)),
            "uncertainty_mae": float(np.mean(np.abs(pred_uncertainty[mask] - labels["uncertainty"][mask]))),
            "leverage_mae": float(np.mean(np.abs(pred_leverage[mask] - labels["leverage"][mask]))),
            "elp_mae": float(np.mean(np.abs(pred_elp[mask] - labels["elp"][mask]))),
            "quadrant_accuracy": float(np.mean(pred_quadrant[mask] == labels["quadrant"][mask])),
        }
    
    return EvaluationMetrics(
        n_samples=n,
        regression_metrics=regression_metrics,
        classification_metrics=classification_metrics,
        per_quadrant_metrics=per_quadrant_metrics,
    )


def analyze_prediction_errors(
    predictor,
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    snapshots: list[dict],
    quadrant_labels: list[str],
    top_k: int = 20,
) -> dict:
    """
    Analyze where the predictor makes largest errors.
    
    Args:
        predictor: Trained predictor
        features: Feature array
        labels: Label arrays
        snapshots: Original snapshot dicts
        quadrant_labels: Quadrant label names
        top_k: Number of worst errors to return
        
    Returns:
        Analysis of prediction errors
    """
    predictions = predictor.predict_batch(features)
    
    pred_elp = np.array([p.elp or 0 for p in predictions])
    elp_errors = np.abs(pred_elp - labels["elp"])
    
    # Find worst predictions
    worst_indices = np.argsort(elp_errors)[-top_k:][::-1]
    
    worst_cases = []
    for idx in worst_indices:
        worst_cases.append({
            "index": int(idx),
            "snapshot_id": snapshots[idx].get("id", ""),
            "true_elp": float(labels["elp"][idx]),
            "pred_elp": float(pred_elp[idx]),
            "error": float(elp_errors[idx]),
            "quadrant": snapshots[idx].get("quadrant", ""),
        })
    
    # Error distribution by quadrant
    error_by_quadrant = {}
    for i, q_label in enumerate(quadrant_labels):
        mask = labels["quadrant"] == i
        if np.sum(mask) > 0:
            error_by_quadrant[q_label] = {
                "mean_error": float(np.mean(elp_errors[mask])),
                "max_error": float(np.max(elp_errors[mask])),
                "error_std": float(np.std(elp_errors[mask])),
            }
    
    return {
        "worst_cases": worst_cases,
        "error_by_quadrant": error_by_quadrant,
        "overall_error_stats": {
            "mean": float(np.mean(elp_errors)),
            "std": float(np.std(elp_errors)),
            "median": float(np.median(elp_errors)),
            "max": float(np.max(elp_errors)),
        },
    }


def run_predictor_evaluation(
    predictor_path: str,
    snapshots_path: str,
    features_path: str,
    output_dir: str,
) -> dict:
    """
    Run full predictor evaluation pipeline.
    
    Args:
        predictor_path: Path to saved predictor
        snapshots_path: Path to labeled snapshots
        features_path: Path to precomputed features
        output_dir: Output directory
        
    Returns:
        Evaluation results summary
    """
    from .multitask_model import TeachabilityPredictor
    from .training import prepare_labels
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load predictor
    predictor = TeachabilityPredictor()
    predictor.load(predictor_path)
    
    # Load data
    with open(snapshots_path) as f:
        snapshots_data = json.load(f)
    snapshots = snapshots_data.get("snapshots", snapshots_data)
    
    features = np.load(features_path)
    
    # Prepare labels
    quadrant_labels = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
    labels = prepare_labels(snapshots, quadrant_labels)
    
    # Evaluate
    metrics = evaluate_predictor(predictor, features, labels, quadrant_labels)
    metrics.save(str(output_path / "evaluation_metrics.json"))
    
    # Error analysis
    error_analysis = analyze_prediction_errors(
        predictor, features, labels, snapshots, quadrant_labels
    )
    with open(output_path / "error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)
    
    return {
        "n_samples": metrics.n_samples,
        "regression_metrics": metrics.regression_metrics,
        "classification_accuracy": metrics.classification_metrics["accuracy"],
        "error_analysis": error_analysis["overall_error_stats"],
    }
