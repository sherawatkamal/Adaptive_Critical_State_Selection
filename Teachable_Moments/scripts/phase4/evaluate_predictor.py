#!/usr/bin/env python3
"""Comprehensive evaluation of the trained predictor.

Evaluates predictor performance on held-out test data:
- Regression metrics for uncertainty, leverage, ELP
- Classification metrics for quadrant prediction
- Error analysis and failure cases
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.utils import setup_logging, save_json, set_seed, get_timestamp
from src.predictor.multitask_model import TeachabilityPredictor, PredictorConfig
from src.predictor.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    analyze_prediction_errors,
)
from src.features.tier1_structural import extract_batch as extract_structural
from src.features.tier2_embeddings import create_embedder


def load_predictor(model_dir: Path) -> tuple[TeachabilityPredictor, dict[str, Any]]:
    """Load trained predictor and its configuration."""
    # Load training results for config
    results_path = model_dir / "training_results.json"
    with open(results_path) as f:
        training_results = json.load(f)
    
    config = training_results.get("config", {})
    
    # Create predictor with same config
    predictor_config = PredictorConfig(
        input_dim=config.get("feature_dim", 12),
        hidden_dim=config.get("hidden_dim", 128),
        n_quadrants=4,
    )
    
    predictor = TeachabilityPredictor(predictor_config)
    predictor.load(model_dir / "predictor.pt")
    
    return predictor, config


def load_test_snapshots(path: Path) -> list[dict[str, Any]]:
    """Load test snapshots."""
    with open(path) as f:
        return json.load(f)


def prepare_test_data(
    snapshots: list[dict[str, Any]],
    use_embeddings: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Prepare test features and labels."""
    logger = logging.getLogger(__name__)
    
    # Extract structural features
    structural_features = extract_structural(snapshots)
    features = structural_features
    
    if use_embeddings:
        logger.info("Extracting embedding features...")
        embedder = create_embedder(embedding_model)
        
        embedding_features = []
        for snap in snapshots:
            state_text = json.dumps(snap.get("state", {}))[:1000]
            emb = embedder.embed(state_text)
            embedding_features.append(emb.embedding)
        
        features = [
            struct + emb
            for struct, emb in zip(structural_features, embedding_features)
        ]
    
    # Extract labels
    labels = []
    for snap in snapshots:
        snap_labels = snap.get("labels", {})
        labels.append({
            "uncertainty": snap_labels.get("uncertainty", 0.5),
            "leverage": snap_labels.get("leverage", {}).get("L_local", 0.0),
            "quadrant": snap_labels.get("quadrant", "Q4"),
            "elp": snap_labels.get("cpt", {}).get("delta", 0.0),
            "snapshot_id": snap.get("snapshot_id", "unknown"),
        })
    
    return features, labels


def evaluate_predictor(
    predictor: TeachabilityPredictor,
    features: list[list[float]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run comprehensive evaluation."""
    # Get predictions
    predictions = predictor.predict_batch(features)
    
    # Extract ground truth
    true_uncertainty = [l["uncertainty"] for l in labels]
    true_leverage = [l["leverage"] for l in labels]
    true_quadrant = [l["quadrant"] for l in labels]
    true_elp = [l["elp"] for l in labels]
    
    # Extract predictions
    pred_uncertainty = [p.uncertainty for p in predictions]
    pred_leverage = [p.leverage for p in predictions]
    pred_quadrant = [p.quadrant for p in predictions]
    pred_elp = [p.elp for p in predictions]
    
    # Compute metrics
    results = {}
    
    # Uncertainty regression
    results["uncertainty"] = compute_regression_metrics(
        true_uncertainty, pred_uncertainty
    )
    
    # Leverage regression
    results["leverage"] = compute_regression_metrics(
        true_leverage, pred_leverage
    )
    
    # ELP regression
    results["elp"] = compute_regression_metrics(
        true_elp, pred_elp
    )
    
    # Quadrant classification
    results["quadrant"] = compute_classification_metrics(
        true_quadrant, pred_quadrant,
        classes=["Q1", "Q2", "Q3", "Q4"],
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate teachability predictor")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/phase4/predictor"),
        help="Directory with trained predictor",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("results/phase1/labeled_snapshots.json"),
        help="Path to test snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4/evaluation"),
        help="Output directory",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (last N%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    
    # Load predictor
    logger.info(f"Loading predictor from {args.model_dir}")
    predictor, training_config = load_predictor(args.model_dir)
    
    use_embeddings = training_config.get("use_embeddings", False)
    embedding_model = training_config.get("embedding_model", "all-MiniLM-L6-v2")
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    all_snapshots = load_test_snapshots(args.test_data)
    
    # Use last N% as test set (different from training)
    import random
    random.seed(args.seed)
    indices = list(range(len(all_snapshots)))
    random.shuffle(indices)
    
    test_size = int(len(all_snapshots) * args.test_ratio)
    test_indices = indices[-test_size:]
    test_snapshots = [all_snapshots[i] for i in test_indices]
    
    logger.info(f"Test set size: {len(test_snapshots)}")
    
    # Prepare test features
    logger.info("Preparing test features...")
    features, labels = prepare_test_data(
        test_snapshots,
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
    )
    
    # Evaluate
    logger.info("Running evaluation...")
    results = evaluate_predictor(predictor, features, labels)
    
    # Analyze errors
    logger.info("Analyzing prediction errors...")
    predictions = predictor.predict_batch(features)
    
    error_analysis = analyze_prediction_errors(
        predictions=predictions,
        labels=labels,
        n_worst=10,
    )
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": get_timestamp(),
        "config": {
            "model_dir": str(args.model_dir),
            "test_data": str(args.test_data),
            "test_ratio": args.test_ratio,
            "test_size": len(test_snapshots),
            "use_embeddings": use_embeddings,
            "seed": args.seed,
        },
        "metrics": results,
        "error_analysis": error_analysis,
    }
    
    save_json(output, args.output_dir / "evaluation_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTOR EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTest set size: {len(test_snapshots)}")
    
    print("\n--- Regression Tasks ---")
    for task in ["uncertainty", "leverage", "elp"]:
        m = results[task]
        print(f"\n{task.upper()}:")
        print(f"  MSE:  {m['mse']:.4f}")
        print(f"  RMSE: {m['rmse']:.4f}")
        print(f"  MAE:  {m['mae']:.4f}")
        print(f"  R²:   {m['r2']:.4f}")
        print(f"  Corr: {m['correlation']:.4f}")
    
    print("\n--- Classification Task ---")
    q_metrics = results["quadrant"]
    print(f"\nQUADRANT:")
    print(f"  Accuracy: {q_metrics['accuracy']:.4f}")
    print(f"  Per-class:")
    for cls, acc in q_metrics.get("per_class_accuracy", {}).items():
        print(f"    {cls}: {acc:.4f}")
    
    print("\n--- Error Analysis ---")
    print(f"Worst predictions:")
    for i, err in enumerate(error_analysis.get("worst_cases", [])[:5], 1):
        print(f"  {i}. {err['snapshot_id']}: error={err['error']:.4f}")
    
    print("=" * 60)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
