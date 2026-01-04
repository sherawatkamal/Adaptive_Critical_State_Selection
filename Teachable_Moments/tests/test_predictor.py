"""Tests for predictor modules."""

import pytest
import numpy as np
from src.features.tier1_structural import extract_structural_features
from src.features.tier2_embeddings import MockEmbedder, extract_embedding_features
from src.predictor.evaluation import compute_classification_metrics, compute_regression_metrics
from src.predictor.multitask_model import PredictorConfig, TeachabilityPredictor
from src.predictor.training import prepare_labels


class TestStructuralFeatures:
    """Tests for tier-1 structural features."""

    def test_extract_structural_features_vector_shape(self):
        snapshot = {
            "step_idx": 0,
            "trajectory_length": 10,
            "observation": "Task: buy a laptop\n\nYou are on the homepage.",
            "valid_actions": ["search[laptop]", "click[Cart]", "back"],
            "action_probs": {"search[laptop]": 0.7, "click[Cart]": 0.2, "back": 0.1},
            "instruction_text": "buy a laptop",
        }

        feats = extract_structural_features(snapshot)
        vec = feats.to_vector()

        assert vec.shape == (12,)
        assert np.isfinite(vec).all()


class TestEmbeddingFeatures:
    """Tests for tier-2 embedding features."""

    def test_extract_embedding_features_dimensions(self):
        embedder = MockEmbedder(dim=8)
        snapshot = {
            "observation": "You are viewing search results.",
            "instruction_text": "buy a laptop",
            "last_action": "search[laptop]",
        }
        feats = extract_embedding_features(snapshot, embedder)

        assert feats.state_embedding.shape == (8,)
        assert feats.task_embedding.shape == (8,)
        assert np.isfinite(feats.state_embedding).all()


class TestTeachabilityPredictor:
    """Tests for multitask predictor model."""

    def test_config_defaults(self):
        config = PredictorConfig()
        assert len(config.hidden_dims) > 0
        assert config.embedding_dim > 0
        assert config.structural_dim > 0

    def test_predictor_predict_smoke(self):
        pytest.importorskip("torch")

        config = PredictorConfig(
            hidden_dims=[16],
            dropout=0.0,
            predict_uncertainty=True,
            predict_leverage=True,
            predict_quadrant=True,
            predict_elp=True,
            use_structural=True,
            use_embeddings=True,
            embedding_dim=8,
            structural_dim=12,
        )
        predictor = TeachabilityPredictor(config)

        predictor._build_model(input_dim=config.structural_dim + config.embedding_dim)
        predictor._is_trained = True

        structural = np.zeros((config.structural_dim,), dtype=np.float32)
        embeddings = np.zeros((config.embedding_dim,), dtype=np.float32)
        result = predictor.predict(structural=structural, embeddings=embeddings)

        assert result.uncertainty is not None
        assert result.leverage is not None
        assert result.elp is not None
        assert result.quadrant in predictor.QUADRANT_LABELS
        assert isinstance(result.quadrant_probs, dict)
        assert len(result.quadrant_probs) == 4


class TestPredictorTrainingUtilities:
    """Tests for predictor training utilities."""

    def test_prepare_labels_supports_new_schema(self):
        snapshots = [
            {
                "U": 0.7,
                "leverage": {"L_local": 0.2},
                "cpt": {"ELP_net": 0.1},
                "quadrant": "Q2_highU_lowL",
            }
        ]
        quadrant_labels = [
            "Q1_highU_highL",
            "Q2_highU_lowL",
            "Q3_lowU_lowL",
            "Q4_lowU_highL",
        ]
        labels = prepare_labels(snapshots, quadrant_labels)

        assert labels["uncertainty"].shape == (1,)
        assert float(labels["uncertainty"][0]) == pytest.approx(0.7)
        assert float(labels["leverage"][0]) == pytest.approx(0.2)
        assert float(labels["elp"][0]) == pytest.approx(0.1)
        assert int(labels["quadrant"][0]) == 1


class TestPredictorEvaluationUtilities:
    """Tests for predictor evaluation utilities."""

    def test_regression_metrics(self):
        predictions = np.array([0.5, 0.6, 0.7, 0.8])
        targets = np.array([0.55, 0.58, 0.72, 0.75])
        metrics = compute_regression_metrics(predictions, targets)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_classification_metrics(self):
        predictions = np.array([0, 1, 2, 3, 0, 1])
        targets = np.array([0, 1, 2, 3, 1, 1])
        metrics = compute_classification_metrics(predictions, targets, n_classes=4)

        assert "accuracy" in metrics
        assert "per_class" in metrics
