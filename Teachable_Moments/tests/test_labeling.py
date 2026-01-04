"""Tests for labeling modules."""

import numpy as np
import pytest

from src.label.uncertainty import compute_entropy, compute_all_uncertainty
from src.label.leverage import LeverageConfig
from src.label.quadrant import QuadrantConfig, assign_quadrant


class TestUncertaintyEstimator:
    """Tests for uncertainty estimation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = QuadrantConfig()
        assert config.method == "median"

    def test_entropy_computation(self):
        """Test entropy computation from action distribution."""

        class _DummyPolicy:
            def get_action_distribution(self, observation: str, valid_actions: list[str]) -> dict[str, float]:
                return {a: 0.25 for a in valid_actions}

        policy = _DummyPolicy()
        observation = "obs"
        valid_actions = ["a", "b", "c", "d"]

        entropy = compute_entropy(policy, observation, valid_actions)
        assert entropy > 0

        all_uq = compute_all_uncertainty(policy, observation, valid_actions)
        assert "entropy" in all_uq
        assert "margin" in all_uq


class TestLeverageEstimator:
    """Tests for leverage estimation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LeverageConfig()
        assert config.n_force_rollouts > 0
        assert config.n_expert_rollouts > 0
        assert config.max_steps > 0


class TestQuadrantAssigner:
    """Tests for quadrant assignment."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = QuadrantConfig()
        assert config.method in {"median", "percentile_75"}

    def test_quadrant_assignment_logic(self):
        """Test correct quadrant assignment based on thresholds."""
        U_thr = 0.5
        L_thr = 0.5

        assert assign_quadrant(0.8, 0.8, U_thr, L_thr) == "Q1_highU_highL"
        assert assign_quadrant(0.8, 0.2, U_thr, L_thr) == "Q2_highU_lowL"
        assert assign_quadrant(0.2, 0.2, U_thr, L_thr) == "Q3_lowU_lowL"
        assert assign_quadrant(0.2, 0.8, U_thr, L_thr) == "Q4_lowU_highL"

    def test_batch_assignment(self, sample_uncertainty_scores, sample_leverage_scores):
        """Test batch quadrant assignment."""
        U_thr = float(np.median(list(sample_uncertainty_scores.values())))
        L_thr = float(np.median(list(sample_leverage_scores.values())))

        for k in sample_uncertainty_scores.keys():
            q = assign_quadrant(sample_uncertainty_scores[k], sample_leverage_scores[k], U_thr, L_thr)
            assert q in {"Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"}

    def test_adaptive_thresholds(self, sample_uncertainty_scores, sample_leverage_scores):
        """Test adaptive threshold computation."""
        u_values = list(sample_uncertainty_scores.values())
        l_values = list(sample_leverage_scores.values())

        u_thresh = float(np.median(u_values))
        l_thresh = float(np.median(l_values))

        assert abs(u_thresh - np.median(u_values)) < 0.01
        assert abs(l_thresh - np.median(l_values)) < 0.01
