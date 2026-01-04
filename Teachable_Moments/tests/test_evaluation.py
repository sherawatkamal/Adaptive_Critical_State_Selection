"""Tests for evaluation modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.eval.end2end import EpisodeResult, compute_success_rate, compute_reward_metrics
from src.eval.stuckness import detect_action_loops
from src.eval.transfer_matrix import TransferMatrix, TransferResult
from src.eval.retention import RetentionAnalysis, RetentionCheckpoint, detect_catastrophic_forgetting
from src.eval.drift_panel import PanelResult, detect_drift


class TestEnd2EndEvaluator:
    """Tests for end-to-end task evaluation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        results = []
        assert compute_success_rate(results) == 0.0

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        results = [
            EpisodeResult(task_id="t1", success=True, total_reward=1.0, num_steps=2, final_observation=""),
            EpisodeResult(task_id="t2", success=False, total_reward=0.0, num_steps=3, final_observation=""),
        ]
        assert 0.0 <= compute_success_rate(results) <= 1.0

    def test_success_rate_computation(self):
        """Test success rate metric computation."""
        results = [
            EpisodeResult(task_id="t1", success=True, total_reward=1.0, num_steps=1, final_observation=""),
            EpisodeResult(task_id="t2", success=True, total_reward=1.0, num_steps=1, final_observation=""),
            EpisodeResult(task_id="t3", success=False, total_reward=0.0, num_steps=1, final_observation=""),
            EpisodeResult(task_id="t4", success=True, total_reward=1.0, num_steps=1, final_observation=""),
        ]
        assert compute_success_rate(results) == pytest.approx(0.75, rel=1e-6)

    def test_average_reward_computation(self):
        """Test average reward metric."""
        results = [
            EpisodeResult(task_id="t1", success=True, total_reward=1.0, num_steps=10, final_observation=""),
            EpisodeResult(task_id="t2", success=False, total_reward=0.3, num_steps=20, final_observation=""),
        ]
        metrics = compute_reward_metrics(results)
        assert metrics["mean"] == pytest.approx(0.65, rel=1e-6)


class TestStucknessEvaluator:
    """Tests for stuckness detection and recovery."""

    def test_config_defaults(self):
        """Test default configuration values."""
        actions = ["a", "b", "a", "b", "a", "b"]
        patterns = detect_action_loops(actions, min_loop_length=2, min_repetitions=2)
        assert patterns

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        actions = ["click[a]", "click[b]", "click[a]", "click[b]"]
        patterns = detect_action_loops(actions, min_loop_length=2, min_repetitions=2)
        assert isinstance(patterns, list)

    def test_loop_detection(self):
        """Test detection of action loops."""
        actions = ["click[a]", "click[b]", "click[a]", "click[b]", "click[a]"]
        patterns = detect_action_loops(actions, min_loop_length=2, min_repetitions=2)
        assert patterns

    def test_no_loop_detection(self):
        """Test no false positive for varied actions."""
        actions = ["click[a]", "click[b]", "click[c]", "click[d]", "click[e]"]
        patterns = detect_action_loops(actions, min_loop_length=2, min_repetitions=3)
        assert patterns == []


class TestTransferEvaluator:
    """Tests for transfer matrix evaluation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        tm = TransferMatrix(supervision="demo")
        assert tm.get_matrix().shape == (4, 4)

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        tm = TransferMatrix(supervision="demo")
        assert tm.results == []

    def test_transfer_matrix_structure(self):
        """Test 4x4 transfer matrix structure."""
        tm = TransferMatrix(
            supervision="demo",
            results=[
                TransferResult(
                    source_quadrant="Q1_highU_highL",
                    target_quadrant="Q2_highU_lowL",
                    source_supervision="demo",
                    baseline_success=0.5,
                    trained_success=0.6,
                    delta=0.1,
                    n_episodes=3,
                )
            ],
        )
        assert tm.get_matrix().shape == (4, 4)

    def test_diagonal_vs_offdiagonal(self):
        """Test diagonal dominance analysis."""
        tm = TransferMatrix(supervision="demo")
        assert np.mean(tm.get_diagonal()) == pytest.approx(0.0)


class TestRetentionEvaluator:
    """Tests for retention curve evaluation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        analysis = RetentionAnalysis(run_id="r1")
        assert analysis.get_mean_retention() == 1.0

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        analysis = RetentionAnalysis(run_id="r1")
        analysis.add_checkpoint(RetentionCheckpoint(checkpoint_step=0, quadrant="Q1", success_rate=0.8, n_episodes=10))
        assert "Q1" in analysis.curves

    def test_catastrophic_forgetting_detection(self):
        """Test detection of catastrophic forgetting."""
        analysis = RetentionAnalysis(run_id="r1")
        for step, sr in [(0, 0.8), (1, 0.75), (2, 0.6), (3, 0.5), (4, 0.45)]:
            analysis.add_checkpoint(RetentionCheckpoint(checkpoint_step=step, quadrant="Q1", success_rate=sr, n_episodes=10))
        forgetting = detect_catastrophic_forgetting(analysis, threshold=0.1)
        assert forgetting["has_catastrophic_forgetting"] is True

    def test_no_forgetting_when_stable(self):
        """Test no false positive when performance stable."""
        analysis = RetentionAnalysis(run_id="r1")
        for step, sr in [(0, 0.8), (1, 0.79), (2, 0.81), (3, 0.8), (4, 0.82)]:
            analysis.add_checkpoint(RetentionCheckpoint(checkpoint_step=step, quadrant="Q1", success_rate=sr, n_episodes=10))
        forgetting = detect_catastrophic_forgetting(analysis, threshold=0.1)
        assert forgetting["has_catastrophic_forgetting"] is False


class TestDriftEvaluator:
    """Tests for distribution drift evaluation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        baseline = PanelResult(
            panel_id="p",
            model_id="m0",
            evaluated_at="t",
            overall_success=0.8,
            by_quadrant={"Q1": 0.8},
            by_difficulty={"easy": 0.8},
            task_results=[],
        )
        current = PanelResult(
            panel_id="p",
            model_id="m1",
            evaluated_at="t",
            overall_success=0.7,
            by_quadrant={"Q1": 0.7},
            by_difficulty={"easy": 0.7},
            task_results=[],
        )
        drift = detect_drift(baseline, current, threshold=0.05)
        assert "has_drift" in drift

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        baseline = PanelResult(
            panel_id="p",
            model_id="m0",
            evaluated_at="t",
            overall_success=0.8,
            by_quadrant={"Q1": 0.8},
            by_difficulty={"easy": 0.8},
            task_results=[],
        )
        current = PanelResult(
            panel_id="p",
            model_id="m1",
            evaluated_at="t",
            overall_success=0.6,
            by_quadrant={"Q1": 0.6},
            by_difficulty={"easy": 0.6},
            task_results=[],
        )
        drift = detect_drift(baseline, current, threshold=0.05)
        assert drift["has_drift"] is True

    def test_variance_computation(self):
        """Test variance across runs computation."""
        vals = np.array([0.8, 0.82, 0.79])
        assert float(np.std(vals)) < 0.1

    def test_drift_detection(self):
        """Test drift from baseline detection."""
        baseline = PanelResult(
            panel_id="p",
            model_id="m0",
            evaluated_at="t",
            overall_success=0.8,
            by_quadrant={"Q1": 0.8},
            by_difficulty={"easy": 0.8},
            task_results=[],
        )
        current = PanelResult(
            panel_id="p",
            model_id="m1",
            evaluated_at="t",
            overall_success=0.6,
            by_quadrant={"Q1": 0.6},
            by_difficulty={"easy": 0.6},
            task_results=[],
        )
        drift = detect_drift(baseline, current, threshold=0.05)
        assert drift["has_drift"] is True
