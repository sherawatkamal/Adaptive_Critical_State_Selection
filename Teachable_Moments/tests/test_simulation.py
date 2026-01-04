"""Tests for simulation modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.simulation.student_rollout import (
    StudentRollout,
    StudentRolloutConfig,
    RolloutResult,
    FailureEvent,
    FailureType,
)
from src.simulation.teacher_rollout import (
    TeacherRollout,
    TeacherRolloutConfig,
    TeacherRolloutResult,
)
from src.simulation.failure_detector import (
    FailureDetector,
    FailureDetectorConfig,
    FailurePattern,
    TeachableGap,
)


class TestStudentRollout:
    """Tests for student rollout and failure collection."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = StudentRolloutConfig()
        assert config.max_steps == 30
        assert config.n_tasks == 100
        assert config.loop_threshold == 2

    def test_rollout_initialization(self):
        """Test rollout can be initialized."""
        config = StudentRolloutConfig(
            model_name="test_student",
            mock_env=True,
        )
        rollout = StudentRollout(config)
        assert rollout.config.model_name == "test_student"

    def test_loop_detection_repeating_single(self):
        """Test detection of single action repetition."""
        config = StudentRolloutConfig(
            loop_detection_window=6,
            loop_threshold=2,
        )
        rollout = StudentRollout(config)
        
        # Same action repeated
        actions = ["click[a]", "click[a]", "click[a]", "click[a]", "click[a]", "click[a]"]
        assert rollout._detect_loop(actions) is True

    def test_loop_detection_pattern(self):
        """Test detection of action pattern loop."""
        config = StudentRolloutConfig(
            loop_detection_window=6,
            loop_threshold=2,
        )
        rollout = StudentRollout(config)
        
        # A-B pattern repeated
        actions = ["click[a]", "click[b]", "click[a]", "click[b]", "click[a]", "click[b]"]
        assert rollout._detect_loop(actions) is True

    def test_no_loop_detection_varied(self):
        """Test no false positive for varied actions."""
        config = StudentRolloutConfig(
            loop_detection_window=6,
            loop_threshold=2,
        )
        rollout = StudentRollout(config)
        
        # Different actions
        actions = ["click[a]", "click[b]", "click[c]", "search[d]", "click[e]", "back"]
        assert rollout._detect_loop(actions) is False

    def test_failure_classification_stuck(self):
        """Test failure classification for stuck loop."""
        config = StudentRolloutConfig()
        rollout = StudentRollout(config)
        
        failure_type = rollout._classify_failure(
            success=False,
            n_steps=20,
            actions=["a", "b", "a", "b", "a", "b"],  # Loop pattern
            final_reward=0.0,
            confidences=[0.5, 0.5, 0.5],
        )
        
        assert failure_type == FailureType.STUCK_LOOP

    def test_failure_classification_timeout(self):
        """Test failure classification for timeout."""
        config = StudentRolloutConfig(max_steps=30)
        rollout = StudentRollout(config)
        
        failure_type = rollout._classify_failure(
            success=False,
            n_steps=30,  # Hit max steps
            actions=["a", "b", "c", "d", "e"],  # No loop
            final_reward=0.0,
            confidences=[0.6, 0.7, 0.5],
        )
        
        assert failure_type == FailureType.TIMEOUT

    def test_failure_classification_confusion(self):
        """Test failure classification for confusion."""
        config = StudentRolloutConfig(confidence_threshold=0.3)
        rollout = StudentRollout(config)
        
        failure_type = rollout._classify_failure(
            success=False,
            n_steps=10,
            actions=["a", "b", "c"],  # No loop
            final_reward=0.0,
            confidences=[0.1, 0.15, 0.2],  # Low confidence
        )
        
        assert failure_type == FailureType.CONFUSION

    def test_failure_classification_success(self):
        """Test no failure on success."""
        config = StudentRolloutConfig()
        rollout = StudentRollout(config)
        
        failure_type = rollout._classify_failure(
            success=True,
            n_steps=10,
            actions=["a", "b", "c"],
            final_reward=1.0,
            confidences=[0.8, 0.9, 0.85],
        )
        
        assert failure_type == FailureType.NONE


class TestTeacherRollout:
    """Tests for teacher rollout and demonstration collection."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TeacherRolloutConfig()
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.0

    def test_rollout_initialization(self):
        """Test rollout can be initialized."""
        config = TeacherRolloutConfig(
            model_name="gpt-4o-mini",
            mock_env=True,
        )
        rollout = TeacherRollout(config)
        assert rollout.config.model_name == "gpt-4o-mini"

    def test_action_parsing(self):
        """Test action parsing from teacher response."""
        config = TeacherRolloutConfig()
        rollout = TeacherRollout(config)
        
        valid_actions = ["search[laptop]", "click[Buy Now]", "back"]
        
        # Test exact match
        response = "I'll search for the product: search[laptop]"
        action = rollout._parse_action(response, valid_actions)
        assert action == "search[laptop]"
        
        # Test partial match
        response = "Click on Buy Now"
        action = rollout._parse_action(response, valid_actions)
        assert action == "click[Buy Now]"

    def test_compare_with_student(self):
        """Test comparison between teacher and student results."""
        config = TeacherRolloutConfig()
        rollout = TeacherRollout(config)
        
        teacher_results = [
            TeacherRolloutResult(
                trajectory_id="t1",
                task_id="task_001",
                success=True,
                total_reward=1.0,
                n_steps=10,
                states=[],
                actions=[],
                rewards=[],
            ),
            TeacherRolloutResult(
                trajectory_id="t2",
                task_id="task_002",
                success=True,
                total_reward=1.0,
                n_steps=8,
                states=[],
                actions=[],
                rewards=[],
            ),
        ]
        
        student_results = [
            {"task_id": "task_001", "success": False, "n_steps": 30},
            {"task_id": "task_002", "success": True, "n_steps": 15},
        ]
        
        comparison = rollout.compare_with_student(teacher_results, student_results)
        
        assert comparison["n_common_tasks"] == 2
        assert comparison["teachable_gaps"] == 1  # task_001
        assert comparison["both_succeed"] == 1    # task_002


class TestFailureDetector:
    """Tests for failure detection and analysis."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = FailureDetectorConfig()
        assert config.min_pattern_count == 3
        assert "stuck_loop" in config.intervention_rules

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        config = FailureDetectorConfig(min_pattern_count=5)
        detector = FailureDetector(config)
        assert detector.config.min_pattern_count == 5

    def test_pattern_detection(self):
        """Test detection of failure patterns."""
        config = FailureDetectorConfig(min_pattern_count=2)
        detector = FailureDetector(config)
        
        # Create multiple similar failures
        failures = [
            FailureEvent(
                trajectory_id=f"traj_{i}",
                step_idx=5,
                failure_type=FailureType.STUCK_LOOP,
                state="product page",
                action_taken="click[back]",
                valid_actions=[],
            )
            for i in range(5)
        ]
        
        patterns = detector._detect_patterns(failures)
        
        assert len(patterns) >= 1
        assert any(p.failure_type == FailureType.STUCK_LOOP for p in patterns)

    def test_teachability_scoring(self):
        """Test teachability score computation."""
        config = FailureDetectorConfig()
        detector = FailureDetector(config)
        
        # Create failures with varying characteristics
        failures = [
            FailureEvent(
                trajectory_id="traj_1",
                step_idx=3,  # Early failure
                failure_type=FailureType.WRONG_ACTION,
                state="search page",
                action_taken="click[wrong]",
                valid_actions=["click[right]"],
                model_confidence=0.2,  # Low confidence
            )
        ]
        
        score = detector._pattern_teachability(failures)
        
        # Should be somewhat teachable (low confidence = uncertain student)
        assert 0 < score < 1

    def test_intervention_recommendation(self):
        """Test intervention type recommendation."""
        config = FailureDetectorConfig()
        detector = FailureDetector(config)
        
        # Stuck loop should get contrast
        failure = FailureEvent(
            trajectory_id="traj_1",
            step_idx=10,
            failure_type=FailureType.STUCK_LOOP,
            state="stuck state",
            action_taken="loop action",
            valid_actions=[],
        )
        
        recommendation = detector._recommend_supervision(failure, [])
        
        assert recommendation == "contrast"

    def test_gap_analysis(self):
        """Test teachable gap identification."""
        config = FailureDetectorConfig()
        detector = FailureDetector(config)
        
        student_results = [
            RolloutResult(
                trajectory_id="s1",
                task_id="task_001",
                success=False,
                total_reward=0.0,
                n_steps=20,
                states=["s1", "s2"],
                actions=["wrong_action"],
                rewards=[0],
                failures=[
                    FailureEvent(
                        trajectory_id="s1",
                        step_idx=0,
                        failure_type=FailureType.WRONG_ACTION,
                        state="s1",
                        action_taken="wrong_action",
                        valid_actions=["right_action"],
                    )
                ],
            )
        ]
        
        teacher_results = [
            {
                "task_id": "task_001",
                "success": True,
                "n_steps": 5,
                "actions": ["right_action"],
                "reasoning": ["Go to goal"],
            }
        ]
        
        gaps = detector._find_teachable_gaps(student_results, teacher_results)
        
        assert len(gaps) == 1
        assert gaps[0].task_id == "task_001"
        assert gaps[0].action_mismatch is True


class TestFailureEvent:
    """Tests for FailureEvent data class."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        event = FailureEvent(
            trajectory_id="traj_001",
            step_idx=5,
            failure_type=FailureType.STUCK_LOOP,
            state="test state",
            action_taken="test action",
            valid_actions=["a", "b", "c"],
            model_confidence=0.3,
        )
        
        d = event.to_dict()
        
        assert d["trajectory_id"] == "traj_001"
        assert d["failure_type"] == "stuck_loop"
        assert d["model_confidence"] == 0.3


class TestRolloutResult:
    """Tests for RolloutResult data class."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = RolloutResult(
            trajectory_id="traj_001",
            task_id="task_001",
            success=False,
            total_reward=0.3,
            n_steps=15,
            states=["s1", "s2"],
            actions=["a1", "a2"],
            rewards=[0, 0.3],
            failures=[],
            duration_seconds=2.5,
            model_name="test_model",
        )
        
        d = result.to_dict()
        
        assert d["trajectory_id"] == "traj_001"
        assert d["success"] is False
        assert d["n_steps"] == 15
        assert d["model_name"] == "test_model"
