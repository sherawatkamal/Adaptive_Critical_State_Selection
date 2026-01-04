"""
Simulation package for trajectory collection via student and teacher rollouts.

This package provides the foundation for discovering teachable moments by:
1. Running student models to find failures
2. Running teacher models for oracle demonstrations
3. Analyzing gaps between student and teacher performance

Modules:
- student_rollout: Student simulation and failure collection
- teacher_rollout: Teacher simulation for expert demonstrations
- failure_detector: Failure analysis and teachability scoring
"""

from .student_rollout import (
    StudentRollout,
    StudentRolloutConfig,
    RolloutResult,
    RolloutState,
    FailureEvent,
    FailureType,
)
from .teacher_rollout import (
    TeacherRollout,
    TeacherRolloutConfig,
    TeacherRolloutResult,
)
from .failure_detector import (
    FailureDetector,
    FailureDetectorConfig,
    FailurePattern,
    TeachableGap,
)

__all__ = [
    # Student rollout
    "StudentRollout",
    "StudentRolloutConfig",
    "RolloutResult",
    "RolloutState",
    "FailureEvent",
    "FailureType",
    # Teacher rollout
    "TeacherRollout",
    "TeacherRolloutConfig",
    "TeacherRolloutResult",
    # Failure detection
    "FailureDetector",
    "FailureDetectorConfig",
    "FailurePattern",
    "TeachableGap",
]
