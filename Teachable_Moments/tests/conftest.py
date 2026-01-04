"""Pytest configuration and shared fixtures for teachable-moments tests."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_trajectory() -> dict[str, Any]:
    """Create a sample trajectory for testing."""
    return {
        "id": "test_traj_001",
        "mission": "go to the red ball",
        "observations": [
            {"image": np.random.rand(7, 7, 3).tolist(), "direction": 0}
            for _ in range(10)
        ],
        "actions": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "rewards": [0.0] * 9 + [1.0],
        "success": True,
    }


@pytest.fixture
def sample_trajectories(sample_trajectory: dict) -> list[dict[str, Any]]:
    """Create multiple sample trajectories."""
    trajectories = []
    for i in range(20):
        traj = sample_trajectory.copy()
        traj["id"] = f"test_traj_{i:03d}"
        traj["success"] = i % 4 != 0  # 75% success rate
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def sample_uncertainty_scores() -> dict[str, float]:
    """Create sample uncertainty scores."""
    return {f"test_traj_{i:03d}": np.random.rand() for i in range(20)}


@pytest.fixture
def sample_leverage_scores() -> dict[str, float]:
    """Create sample leverage scores."""
    return {f"test_traj_{i:03d}": np.random.rand() for i in range(20)}


@pytest.fixture
def sample_quadrant_assignments() -> list[dict[str, Any]]:
    """Create sample quadrant assignments."""
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    return [
        {
            "trajectory_id": f"test_traj_{i:03d}",
            "quadrant": quadrants[i % 4],
            "uncertainty": np.random.rand(),
            "leverage": np.random.rand(),
        }
        for i in range(20)
    ]


@pytest.fixture
def trajectories_file(temp_dir: Path, sample_trajectories: list) -> Path:
    """Create a temporary trajectories file."""
    path = temp_dir / "trajectories.json"
    with open(path, "w") as f:
        json.dump(sample_trajectories, f)
    return path


@pytest.fixture
def quadrants_file(temp_dir: Path, sample_quadrant_assignments: list) -> Path:
    """Create a temporary quadrant assignments file."""
    path = temp_dir / "quadrant_assignments.json"
    with open(path, "w") as f:
        json.dump({"assignments": sample_quadrant_assignments}, f)
    return path


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Create a mock experiment configuration."""
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
        },
        "labeling": {
            "uncertainty": {
                "method": "mc_dropout",
                "n_samples": 5,
                "threshold": 0.5,
            },
            "leverage": {
                "method": "reward_delta",
                "horizon": 3,
                "threshold": 0.5,
            },
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "max_steps": 100,
        },
        "evaluation": {
            "n_episodes": 10,
            "max_steps": 32,
        },
    }


# Markers for slow tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark as integration test")
