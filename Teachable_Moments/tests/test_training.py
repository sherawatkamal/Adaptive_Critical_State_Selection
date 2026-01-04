"""Tests for training modules."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.training.micro_trainer import MicroTrainingConfig, reset_lora_parameters
from src.training.per_quadrant import TrainingMatrix, create_training_matrix
from src.training.sft_trainer import SFTConfig


class TestSFTTrainer:
    """Tests for SFT trainer."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SFTConfig()
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.epochs > 0


class TestMicroTrainer:
    """Tests for micro-training (CPT validation)."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = MicroTrainingConfig()
        assert config.n_steps > 0
        assert config.learning_rate > 0

    def test_reset_lora_parameters_smoke(self):
        """Test LoRA reset runs on a minimal module."""
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class DummyLora(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
                self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = DummyLora()

        m = DummyModel()
        reset_lora_parameters(m)


class TestPerQuadrantOrchestrator:
    """Tests for per-quadrant training orchestration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        matrix = TrainingMatrix.create_default()
        assert len(matrix.runs) == 14

    def test_create_training_matrix_has_expected_size(self):
        m = create_training_matrix()
        assert isinstance(m, list)
        assert len(m) == 14
