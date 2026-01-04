"""
Multitask teachability predictor model.

Predicts teachability metrics from features, enabling cheap
identification of teachable moments without full labeling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PredictorConfig:
    """Configuration for teachability predictor."""
    
    # Architecture
    hidden_dims: list[int] = None
    dropout: float = 0.1
    
    # Tasks
    predict_uncertainty: bool = True
    predict_leverage: bool = True
    predict_quadrant: bool = True
    predict_elp: bool = True
    
    # Feature configuration
    use_structural: bool = True
    use_embeddings: bool = True
    embedding_dim: int = 384
    structural_dim: int = 12
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]
    
    @classmethod
    def from_yaml(cls, path: str) -> "PredictorConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("predictor", {}))
    
    def to_dict(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "predict_uncertainty": self.predict_uncertainty,
            "predict_leverage": self.predict_leverage,
            "predict_quadrant": self.predict_quadrant,
            "predict_elp": self.predict_elp,
            "use_structural": self.use_structural,
            "use_embeddings": self.use_embeddings,
            "embedding_dim": self.embedding_dim,
            "structural_dim": self.structural_dim,
        }


@dataclass
class PredictionResult:
    """Prediction output from teachability predictor."""
    
    # Regression outputs
    uncertainty: Optional[float] = None
    leverage: Optional[float] = None
    elp: Optional[float] = None
    
    # Classification outputs
    quadrant: Optional[str] = None
    quadrant_probs: Optional[dict[str, float]] = None
    
    # Confidence
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "uncertainty": self.uncertainty,
            "leverage": self.leverage,
            "elp": self.elp,
            "quadrant": self.quadrant,
            "quadrant_probs": self.quadrant_probs,
            "confidence": self.confidence,
        }


class TeachabilityPredictor:
    """
    Multitask predictor for teachability metrics.
    
    Predicts uncertainty, leverage, quadrant, and ELP from features.
    """
    
    QUADRANT_LABELS = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        self.config = config or PredictorConfig()
        self.model = None
        self._is_trained = False
    
    def _build_model(self, input_dim: int):
        """Build the neural network model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required for teachability predictor")
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
            ])
            prev_dim = hidden_dim
        
        # Shared trunk
        self.trunk = nn.Sequential(*layers)
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        
        if self.config.predict_uncertainty:
            self.heads["uncertainty"] = nn.Linear(prev_dim, 1)
        
        if self.config.predict_leverage:
            self.heads["leverage"] = nn.Linear(prev_dim, 1)
        
        if self.config.predict_elp:
            self.heads["elp"] = nn.Linear(prev_dim, 1)
        
        if self.config.predict_quadrant:
            self.heads["quadrant"] = nn.Linear(prev_dim, 4)
        
        self.model = nn.ModuleDict({
            "trunk": self.trunk,
            "heads": self.heads,
        })
        
        return self.model
    
    def _prepare_features(
        self,
        structural: Optional[np.ndarray],
        embeddings: Optional[np.ndarray],
    ) -> np.ndarray:
        """Concatenate feature types."""
        features = []
        
        if self.config.use_structural and structural is not None:
            features.append(structural)
        
        if self.config.use_embeddings and embeddings is not None:
            features.append(embeddings)
        
        if not features:
            raise ValueError("No features provided")
        
        return np.concatenate(features, axis=-1)
    
    def predict(
        self,
        structural: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """
        Predict teachability metrics from features.
        
        Args:
            structural: Structural feature vector
            embeddings: Embedding feature vector
            
        Returns:
            PredictionResult with predictions
        """
        import torch
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        features = self._prepare_features(structural, embeddings)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            trunk_out = self.model["trunk"](x)
            
            result = PredictionResult()
            
            if "uncertainty" in self.model["heads"]:
                result.uncertainty = float(self.model["heads"]["uncertainty"](trunk_out).squeeze())
            
            if "leverage" in self.model["heads"]:
                result.leverage = float(self.model["heads"]["leverage"](trunk_out).squeeze())
            
            if "elp" in self.model["heads"]:
                result.elp = float(self.model["heads"]["elp"](trunk_out).squeeze())
            
            if "quadrant" in self.model["heads"]:
                logits = self.model["heads"]["quadrant"](trunk_out).squeeze()
                probs = torch.softmax(logits, dim=-1)
                
                result.quadrant_probs = {
                    label: float(probs[i])
                    for i, label in enumerate(self.QUADRANT_LABELS)
                }
                result.quadrant = self.QUADRANT_LABELS[int(torch.argmax(probs))]
                result.confidence = float(torch.max(probs))
        
        return result
    
    def predict_batch(
        self,
        structural: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[PredictionResult]:
        """
        Predict for a batch of examples.
        
        Args:
            structural: (N, structural_dim) array
            embeddings: (N, embedding_dim) array
            
        Returns:
            List of PredictionResult
        """
        import torch
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        features = self._prepare_features(structural, embeddings)
        x = torch.tensor(features, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            trunk_out = self.model["trunk"](x)
            
            results = []
            n = x.shape[0]
            
            for i in range(n):
                result = PredictionResult()
                
                if "uncertainty" in self.model["heads"]:
                    result.uncertainty = float(self.model["heads"]["uncertainty"](trunk_out[i:i+1]).squeeze())
                
                if "leverage" in self.model["heads"]:
                    result.leverage = float(self.model["heads"]["leverage"](trunk_out[i:i+1]).squeeze())
                
                if "elp" in self.model["heads"]:
                    result.elp = float(self.model["heads"]["elp"](trunk_out[i:i+1]).squeeze())
                
                if "quadrant" in self.model["heads"]:
                    logits = self.model["heads"]["quadrant"](trunk_out[i:i+1]).squeeze()
                    probs = torch.softmax(logits, dim=-1)
                    
                    result.quadrant_probs = {
                        label: float(probs[j])
                        for j, label in enumerate(self.QUADRANT_LABELS)
                    }
                    result.quadrant = self.QUADRANT_LABELS[int(torch.argmax(probs))]
                    result.confidence = float(torch.max(probs))
                
                results.append(result)
        
        return results
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import torch
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save model weights
        if self.model is not None:
            torch.save(self.model.state_dict(), path / "model.pt")
        
        logger.info(f"Saved predictor to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        import torch
        
        path = Path(path)
        
        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        self.config = PredictorConfig(**config_dict)
        
        # Compute input dim
        input_dim = 0
        if self.config.use_structural:
            input_dim += self.config.structural_dim
        if self.config.use_embeddings:
            input_dim += self.config.embedding_dim
        
        # Build and load model
        self._build_model(input_dim)
        self.model.load_state_dict(torch.load(path / "model.pt"))
        self._is_trained = True
        
        logger.info(f"Loaded predictor from {path}")


class MockPredictor:
    """Mock predictor for testing."""
    
    QUADRANT_LABELS = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
    
    def predict(
        self,
        structural: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        return PredictionResult(
            uncertainty=0.5,
            leverage=0.3,
            elp=0.4,
            quadrant="Q1_highU_highL",
            quadrant_probs={q: 0.25 for q in self.QUADRANT_LABELS},
            confidence=0.5,
        )
    
    def predict_batch(
        self,
        structural: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[PredictionResult]:
        n = structural.shape[0] if structural is not None else embeddings.shape[0]
        return [self.predict(structural, embeddings) for _ in range(n)]
