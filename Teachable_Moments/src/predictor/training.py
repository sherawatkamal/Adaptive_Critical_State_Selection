"""
Training for teachability predictor.

Handles data preparation, training loop, and model selection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for predictor training."""
    
    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Data
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Loss weights
    uncertainty_weight: float = 1.0
    leverage_weight: float = 1.0
    quadrant_weight: float = 1.0
    elp_weight: float = 1.0
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Logging
    log_interval: int = 10
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("predictor_training", {}))


@dataclass
class TrainingResult:
    """Result of predictor training."""
    
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    task_metrics: dict[str, dict]
    
    def to_dict(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "task_metrics": self.task_metrics,
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def prepare_labels(
    snapshots: list[dict],
    quadrant_labels: list[str],
) -> dict[str, np.ndarray]:
    """
    Extract labels from snapshots.
    
    Args:
        snapshots: List of labeled snapshots
        quadrant_labels: List of quadrant label names
        
    Returns:
        Dict of label arrays
    """
    n = len(snapshots)
    
    labels = {
        "uncertainty": np.zeros(n, dtype=np.float32),
        "leverage": np.zeros(n, dtype=np.float32),
        "elp": np.zeros(n, dtype=np.float32),
        "quadrant": np.zeros(n, dtype=np.int64),
    }
    
    for i, snap in enumerate(snapshots):
        # Uncertainty (entropy)
        # Check uncertainty_features first (new schema), then uncertainty (old schema)
        uncertainty = snap.get("uncertainty_features") or snap.get("uncertainty") or {}
        # Also handle if flat in snapshot (e.g. "U")
        if "U" in snap:
            labels["uncertainty"][i] = float(snap["U"])
        else:
            labels["uncertainty"][i] = float(uncertainty.get("entropy", 0.0))
        
        # Leverage (L_local)
        leverage = snap.get("leverage") or {}
        # Handle if leverage is object or dict
        if hasattr(leverage, "L_local"):
            labels["leverage"][i] = float(leverage.L_local)
        else:
            labels["leverage"][i] = float(leverage.get("L_local", 0.0))
        
        # ELP
        cpt = snap.get("cpt") or {}
        if hasattr(cpt, "ELP_net"):
             labels["elp"][i] = float(cpt.ELP_net)
        else:
             labels["elp"][i] = float(cpt.get("ELP_net", 0.0))
        
        # Quadrant
        quadrant = snap.get("quadrant", "Q1_highU_highL")
        labels["quadrant"][i] = quadrant_labels.index(quadrant) if quadrant in quadrant_labels else 0
    
    return labels


def train_predictor(
    predictor,
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    config: Optional[TrainingConfig] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> TrainingResult:
    """
    Train teachability predictor.
    
    Args:
        predictor: TeachabilityPredictor instance
        features: (N, feature_dim) array
        labels: Dict of label arrays
        config: Training configuration
        progress_callback: Called with (epoch, total_epochs, loss)
        
    Returns:
        TrainingResult with training history
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split
    
    if config is None:
        config = TrainingConfig()
    
    n = features.shape[0]
    feature_dim = features.shape[1]
    
    # Build model
    predictor._build_model(feature_dim)
    
    # Prepare data
    X = torch.tensor(features, dtype=torch.float32)
    
    y_uncertainty = torch.tensor(labels["uncertainty"], dtype=torch.float32).unsqueeze(1)
    y_leverage = torch.tensor(labels["leverage"], dtype=torch.float32).unsqueeze(1)
    y_elp = torch.tensor(labels["elp"], dtype=torch.float32).unsqueeze(1)
    y_quadrant = torch.tensor(labels["quadrant"], dtype=torch.long)
    
    dataset = TensorDataset(X, y_uncertainty, y_leverage, y_elp, y_quadrant)
    
    # Split data
    val_size = int(n * config.val_split)
    test_size = int(n * config.test_split)
    train_size = n - val_size - test_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        predictor.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training
        predictor.model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            x, y_u, y_l, y_e, y_q = batch
            
            optimizer.zero_grad()
            
            trunk_out = predictor.model["trunk"](x)
            
            loss = 0.0
            
            if "uncertainty" in predictor.model["heads"]:
                pred_u = predictor.model["heads"]["uncertainty"](trunk_out)
                loss += config.uncertainty_weight * mse_loss(pred_u, y_u)
            
            if "leverage" in predictor.model["heads"]:
                pred_l = predictor.model["heads"]["leverage"](trunk_out)
                loss += config.leverage_weight * mse_loss(pred_l, y_l)
            
            if "elp" in predictor.model["heads"]:
                pred_e = predictor.model["heads"]["elp"](trunk_out)
                loss += config.elp_weight * mse_loss(pred_e, y_e)
            
            if "quadrant" in predictor.model["heads"]:
                pred_q = predictor.model["heads"]["quadrant"](trunk_out)
                loss += config.quadrant_weight * ce_loss(pred_q, y_q)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        predictor.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y_u, y_l, y_e, y_q = batch
                
                trunk_out = predictor.model["trunk"](x)
                
                loss = 0.0
                
                if "uncertainty" in predictor.model["heads"]:
                    pred_u = predictor.model["heads"]["uncertainty"](trunk_out)
                    loss += config.uncertainty_weight * mse_loss(pred_u, y_u)
                
                if "leverage" in predictor.model["heads"]:
                    pred_l = predictor.model["heads"]["leverage"](trunk_out)
                    loss += config.leverage_weight * mse_loss(pred_l, y_l)
                
                if "elp" in predictor.model["heads"]:
                    pred_e = predictor.model["heads"]["elp"](trunk_out)
                    loss += config.elp_weight * mse_loss(pred_e, y_e)
                
                if "quadrant" in predictor.model["heads"]:
                    pred_q = predictor.model["heads"]["quadrant"](trunk_out)
                    loss += config.quadrant_weight * ce_loss(pred_q, y_q)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - config.min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Logging
        if epoch % config.log_interval == 0:
            logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        if progress_callback:
            progress_callback(epoch, config.epochs, avg_val_loss)
    
    predictor._is_trained = True
    
    # Compute task-specific metrics on validation set
    task_metrics = evaluate_predictor(predictor, val_loader)
    
    return TrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        task_metrics=task_metrics,
    )


def evaluate_predictor(predictor, data_loader) -> dict:
    """
    Evaluate predictor on a dataset.
    
    Args:
        predictor: Trained predictor
        data_loader: DataLoader with test data
        
    Returns:
        Dict of task-specific metrics
    """
    import torch
    
    predictor.model.eval()
    
    all_preds = {"uncertainty": [], "leverage": [], "elp": [], "quadrant": []}
    all_labels = {"uncertainty": [], "leverage": [], "elp": [], "quadrant": []}
    
    with torch.no_grad():
        for batch in data_loader:
            x, y_u, y_l, y_e, y_q = batch
            
            trunk_out = predictor.model["trunk"](x)
            
            if "uncertainty" in predictor.model["heads"]:
                pred_u = predictor.model["heads"]["uncertainty"](trunk_out)
                all_preds["uncertainty"].extend(pred_u.squeeze().tolist())
                all_labels["uncertainty"].extend(y_u.squeeze().tolist())
            
            if "leverage" in predictor.model["heads"]:
                pred_l = predictor.model["heads"]["leverage"](trunk_out)
                all_preds["leverage"].extend(pred_l.view(-1).tolist())
                all_labels["leverage"].extend(y_l.view(-1).tolist())
            
            if "elp" in predictor.model["heads"]:
                pred_e = predictor.model["heads"]["elp"](trunk_out)
                all_preds["elp"].extend(pred_e.view(-1).tolist())
                all_labels["elp"].extend(y_e.view(-1).tolist())
            
            if "quadrant" in predictor.model["heads"]:
                pred_q = predictor.model["heads"]["quadrant"](trunk_out)
                all_preds["quadrant"].extend(torch.argmax(pred_q, dim=1).view(-1).tolist())
                all_labels["quadrant"].extend(y_q.view(-1).tolist())
    
    metrics = {}
    
    # Regression metrics
    for task in ["uncertainty", "leverage", "elp"]:
        if all_preds[task]:
            preds = np.array(all_preds[task])
            labels = np.array(all_labels[task])
            
            mse = float(np.mean((preds - labels) ** 2))
            mae = float(np.mean(np.abs(preds - labels)))
            
            # Correlation
            if np.std(preds) > 0 and np.std(labels) > 0:
                corr = float(np.corrcoef(preds, labels)[0, 1])
            else:
                corr = 0.0
            
            metrics[task] = {"mse": mse, "mae": mae, "correlation": corr}
            # Ranking-centric metrics for ELP: how good are we at selecting teachable moments?
            if task == "elp":
                try:
                    from .evaluation import compute_elp_ranking_metrics
                    rank_m = compute_elp_ranking_metrics(labels, preds, ks=(10, 50))
                    metrics[task].update({
                        "spearman": rank_m.get("spearman", 0.0),
                        "precision_pos_at_10": rank_m.get("precision_pos_at_10", 0.0),
                        "precision_pos_at_50": rank_m.get("precision_pos_at_50", 0.0),
                        "topk_overlap_at_10": rank_m.get("topk_overlap_at_10", 0.0),
                        "topk_overlap_at_50": rank_m.get("topk_overlap_at_50", 0.0),
                        "ndcg_at_10": rank_m.get("ndcg_at_10", 0.0),
                        "ndcg_at_50": rank_m.get("ndcg_at_50", 0.0),
                    })
                except Exception:
                    pass
    
    # Classification metrics
    if all_preds["quadrant"]:
        preds = np.array(all_preds["quadrant"])
        labels = np.array(all_labels["quadrant"])
        
        accuracy = float(np.mean(preds == labels))
        metrics["quadrant"] = {"accuracy": accuracy}
    
    return metrics


def run_predictor_training(
    snapshots_path: str,
    features_path: str,
    output_dir: str,
    config: Optional[TrainingConfig] = None,
) -> dict:
    """
    Run full predictor training pipeline.
    
    Args:
        snapshots_path: Path to labeled snapshots JSON
        features_path: Path to precomputed features
        output_dir: Output directory
        config: Training configuration
        
    Returns:
        Training results summary
    """
    from .multitask_model import TeachabilityPredictor, PredictorConfig
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(snapshots_path) as f:
        snapshots_data = json.load(f)
    snapshots = snapshots_data.get("snapshots", snapshots_data)
    
    features = np.load(features_path)
    
    # Prepare labels
    quadrant_labels = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
    labels = prepare_labels(snapshots, quadrant_labels)
    
    # Create predictor
    predictor = TeachabilityPredictor()
    
    # Train
    result = train_predictor(predictor, features, labels, config)
    
    # Save
    predictor.save(str(output_path / "predictor"))
    result.save(str(output_path / "training_result.json"))
    
    return {
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "task_metrics": result.task_metrics,
        "model_path": str(output_path / "predictor"),
    }
