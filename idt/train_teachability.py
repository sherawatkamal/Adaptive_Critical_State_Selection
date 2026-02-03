"""
Baseline teachability predictor: feature extraction + logreg/MLP.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_features(record: Dict[str, Any], traj: Optional[Any] = None) -> np.ndarray:
    """
    Features from StepContext / record:
    - step index (normalized by max_steps e.g. 50)
    - action type one-hot (replace/insert/edit_query)
    - lengths: instruction, observation, history
    - optional: agent entropy, margin, top1-top2 gap (if in record).
    """
    feats: List[float] = []
    step_t = record.get("step_t") or 0
    feats.append(step_t / max(1, 50))
    patch_type = record.get("patch_type") or "replace"
    for t in ("replace", "insert", "edit_query"):
        feats.append(1.0 if t == patch_type else 0.0)
    # Dummy lengths if no traj
    inst_len = len(record.get("instruction", ""))
    obs_len = len(record.get("observation", ""))
    hist_len = record.get("history_len", 0)
    feats.append(min(inst_len / 500.0, 1.0))
    feats.append(min(obs_len / 2000.0, 1.0))
    feats.append(min(hist_len / 50.0, 1.0))
    if "entropy" in record:
        feats.append(float(record["entropy"]))
    else:
        feats.append(0.0)
    if "margin" in record:
        feats.append(float(record["margin"]))
    else:
        feats.append(0.0)
    return np.array(feats, dtype=np.float64)


def load_records_with_features(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset JSONL and return (X, y) for teachable_label."""
    records = []
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    X = np.stack([extract_features(r) for r in records]) if records else np.zeros((0, 9))
    y = np.array([1 if r.get("teachable_label") else 0 for r in records], dtype=np.float64)
    return X, y


def train_baseline(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Any:
    """Train logistic regression or MLP; return fitted model and scaler."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn required for train_teachability")
    if len(np.unique(y)) < 2:
        logger.warning("Only one class in labels; using DummyClassifier (constant predictor)")
        from sklearn.dummy import DummyClassifier
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
        model.fit(X_s, y)
        return model, scaler
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    if model_type == "logreg":
        model = LogisticRegression(max_iter=500, random_state=random_state)
    else:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=random_state)
    model.fit(X_train_s, y_train)
    acc = model.score(X_val_s, y_val)
    logger.info("Val accuracy: %.4f", acc)
    return model, scaler


def save_model(model: Any, scaler: Any, path: str | Path) -> None:
    """Save model and scaler (sklearn joblib or pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"model": model, "scaler": scaler}, path)
    except ImportError:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)


def load_model(path: str | Path) -> Tuple[Any, Any]:
    """Load model and scaler."""
    path = Path(path)
    try:
        import joblib
        d = joblib.load(path)
    except ImportError:
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
    return d["model"], d["scaler"]
