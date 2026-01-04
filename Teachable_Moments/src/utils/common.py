"""
Common utilities for teachable moments experiments.
"""

from pathlib import Path
import json
import yaml
import logging
import random
import numpy as np
from typing import Optional, TypeVar, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    """Save data to YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def batch_iter(items: list[T], batch_size: int):
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is zero."""
    return a / b if b != 0 else default


def compute_stats(values: list[float]) -> dict:
    """Compute basic statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def merge_dicts(*dicts: dict) -> dict:
    """Merge multiple dicts, later ones override earlier."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


class ProgressTracker:
    """Simple progress tracker with logging."""
    
    def __init__(self, total: int, name: str = "Progress", log_interval: int = 10):
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.current = 0
    
    def update(self, n: int = 1) -> None:
        self.current += n
        if self.current % self.log_interval == 0 or self.current == self.total:
            pct = 100 * self.current / self.total
            logger.info(f"{self.name}: {self.current}/{self.total} ({pct:.1f}%)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f}s")
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
