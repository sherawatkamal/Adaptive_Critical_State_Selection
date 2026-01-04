"""Shared utilities for analysis scripts.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

# Publication style settings
STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

QUADRANT_LABELS = {
    # Canonical quadrant semantics (per blueprint):
    #   Q1: high U, high L  ("teachable")
    #   Q2: high U, low L
    #   Q3: low U, low L
    #   Q4: low U, high L
    "Q1": "High U / High L",
    "Q2": "High U / Low L",
    "Q3": "Low U / Low L",
    "Q4": "Low U / High L",

    # v8 canonical detailed keys
    "Q1_highU_highL": "High U / High L",
    "Q2_highU_lowL": "High U / Low L",
    "Q3_lowU_lowL": "Low U / Low L",
    "Q4_lowU_highL": "Low U / High L",

    # Legacy keys (for backward compatibility only)
    "Q1_high_high": "High U / High L",
    "Q2_high_low": "High U / Low L",
    "Q3_low_low": "Low U / Low L",
    "Q4_low_high": "Low U / High L",
}

QUADRANT_COLORS = {
    # Prefer a stable palette across figures.
    "Q1": "#2ecc71",  # Green (teachable)
    "Q2": "#f39c12",  # Orange
    "Q3": "#95a5a6",  # Gray
    "Q4": "#3498db",  # Blue

    # v8 canonical detailed keys
    "Q1_highU_highL": "#2ecc71",
    "Q2_highU_lowL": "#f39c12",
    "Q3_lowU_lowL": "#95a5a6",
    "Q4_lowU_highL": "#3498db",

    # Legacy keys (for backward compatibility only)
    "Q1_high_high": "#2ecc71",
    "Q2_high_low": "#f39c12",
    "Q3_low_low": "#95a5a6",
    "Q4_low_high": "#3498db",
}

SUPERVISION_COLORS = {
    # v8 canonical
    "demo": "#2ecc71",
    "contrast": "#9b59b6",
    "hint": "#1abc9c",

    # Common aliases
    "demonstration": "#2ecc71",
    "contrastive": "#9b59b6",
    "baseline": "#7f8c8d",
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path) as f:
        return json.load(f)

def setup_style():
    """Apply publication style settings."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("colorblind")

def format_value(value: float, precision: int = 2, as_percent: bool = False) -> str:
    """Format a numeric value for tables."""
    if as_percent:
        return f"{value * 100:.{precision}f}\%"
    return f"{value:.{precision}f}"

def format_with_std(mean: float, std: float, precision: int = 2) -> str:
    """Format mean ± std."""
    return f"{mean:.{precision}f} $\pm$ {std:.{precision}f}"

def get_common_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory",
    )
    return parser
