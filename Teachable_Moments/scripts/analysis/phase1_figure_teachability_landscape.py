#!/usr/bin/env python3
"""Generate Figure 1: Teachability Landscape (Phase 1)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scripts.analysis.plotting_utils import setup_style, setup_logging, get_common_args

logger = setup_logging()

def generate_figure1_teachability_landscape(snapshots_path, output_path):
    """
    Generate Figure 1: 2D scatter plot of uncertainty vs leverage.
    Shows all snapshots colored by quadrant with density contours.
    """
    # Load data
    if not snapshots_path.exists():
        raise FileNotFoundError(f"Snapshots not found: {snapshots_path}")
        
    df = pd.read_parquet(snapshots_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define quadrant colors
    quadrant_colors = {
        "Q1_high_high": "#2ecc71",    # Green - teachable
        "Q2_high_low": "#e74c3c",     # Red - lost causes
        "Q3_low_high": "#3498db",     # Blue - already capable
        "Q4_low_low": "#95a5a6",      # Gray - uninteresting
    }
    
    # Plot each quadrant
    for quadrant, color in quadrant_colors.items():
        mask = df["quadrant"] == quadrant
        ax.scatter(
            df.loc[mask, "uncertainty"],
            df.loc[mask, "leverage"],
            c=color,
            label=quadrant,
            alpha=0.6,
            s=20,
        )
    
    # Add quadrant boundaries (at median)
    u_median = df["uncertainty"].median()
    l_median = df["leverage"].median()
    ax.axvline(u_median, color="black", linestyle="--", alpha=0.5)
    ax.axhline(l_median, color="black", linestyle="--", alpha=0.5)
    
    # Labels and legend
    ax.set_xlabel("Uncertainty U(s)", fontsize=12)
    ax.set_ylabel("Leverage L(s)", fontsize=12)
    ax.set_title("Teachability Landscape", fontsize=14)
    ax.legend(loc="best")
    
    # Add quadrant annotations
    ax.annotate("Q1: Teachable", xy=(0.75, 0.75), xycoords="axes fraction", fontsize=10)
    ax.annotate("Q2: Lost Causes", xy=(0.75, 0.25), xycoords="axes fraction", fontsize=10)
    ax.annotate("Q3: Already Capable", xy=(0.25, 0.75), xycoords="axes fraction", fontsize=10)
    ax.annotate("Q4: Uninteresting", xy=(0.25, 0.25), xycoords="axes fraction", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Generated teachability landscape: {output_path}")

def main():
    parser = get_common_args("Generate Figure 1: Teachability Landscape")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_figure1_teachability_landscape(
        args.results_dir / "phase1" / "labeled_snapshots.parquet",
        args.output_dir / "figure1_teachability_landscape.pdf",
    )

if __name__ == "__main__":
    main()
