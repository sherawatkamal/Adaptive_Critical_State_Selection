#!/usr/bin/env python3
"""Generate Quadrant Distribution (Phase 1)."""

import matplotlib.pyplot as plt
import numpy as np
from scripts.analysis.plotting_utils import (
    setup_style, setup_logging, get_common_args, load_json, 
    QUADRANT_LABELS, QUADRANT_COLORS
)

logger = setup_logging()

def generate_quadrant_distribution(results_dir, output_dir):
    """Generate quadrant distribution visualization."""
    labels_path = results_dir / "phase1" / "label_analysis.json"
    
    data = load_json(labels_path)
    counts = data.get("quadrant_counts", {})
    
    if not counts:
        raise ValueError("No quadrant counts found in label analysis.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart
    quadrants = list(QUADRANT_COLORS.keys())
    # Filter to only Q1-Q4 if present in keys
    quadrants = [q for q in quadrants if len(q) == 2] 
    
    values = [counts.get(q, 0) for q in quadrants]
    colors = [QUADRANT_COLORS[q] for q in quadrants]
    labels = [QUADRANT_LABELS[q] for q in quadrants]
    
    axes[0].bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Quadrant")
    axes[0].set_ylabel("Number of Trajectories")
    axes[0].set_title("Quadrant Distribution")
    axes[0].tick_params(axis="x", rotation=45)
    
    # 2D scatter showing uncertainty vs leverage
    quadrant_path = results_dir / "phase1" / "quadrant_assignments.json"
    
    if not quadrant_path.exists():
        raise FileNotFoundError(f"Missing required file: {quadrant_path}")

    qdata = load_json(quadrant_path)
    assignments = qdata.get("assignments", [])
    
    for quadrant in quadrants:
        color = QUADRANT_COLORS[quadrant]
        q_points = [a for a in assignments if a.get("quadrant") == quadrant]
        if q_points:
            u_vals = [p.get("uncertainty", 0.5) for p in q_points[:100]]
            l_vals = [p.get("leverage", 0.5) for p in q_points[:100]]
            axes[1].scatter(
                u_vals,
                l_vals,
                c=color,
                label=QUADRANT_LABELS[quadrant],
                alpha=0.6,
                s=30,
            )
            
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Uncertainty")
    axes[1].set_ylabel("Leverage")
    axes[1].set_title("Uncertainty-Leverage Space")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / "quadrant_distribution.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Generated quadrant distribution: {output_path}")

def main():
    parser = get_common_args("Generate Quadrant Distribution")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_quadrant_distribution(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
