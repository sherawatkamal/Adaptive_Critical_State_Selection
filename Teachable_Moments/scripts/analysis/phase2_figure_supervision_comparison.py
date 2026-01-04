#!/usr/bin/env python3
"""Generate Supervision Comparison (Phase 2)."""

import matplotlib.pyplot as plt
import numpy as np
from scripts.analysis.plotting_utils import (
    setup_style, setup_logging, get_common_args, load_json,
    SUPERVISION_COLORS
)

logger = setup_logging()

def generate_supervision_comparison(results_dir, output_dir):
    """Generate supervision type effectiveness comparison."""
    summary_path = results_dir / "phase2" / "complete_training_summary.json"
    
    data = load_json(summary_path)
    
    by_supervision = data.get("by_supervision", {})
    if not by_supervision:
         raise ValueError("No supervision data found.")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Overall comparison
    sup_types = list(SUPERVISION_COLORS.keys())
    # Filter only available types
    sup_types = [s for s in sup_types if s in by_supervision]
    
    if not sup_types:
        raise ValueError("None of the expected supervision types found in data.")

    means = [by_supervision.get(s, {}).get("mean_success_rate", 0.0) for s in sup_types]
    stds = [by_supervision.get(s, {}).get("std", 0.0) for s in sup_types]
    colors = [SUPERVISION_COLORS[s] for s in sup_types]
    
    bars = axes[0].bar(
        [s.capitalize() for s in sup_types],
        means,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=5,
    )
    
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Overall Supervision Effectiveness")
    axes[0].set_ylim(0, 1)
    
    # By quadrant comparison
    by_quadrant = data.get("by_quadrant", {})
    
    x = np.arange(4)
    width = 0.25
    
    # Check if we have quadrant data
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    for i, sup_type in enumerate(SUPERVISION_COLORS.keys()):
        if sup_type not in sup_types: continue
        
        rates = []
        for q in quadrants:
            q_data = by_quadrant.get(q, {}).get("by_supervision", {})
            rate = q_data.get(sup_type, {}).get("success_rate", 0.0)
            rates.append(rate)
        
        axes[1].bar(
            x + i * width,
            rates,
            width,
            label=sup_type.capitalize(),
            color=SUPERVISION_COLORS[sup_type],
            edgecolor="black",
            linewidth=0.5,
        )
    
    axes[1].set_xlabel("Quadrant")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Supervision by Quadrant")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(quadrants)
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / "supervision_comparison.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Generated supervision comparison: {output_path}")

def main():
    parser = get_common_args("Generate Supervision Comparison")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_supervision_comparison(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
