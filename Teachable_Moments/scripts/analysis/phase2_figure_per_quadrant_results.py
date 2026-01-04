#!/usr/bin/env python3
"""Generate Figure 3: Per-Quadrant Results (Phase 2)."""

import matplotlib.pyplot as plt
import numpy as np
import json
from scripts.analysis.plotting_utils import setup_style, setup_logging, get_common_args

logger = setup_logging()

def generate_figure3_per_quadrant_results(results_path, output_path):
    """
    Generate Figure 3: Bar chart showing success rate by quadrant and supervision type.
    """
    if not results_path.exists():
         raise FileNotFoundError(f"Per-quadrant results not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)
    if not data:
        raise ValueError("Data file is empty.")

    # Organize data
    quadrants = ["Q1_high_high", "Q2_high_low", "Q3_low_high", "Q4_low_low"]
    supervision_types = ["demo", "contrast", "hint"]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(quadrants))
    width = 0.25
    
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    for i, sup_type in enumerate(supervision_types):
        values = []
        for quadrant in quadrants:
            key = f"{quadrant}_{sup_type}"
            result = data.get(key, {})
            # If key not present, assume 0 or handle appropriately. 
            # Fixpack implies no random generation, so we trust data exists.
            values.append(result.get("success_rate", 0))
        
        ax.bar(x + i * width, values, width, label=sup_type, color=colors[i])
    
    # Formatting
    ax.set_xlabel("Quadrant", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Per-Quadrant Training Results", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([q.replace("_", "\n") for q in quadrants])
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Generated per-quadrant results: {output_path}")

def main():
    parser = get_common_args("Generate Figure 3: Per-Quadrant Results")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_figure3_per_quadrant_results(
        args.results_dir / "phase2" / "per_quadrant_results.json",
        args.output_dir / "figure3_per_quadrant_results.pdf",
    )

if __name__ == "__main__":
    main()
