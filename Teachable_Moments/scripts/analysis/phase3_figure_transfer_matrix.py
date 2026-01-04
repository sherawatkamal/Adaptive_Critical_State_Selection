#!/usr/bin/env python3
"""Generate Transfer Matrix (Phase 3)."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scripts.analysis.plotting_utils import (
    setup_style, setup_logging, get_common_args, load_json, QUADRANT_LABELS
)

logger = setup_logging()

def generate_transfer_matrix(results_dir, output_dir):
    """Generate transfer matrix heatmap."""
    transfer_path = results_dir / "phase3" / "transfer_evaluation.json"
    
    data = load_json(transfer_path)
    matrix_data = data.get("transfer_matrix", {})
    if not matrix_data:
        raise ValueError("No transfer matrix data found.")
    
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    matrix = np.zeros((4, 4))
    
    for i, train_q in enumerate(quadrants):
        for j, eval_q in enumerate(quadrants):
            key = f"{train_q}_to_{eval_q}"
            if key in matrix_data:
                matrix[i, j] = matrix_data[key].get("success_rate", 0.0)
            else:
                # Should we raise here? Or just 0?
                # Usually missing data in matrix means 0 or uncomputed.
                matrix[i, j] = 0.0
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    quadrant_display = [QUADRANT_LABELS[q] for q in quadrants]
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=quadrant_display,
        yticklabels=quadrant_display,
        ax=ax,
        cbar_kws={"label": "Success Rate"},
    )
    
    ax.set_xlabel("Evaluation Quadrant")
    ax.set_ylabel("Training Quadrant")
    ax.set_title("Transfer Performance Matrix")
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    output_path = output_dir / "transfer_matrix.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Generated transfer matrix: {output_path}")

def main():
    parser = get_common_args("Generate Transfer Matrix")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_transfer_matrix(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
