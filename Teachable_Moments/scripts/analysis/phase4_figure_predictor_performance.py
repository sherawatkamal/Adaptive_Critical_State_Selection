#!/usr/bin/env python3
"""Generate Predictor Performance (Phase 4)."""

import matplotlib.pyplot as plt
import numpy as np
import logging
from scripts.analysis.plotting_utils import (
    setup_style, setup_logging, get_common_args, load_json, QUADRANT_COLORS
)

logger = setup_logging()

def generate_predictor_performance(results_dir, output_dir):
    """Generate predictor performance visualizations."""
    pred_path = results_dir / "phase4" / "predictor_evaluation.json"
    
    # Strictly remove placeholder logic.
    data = load_json(pred_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Regression task R2 scores
    reg_tasks = ["uncertainty", "leverage", "elp"]
    r2_scores = [data.get(t, {}).get("r2", 0.0) for t in reg_tasks]
    correlations = [data.get(t, {}).get("correlation", 0.0) for t in reg_tasks]
    
    x = np.arange(len(reg_tasks))
    width = 0.35
    
    axes[0].bar(x - width / 2, r2_scores, width, label="R2", color="#3498db")
    axes[0].bar(x + width / 2, correlations, width, label="Correlation", color="#2ecc71")
    
    axes[0].set_ylabel("Score")
    axes[0].set_title("Regression Task Performance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([t.capitalize() for t in reg_tasks])
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Quadrant classification per-class accuracy
    quad_data = data.get("quadrant", {})
    per_class = quad_data.get("per_class", {})
    
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    accuracies = [per_class.get(q, 0.0) for q in quadrants]
    colors = [QUADRANT_COLORS.get(q, "grey") for q in quadrants]
    
    axes[1].bar(quadrants, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axhline(
        y=quad_data.get("accuracy", 0.0),
        color="red",
        linestyle="--",
        label=f"Overall: {quad_data.get('accuracy', 0.0):.2f}",
    )
    
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Quadrant Classification by Class")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / "predictor_performance.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Generated predictor performance: {output_path}")

def main():
    parser = get_common_args("Generate Predictor Performance")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_predictor_performance(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
