#!/usr/bin/env python3
"""Generate Figure 2: CPT Validation (Phase 1b)."""

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import pearsonr
from scripts.analysis.plotting_utils import setup_style, setup_logging, get_common_args

logger = setup_logging()

def generate_figure2_cpt_validation(validation_results_path, output_path):
    """
    Generate Figure 2: Scatter plot of ELP_net vs actual improvement (delta_micro).
    Shows correlation between CPT prediction and actual fine-tuning improvement.
    """
    if not validation_results_path.exists():
        raise FileNotFoundError(f"Validation results not found: {validation_results_path}")

    with open(validation_results_path) as f:
        data = json.load(f)
        
    results = data.get("results", [])
    if not results:
        raise ValueError("No results found in validation results file.")
        
    elp_values = [r["elp_net"] for r in results]
    delta_values = [r["delta_micro"] for r in results]
    quadrants = [r["quadrant"] for r in results]
    
    # Compute correlation
    if len(elp_values) > 1:
        r, p = pearsonr(elp_values, delta_values)
    else:
        r, p = 0.0, 1.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color by quadrant
    quadrant_colors = {
        "Q1_high_high": "#2ecc71",
        "Q2_high_low": "#e74c3c",
        "Q3_low_high": "#3498db",
        "Q4_low_low": "#95a5a6",
    }
    
    for quadrant, color in quadrant_colors.items():
        mask = [q == quadrant for q in quadrants]
        x = [elp_values[i] for i, m in enumerate(mask) if m]
        y = [delta_values[i] for i, m in enumerate(mask) if m]
        if x:
            ax.scatter(x, y, c=color, label=quadrant, alpha=0.7, s=40)
    
    # Add regression line
    if len(elp_values) > 1:
        z = np.polyfit(elp_values, delta_values, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(elp_values), max(elp_values), 100)
        ax.plot(x_line, p_line(x_line), "k--", alpha=0.5, label=f"rho = {r:.3f}")
    
    # Labels
    ax.set_xlabel("ELP_net (CPT Prediction)", fontsize=12)
    ax.set_ylabel("Delta_micro (Actual Improvement)", fontsize=12)
    ax.set_title(f"CPT Validation (rho = {r:.3f}, p = {p:.4f})", fontsize=14)
    ax.legend(loc="best")
    
    # Add reference lines
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Generated CPT validation plot: {output_path}")

def main():
    parser = get_common_args("Generate Figure 2: CPT Validation")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_figure2_cpt_validation(
        args.results_dir / "phase1b" / "cpt_validation_results.json",
        args.output_dir / "figure2_cpt_validation.pdf",
    )

if __name__ == "__main__":
    main()
