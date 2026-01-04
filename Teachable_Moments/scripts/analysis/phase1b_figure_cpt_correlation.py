#!/usr/bin/env python3
"""Generate CPT Correlation (Phase 1b)."""

import matplotlib.pyplot as plt
import numpy as np
import json
from scripts.analysis.plotting_utils import setup_style, setup_logging, get_common_args, load_json

logger = setup_logging()

def generate_cpt_correlation(results_dir, output_dir):
    """Generate CPT-ELP correlation scatter plot."""
    corr_path = results_dir / "phase1b" / "cpt_correlation.json"
    
    data = load_json(corr_path)
    points = data.get("data_points", [])
    
    if not points:
        raise ValueError("No data points found in CPT correlation file.")
        
    cpt = np.array([p.get("cpt", 0.5) for p in points])
    elp = np.array([p.get("elp", 0.5) for p in points])
    correlation = data.get("pearson_r", 0.0)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(cpt, elp, alpha=0.6, s=40, c="#3498db", edgecolors="white", linewidth=0.5)
    
    # Regression line
    if len(cpt) > 1:
        z = np.polyfit(cpt, elp, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(cpt), max(cpt), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"r = {correlation:.2f}")
    
    ax.set_xlabel("Counterfactual Policy Transfer (CPT)")
    ax.set_ylabel("Empirical Learning Progress (ELP)")
    ax.set_title("CPT-ELP Correlation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Add diagonal reference
    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="Perfect correlation")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    output_path = output_dir / "cpt_correlation.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Generated CPT correlation plot: {output_path}")

def main():
    parser = get_common_args("Generate CPT Correlation")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_cpt_correlation(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
