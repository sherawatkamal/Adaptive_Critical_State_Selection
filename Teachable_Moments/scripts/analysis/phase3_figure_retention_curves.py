#!/usr/bin/env python3
"""Generate Retention Curves (Phase 3)."""

import matplotlib.pyplot as plt
from scripts.analysis.plotting_utils import (
    setup_style, setup_logging, get_common_args, load_json, 
    QUADRANT_LABELS, QUADRANT_COLORS
)

logger = setup_logging()

def generate_retention_curves(results_dir, output_dir):
    """Generate retention curves showing performance over checkpoints."""
    retention_path = results_dir / "phase3" / "retention_evaluation.json"
    
    data = load_json(retention_path)
    retention_data = data.get("retention_curves", {})
    if not retention_data:
        raise ValueError("No retention curve data found.")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for quadrant, color in QUADRANT_COLORS.items():
        if quadrant in retention_data:
            qdata = retention_data[quadrant]
            checkpoints = qdata.get("checkpoints", [])
            success_rates = qdata.get("success_rates", [])
            
            ax.plot(
                checkpoints,
                success_rates,
                marker="o",
                color=color,
                label=QUADRANT_LABELS.get(quadrant, quadrant),
                linewidth=2,
                markersize=6,
            )
    
    ax.set_xlabel("Training Steps (Checkpoint)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Retention Curves by Quadrant")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    output_path = output_dir / "retention_curves.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Generated retention curves: {output_path}")

def main():
    parser = get_common_args("Generate Retention Curves")
    args = parser.parse_args()
    
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_retention_curves(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
