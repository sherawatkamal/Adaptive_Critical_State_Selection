#!/usr/bin/env python3
"""Generate Supervision Effectiveness Table (Phase 2)."""

import numpy as np
from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args, load_json
)

logger = setup_logging()

def generate_supervision_effectiveness_table(results_dir, output_dir):
    """Generate supervision effectiveness by quadrant table."""
    summary_path = results_dir / "phase2" / "complete_training_summary.json"
    data = load_json(summary_path)
    
    if "by_quadrant" not in data:
         raise ValueError("Missing 'by_quadrant' in summary data")
         
    by_quadrant = data["by_quadrant"] # Structure: {Q1: {by_supervision: {demo: {success_rate: 0.X}}}}
    
    sup_types = ["demonstration", "contrastive", "hint"]
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    
    # Latex
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Success rate by supervision type and quadrant. Best supervision for each quadrant in bold.}",
        "\\label{tab:supervision_effectiveness}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Quadrant & Demonstration & Contrastive & Hint \\\\",
        "\\midrule",
    ]
    
    for q in quadrants:
        q_data = by_quadrant.get(q, {}).get("by_supervision", {})
        values = []
        for s in sup_types:
            values.append(q_data.get(s, {}).get("success_rate", 0.0))
            
        max_idx = np.argmax(values) if values else -1
        formatted = []
        for i, v in enumerate(values):
            if i == max_idx:
                formatted.append(f"\\textbf{{{v:.2f}}}")
            else:
                formatted.append(f"{v:.2f}")
        latex_lines.append(f"{q} & {' & '.join(formatted)} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_path = output_dir / "supervision_effectiveness.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    # Markdown
    md_lines = [
        "| Quadrant | Demonstration | Contrastive | Hint |",
        "|----------|---------------|-------------|------|",
    ]
    
    for q in quadrants:
        q_data = by_quadrant.get(q, {}).get("by_supervision", {})
        values = []
        for s in sup_types:
            values.append(q_data.get(s, {}).get("success_rate", 0.0))
            
        max_idx = np.argmax(values) if values else -1
        formatted = []
        for i, v in enumerate(values):
            if i == max_idx:
                formatted.append(f"**{v:.2f}**")
            else:
                formatted.append(f"{v:.2f}")
        md_lines.append(f"| {q} | {' | '.join(formatted)} |")
    
    md_path = output_dir / "supervision_effectiveness.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Generated supervision effectiveness table: {latex_path}")

def main():
    parser = get_common_args("Generate Supervision Effectiveness Table")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_supervision_effectiveness_table(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
