#!/usr/bin/env python3
"""Generate Transfer Matrix Table (Phase 3)."""

import numpy as np
from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args, load_json
)

logger = setup_logging()

def generate_transfer_table(results_dir, output_dir):
    """Generate transfer matrix table."""
    transfer_path = results_dir / "phase3" / "transfer_evaluation.json"
    data = load_json(transfer_path)
    
    matrix_data = data.get("transfer_matrix", {})
    if not matrix_data:
        raise ValueError("No transfer matrix data")

    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    matrix = np.zeros((4, 4))
    
    for i, train_q in enumerate(quadrants):
        for j, eval_q in enumerate(quadrants):
            key = f"{train_q}_to_{eval_q}"
            if key in matrix_data:
                matrix[i, j] = matrix_data[key].get("success_rate", 0.0)
    
    # LaTeX
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Transfer matrix: rows are training quadrants, columns are evaluation quadrants. Diagonal elements (bold) show in-distribution performance.}",
        "\\label{tab:transfer}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Train $\\backslash$ Eval & Q1 & Q2 & Q3 & Q4 \\\\",
        "\\midrule",
    ]
    
    for i, q in enumerate(quadrants):
        row_values = []
        for j in range(4):
            val = matrix[i, j]
            if i == j:
                row_values.append(f"\\textbf{{{val:.2f}}}")
            else:
                row_values.append(f"{val:.2f}")
        latex_lines.append(f"{q} & {' & '.join(row_values)} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_path = output_dir / "transfer_matrix.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    # Markdown
    md_lines = [
        "| Train \\ Eval | Q1 | Q2 | Q3 | Q4 |",
        "|--------------|-----|-----|-----|-----|",
    ]
    
    for i, q in enumerate(quadrants):
        row_values = [f"**{matrix[i,j]:.2f}**" if i == j else f"{matrix[i,j]:.2f}" for j in range(4)]
        md_lines.append(f"| {q} | {' | '.join(row_values)} |")
    
    md_path = output_dir / "transfer_matrix.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Generated transfer table: {latex_path}")

def main():
    parser = get_common_args("Generate Transfer Table")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_transfer_table(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
