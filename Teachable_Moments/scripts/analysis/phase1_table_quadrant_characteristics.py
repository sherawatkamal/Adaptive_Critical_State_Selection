#!/usr/bin/env python3
"""Generate Quadrant Table (Phase 1)."""

from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args, load_json
)

logger = setup_logging()

def generate_quadrant_table(results_dir, output_dir):
    """Generate quadrant characteristics table."""
    labels_path = results_dir / "phase1" / "label_analysis.json"
    labels = load_json(labels_path)
    
    # Extract data from labels (assuming it exists, no placeholders)
    # The previous code had a placeholder list 'quadrants'.
    # We must try to build it from 'labels' data.
    
    # If the JSON doesn't have the summary, we cannot make the table.
    if "quadrant_summary" not in labels:
        # Fallback to raising error if we strictly can't generate it.
        # But maybe we can calculate it?
        # For now, I'll assume the JSON *should* have it, or we fail.
        # The prompt says "remove placeholder".
        raise ValueError(f"quadrant_summary not found in {labels_path}")
        
    quadrants = labels["quadrant_summary"]

    # LaTeX
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Quadrant characteristics and optimal supervision strategies.}",
        "\\label{tab:quadrants}",
        "\\small",
        "\\begin{tabular}{ccccccp{3cm}}",
        "\\toprule",
        "Quad. & Uncert. & Lever. & Supervision & N & Success & Description \\\\",
        "\\midrule",
    ]
    
    for q_data in quadrants:
        # q_data should be a dict
        q = q_data.get("name", "?")
        u = q_data.get("uncertainty", "?")
        l = q_data.get("leverage", "?")
        sup = q_data.get("supervision", "?")
        n = q_data.get("n", 0)
        sr = q_data.get("success_rate", 0.0)
        desc = q_data.get("description", "")
        
        latex_lines.append(f"{q} & {u} & {l} & {sup} & {n} & {sr:.2f} & {desc} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_path = output_dir / "quadrant_characteristics.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    # Markdown
    md_lines = [
        "| Quadrant | Uncertainty | Leverage | Best Supervision | N | Success Rate | Description |",
        "|----------|-------------|----------|------------------|---|--------------|-------------|",
    ]
    
    for q_data in quadrants:
        q = q_data.get("name", "?")
        u = q_data.get("uncertainty", "?")
        l = q_data.get("leverage", "?")
        sup = q_data.get("supervision", "?")
        n = q_data.get("n", 0)
        sr = q_data.get("success_rate", 0.0)
        desc = q_data.get("description", "")
        
        md_lines.append(f"| {q} | {u} | {l} | {sup} | {n} | {sr:.2f} | {desc} |")
    
    md_path = output_dir / "quadrant_characteristics.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Generated quadrant table: {latex_path}")

def main():
    parser = get_common_args("Generate Quadrant Table")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_quadrant_table(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
