#!/usr/bin/env python3
"""Generate Main Results Table (Summary)."""

from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args, load_json, format_with_std
)

logger = setup_logging()

def generate_main_results_table(results_dir, output_dir):
    """Generate main results comparison table."""
    # Load data
    summary_path = results_dir / "phase2" / "complete_training_summary.json"
    end2end_path = results_dir / "phase3" / "end2end_evaluation.json"
    
    summary = load_json(summary_path)
    end2end = load_json(end2end_path)
    
    # Previously hardcoded methods placeholders.
    # Now we must extract. 
    # Example structure expected:
    # end2end: {methods: [{'name': 'Random', 'success_rate': 0.58, ...}]}
    
    # Check if 'comparison_table' or similar exists.
    # Because we removed placeholders, we assume the JSONs provide the list.
    methods = []
    if "methods" in end2end:
         # Assume detailed list of dicts
         for m in end2end["methods"]:
             methods.append((
                 m.get("name", "Unknown"),
                 m.get("success_rate", 0.0),
                 m.get("success_std", 0.0),
                 m.get("steps", 0.0),
                 m.get("steps_std", 0.0)
             ))
    else:
        # If not structured, maybe we can't generate it.
        # But we shouldn't crash if the file exists but schema differs slightly?
        # The prompt strongly requested removing placeholder logic.
        # "remove any code path that generates random/fake placeholder data when files are missing"
        # Since files are NOT missing (we loaded them), we do our best.
        # If empty, table will be empty.
        pass

    if not methods:
         logger.warning("No method comparison data found in end2end_evaluation.json")
    
    # LaTeX table
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main Results: Comparison of training strategies on the BabyAI benchmark.}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Success Rate & Avg. Steps & Std. Dev. \\\\",
        "\\midrule",
    ]
    
    for method, sr, sr_std, steps, steps_std in methods:
        if "Adaptive" in method or "Ours" in method:
            latex_lines.append(
                f"\\textbf{{{method}}} & \\textbf{{{format_with_std(sr, sr_std)}}} & "
                f"\\textbf{{{format_with_std(steps, steps_std)}}} \\\\"
            )
        else:
            latex_lines.append(
                f"{method} & {format_with_std(sr, sr_std)} & {format_with_std(steps, steps_std)} \\\\"
            )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_path = output_dir / "main_results.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    # Markdown table
    md_lines = [
        "| Method | Success Rate | Avg. Steps |",
        "|--------|--------------|------------|",
    ]
    
    for method, sr, sr_std, steps, steps_std in methods:
        md_lines.append(f"| {method} | {sr:.2f} ± {sr_std:.2f} | {steps:.1f} ± {steps_std:.1f} |")
    
    md_path = output_dir / "main_results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Generated main results table: {latex_path}, {md_path}")

def main():
    parser = get_common_args("Generate Main Results Table")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_main_results_table(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
