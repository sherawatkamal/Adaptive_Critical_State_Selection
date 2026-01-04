#!/usr/bin/env python3
"""Generate Predictor Table (Phase 4)."""

from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args, load_json
)

logger = setup_logging()

def generate_predictor_table(results_dir, output_dir):
    """Generate predictor performance table."""
    pred_path = results_dir / "phase4" / "predictor_evaluation.json"
    data = load_json(pred_path)
    
    # Expected: {uncertainty: {r2, correlation, mse}, ...}
    # We define what we look for.
    tasks = [
        ("Uncertainty", "Regression", "uncertainty"),
        ("Leverage", "Regression", "leverage"),
        ("ELP", "Regression", "elp"),
        ("Quadrant", "Classification", "quadrant"),
    ]
    
    # LaTeX
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Predictor performance on held-out test set.}",
        "\\label{tab:predictor}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Task & Type & Correlation/Acc & $R^2$/- & MSE/- \\\\",
        "\\midrule",
    ]
    
    for task_name, task_type, key in tasks:
        stats = data.get(key, {})
        
        if task_type == "Regression":
            corr = stats.get("correlation", 0.0)
            r2 = stats.get("r2", 0.0)
            mse = stats.get("mse", 0.0)
            latex_lines.append(f"{task_name} & {task_type} & {corr:.2f} & {r2:.2f} & {mse:.3f} \\\\")
        else:
            acc = stats.get("accuracy", 0.0)
            latex_lines.append(f"{task_name} & {task_type} & {acc:.2f} & - & - \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_path = output_dir / "predictor_performance.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    # Markdown
    md_lines = [
        "| Task | Type | Correlation/Accuracy | R² | MSE |",
        "|------|------|---------------------|-----|-----|",
    ]
    
    for task_name, task_type, key in tasks:
        stats = data.get(key, {})
        if task_type == "Regression":
            corr = stats.get("correlation", 0.0)
            r2 = stats.get("r2", 0.0)
            mse = stats.get("mse", 0.0)
            md_lines.append(f"| {task_name} | {task_type} | {corr:.2f} | {r2:.2f} | {mse:.3f} |")
        else:
            acc = stats.get("accuracy", 0.0)
            md_lines.append(f"| {task_name} | {task_type} | {acc:.2f} | - | - |")
    
    md_path = output_dir / "predictor_performance.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Generated predictor table: {latex_path}")

def main():
    parser = get_common_args("Generate Predictor Table")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_predictor_table(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
