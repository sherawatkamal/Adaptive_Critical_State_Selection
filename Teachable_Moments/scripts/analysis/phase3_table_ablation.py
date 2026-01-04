#!/usr/bin/env python3
"""Generate Ablation Table (Phase 3)."""

from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args, load_json
)

logger = setup_logging()

def generate_ablation_table(results_dir, output_dir):
    """Generate ablation study results table."""
    # Ablations are usually in phase3 or special ablation file
    ablation_path = results_dir / "phase3" / "ablation_results.json"
    data = load_json(ablation_path)

    # Expected list of dicts: {name, config: {u, cpt, l, adapt}, success_rate}
    ablations = data.get("ablations", [])
    if not ablations:
        raise ValueError("No ablation data found.")

    # LaTeX
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study results showing contribution of each component.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Configuration & Uncert. & CPT & Lever. & Adapt. & Success \\\\",
        "\\midrule",
    ]
    
    for entry in ablations:
        name = entry.get("name", "")
        cfg = entry.get("config", {})
        sr = entry.get("success_rate", 0.0)
        
        # Format checks
        u = "\\checkmark" if cfg.get("uncertainty") else "-"
        cpt = "\\checkmark" if cfg.get("cpt") else "-"
        l = "\\checkmark" if cfg.get("leverage") else "-"
        adapt = "\\checkmark" if cfg.get("adaptive") else "-"
        
        if name == "Full Model":
            latex_lines.append(
                f"\\textbf{{{name}}} & {u} & {cpt} & {l} & {adapt} & \\textbf{{{sr:.2f}}} \\\\"
            )
        else:
            latex_lines.append(f"{name} & {u} & {cpt} & {l} & {adapt} & {sr:.2f} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_path = output_dir / "ablation.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    # Markdown
    md_lines = [
        "| Configuration | Uncertainty | CPT | Leverage | Adaptive | Success |",
        "|---------------|-------------|-----|----------|----------|---------|",
    ]
    
    for entry in ablations:
        name = entry.get("name", "")
        cfg = entry.get("config", {})
        sr = entry.get("success_rate", 0.0)
        
        u = "✓" if cfg.get("uncertainty") else "-"
        cpt = "✓" if cfg.get("cpt") else "-"
        l = "✓" if cfg.get("leverage") else "-"
        adapt = "✓" if cfg.get("adaptive") else "-"
        
        md_lines.append(f"| {name} | {u} | {cpt} | {l} | {adapt} | {sr:.2f} |")
    
    md_path = output_dir / "ablation.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Generated ablation table: {latex_path}")

def main():
    parser = get_common_args("Generate Ablation Table")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_ablation_table(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
