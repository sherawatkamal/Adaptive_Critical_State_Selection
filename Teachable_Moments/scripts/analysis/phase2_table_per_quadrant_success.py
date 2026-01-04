#!/usr/bin/env python3
"""Generate Table 1: Per-Quadrant Success (Phase 2)."""

import pandas as pd
import json
from scripts.analysis.plotting_utils import (
    setup_logging, get_common_args
)

logger = setup_logging()

def generate_table1_per_quadrant_success(results_path, output_path):
    """
    Generate Table 1: Per-quadrant success rates.
    Shows success rate by quadrant and supervision type, with baseline comparisons.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    # Build table
    rows = []
    quadrants = ["Q1_high_high", "Q2_high_low", "Q3_low_high", "Q4_low_low"]
    supervision_types = ["demo", "contrast", "hint"]
    
    for quadrant in quadrants:
        row = {"Quadrant": quadrant}
        for sup_type in supervision_types:
            key = f"{quadrant}_{sup_type}"
            result = data.get(key, {})
            row[sup_type] = f"{result.get('success_rate', 0):.1%}"
        rows.append(row)
    
    # Add baseline rows
    # Assuming baselines are present in the JSON
    for baseline in ["B1_uniform", "B2_all"]:
        if baseline in data:
            result = data.get(baseline, {})
            rows.append({
                "Quadrant": baseline,
                "demo": f"{result.get('success_rate', 0):.1%}",
                "contrast": "-",
                "hint": "-",
            })
    
    df = pd.DataFrame(rows)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False)
    latex_path = output_path.with_suffix(".tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    
    # Save as CSV
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Generated per-quadrant success table: {latex_path}, {csv_path}")

def main():
    parser = get_common_args("Generate Table 1 Per-Quadrant")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_table1_per_quadrant_success(
        args.results_dir / "phase2" / "per_quadrant_results.json",
        args.output_dir / "table1_per_quadrant_success",
    )

if __name__ == "__main__":
    main()
