"""H7 Table: selection summary.

Builds a compact CSV suitable for the paper's H7 result:

  - failure-panel gain (Δ success vs BASE)
  - retention drop (fine-tuned − base)
  - optionally: overall task reward/success (if present)

Inputs (from eval suite output dir):
  - per_quadrant_results.csv
  - retention_results.csv
  - overall_task_results.csv (optional)

Output:
  - <output-dir>/table_h7_selection_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Table H7: selection summary")
    ap.add_argument("--eval-dir", type=Path, default=Path("results/phase5/v8"))
    ap.add_argument("--output-dir", type=Path, default=Path("tables/v8"))
    ap.add_argument("--prefix", type=str, default="SEL_")
    args = ap.parse_args()

    p_panel = args.eval_dir / "per_quadrant_results.csv"
    p_ret = args.eval_dir / "retention_results.csv"
    p_overall = args.eval_dir / "overall_task_results.csv"
    if not p_panel.exists() or not p_ret.exists():
        raise SystemExit(f"Missing required eval outputs in {args.eval_dir}")

    panel = load_csv(p_panel)
    ret = load_csv(p_ret)
    overall = load_csv(p_overall) if p_overall.exists() else []

    # failure-panel mean success by model
    panel_sums: Dict[str, float] = {}
    panel_counts: Dict[str, int] = {}
    for r in panel:
        m = r.get("model")
        if not m:
            continue
        sr = float(r.get("success_rate"))
        panel_sums[m] = panel_sums.get(m, 0.0) + sr
        panel_counts[m] = panel_counts.get(m, 0) + 1
    panel_mean = {m: panel_sums[m] / max(1, panel_counts[m]) for m in panel_sums}
    base_panel = panel_mean.get("BASE")
    if base_panel is None:
        raise SystemExit("BASE row missing from per_quadrant_results.csv")

    ret_drop = {r.get("model"): float(r.get("retention_drop")) for r in ret if r.get("model")}
    ret_rate = {r.get("model"): float(r.get("retention_success_rate")) for r in ret if r.get("model")}

    overall_by = {r.get("model"): r for r in overall if r.get("model")}

    pretty = {
        "SEL_random": "random",
        "SEL_entropy": "uncertainty",
        "SEL_predictor": "predictor",
        "SEL_oracle": "oracle",
    }

    rows: List[Dict[str, Any]] = []
    for model, mean_sr in sorted(panel_mean.items()):
        if model == "BASE" or not model.startswith(args.prefix):
            continue
        if model not in ret_drop:
            continue
        o = overall_by.get(model, {})
        rows.append(
            {
                "method": pretty.get(model, model),
                "run_id": model,
                "failure_panel_gain": mean_sr - base_panel,
                "retention_drop": ret_drop.get(model),
                "retention_success_rate": ret_rate.get(model),
                "overall_success_rate_any": o.get("success_rate_any", ""),
                "overall_mean_reward": o.get("mean_reward", ""),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "table_h7_selection_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
