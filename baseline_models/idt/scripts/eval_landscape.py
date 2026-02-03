#!/usr/bin/env python3
"""
CLI: Generate evaluation metrics and teachability landscape plots.
  python -m idt.scripts.eval_landscape --dataset_path ... --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idt.eval import aggregate_metrics, load_dataset, plot_teachability_landscape

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="idt_eval_out")
    args = ap.parse_args()

    records = load_dataset(args.dataset_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = aggregate_metrics(records)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", json.dumps(metrics, indent=2))

    plot_teachability_landscape(records, out_dir)
    print("Plots saved to", out_dir)


if __name__ == "__main__":
    main()
