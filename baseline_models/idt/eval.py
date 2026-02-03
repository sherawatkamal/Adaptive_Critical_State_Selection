"""
Evaluation: aggregate metrics, teachability landscape (fraction teachable vs K, vs patch type).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load IDT JSONL dataset."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """% teachable, patch type histogram, mean recovery rate, compute used."""
    if not records:
        return {
            "num_records": 0,
            "pct_teachable": 0.0,
            "patch_type_histogram": {},
            "mean_R1": 0.0,
            "mean_R3": 0.0,
            "mean_R5": 0.0,
            "total_env_steps": 0,
            "total_model_calls": 0,
        }
    teachable = sum(1 for r in records if r.get("teachable_label"))
    hist: Dict[str, int] = defaultdict(int)
    for r in records:
        t = r.get("patch_type") or "unknown"
        hist[t] += 1
    r1 = [r.get("R1", 0) for r in records]
    r3 = [r.get("R3", 0) for r in records]
    r5 = [r.get("R5", 0) for r in records]
    env_steps = sum(r.get("compute_counters", {}).get("env_steps", 0) for r in records)
    model_calls = sum(r.get("compute_counters", {}).get("model_calls", 0) for r in records)
    return {
        "num_records": len(records),
        "pct_teachable": teachable / len(records) * 100.0,
        "patch_type_histogram": dict(hist),
        "mean_R1": sum(r1) / len(r1) if r1 else 0.0,
        "mean_R3": sum(r3) / len(r3) if r3 else 0.0,
        "mean_R5": sum(r5) / len(r5) if r5 else 0.0,
        "total_env_steps": env_steps,
        "total_model_calls": model_calls,
    }


def plot_teachability_landscape(
    records: List[Dict[str, Any]],
    out_dir: str | Path,
) -> None:
    """Fraction teachable vs K (R1/R3/R5) and vs patch type. Matplotlib only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    # By K: fraction with R1/R3/R5 >= 0.6
    threshold = 0.6
    n = len(records)
    frac_r1 = sum(1 for r in records if (r.get("R1") or 0) >= threshold) / n
    frac_r3 = sum(1 for r in records if (r.get("R3") or 0) >= threshold) / n
    frac_r5 = sum(1 for r in records if (r.get("R5") or 0) >= threshold) / n
    fig, ax = plt.subplots()
    ax.bar(["R1", "R3", "R5"], [frac_r1, frac_r3, frac_r5], color="steelblue")
    ax.set_ylabel("Fraction teachable (R>=%.2f)" % threshold)
    ax.set_title("Teachability landscape by attempt budget K")
    fig.savefig(out_dir / "landscape_by_k.png", dpi=100, bbox_inches="tight")
    plt.close()

    # By patch type
    hist: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        t = r.get("patch_type") or "unknown"
        hist[t].append(1.0 if (r.get("R5") or 0) >= threshold else 0.0)
    types = list(hist.keys())
    fracs = [sum(hist[t]) / len(hist[t]) if hist[t] else 0.0 for t in types]
    fig, ax = plt.subplots()
    ax.bar(types, fracs, color="coral")
    ax.set_ylabel("Fraction teachable (R5>=%.2f)" % threshold)
    ax.set_title("Teachability by patch type")
    plt.xticks(rotation=15)
    fig.savefig(out_dir / "landscape_by_patch_type.png", dpi=100, bbox_inches="tight")
    plt.close()
