#!/usr/bin/env python3
"""Generate comprehensive training summary.

Aggregates results from per-quadrant training and baselines into
a unified summary for Phase 3 evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import setup_logging, save_json, get_timestamp


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def collect_model_info(base_dir: Path) -> dict[str, Any]:
    """Collect information about all trained models."""
    models = {
        "per_quadrant": [],
        "baselines": [],
    }
    
    # Per-quadrant models
    quadrant_summary = load_json_file(base_dir / "models" / "training_summary.json")
    if quadrant_summary:
        for result in quadrant_summary.get("results", []):
            models["per_quadrant"].append({
                "name": f"{result['quadrant']}_{result['supervision']}",
                "quadrant": result["quadrant"],
                "supervision": result["supervision"],
                "n_samples": result["n_samples"],
                "final_loss": result["final_loss"],
                "checkpoint": result["checkpoint_path"],
            })
    
    # Baselines
    baseline_summary = load_json_file(base_dir / "baselines" / "baseline_summary.json")
    if baseline_summary:
        for result in baseline_summary.get("results", []):
            models["baselines"].append({
                "name": result["name"],
                "n_samples": result["n_samples"],
                "final_loss": result["final_loss"],
                "checkpoint": result["checkpoint_path"],
            })
    
    return models


def compute_training_statistics(models: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate statistics across all training runs."""
    stats = {}
    
    # Per-quadrant stats
    pq_models = models.get("per_quadrant", [])
    if pq_models:
        losses = [m["final_loss"] for m in pq_models]
        samples = [m["n_samples"] for m in pq_models]
        
        stats["per_quadrant"] = {
            "n_models": len(pq_models),
            "total_samples": sum(samples),
            "avg_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
        }
        
        # By quadrant
        by_quadrant = {}
        for m in pq_models:
            q = m["quadrant"]
            if q not in by_quadrant:
                by_quadrant[q] = []
            by_quadrant[q].append(m["final_loss"])
        
        stats["by_quadrant"] = {
            q: {"avg_loss": sum(losses) / len(losses), "n_models": len(losses)}
            for q, losses in by_quadrant.items()
        }
        
        # By supervision type
        by_sup = {}
        for m in pq_models:
            s = m["supervision"]
            if s not in by_sup:
                by_sup[s] = []
            by_sup[s].append(m["final_loss"])
        
        stats["by_supervision"] = {
            s: {"avg_loss": sum(losses) / len(losses), "n_models": len(losses)}
            for s, losses in by_sup.items()
        }
    
    # Baseline stats
    baselines = models.get("baselines", [])
    if baselines:
        stats["baselines"] = {
            m["name"]: {"loss": m["final_loss"], "n_samples": m["n_samples"]}
            for m in baselines
        }
    
    return stats


def generate_report(models: dict[str, Any], stats: dict[str, Any]) -> str:
    """Generate human-readable training report."""
    lines = [
        "=" * 60,
        "PHASE 2 TRAINING SUMMARY",
        "=" * 60,
        "",
    ]
    
    # Overview
    pq_stats = stats.get("per_quadrant", {})
    lines.extend([
        "OVERVIEW",
        "-" * 40,
        f"Per-quadrant models: {pq_stats.get('n_models', 0)}",
        f"Baseline models: {len(models.get('baselines', []))}",
        f"Total training samples: {pq_stats.get('total_samples', 0)}",
        "",
    ])
    
    # Per-quadrant summary
    if pq_stats:
        lines.extend([
            "PER-QUADRANT TRAINING",
            "-" * 40,
            f"Average loss: {pq_stats.get('avg_loss', 0):.4f}",
            f"Loss range: [{pq_stats.get('min_loss', 0):.4f}, {pq_stats.get('max_loss', 0):.4f}]",
            "",
        ])
        
        # By quadrant
        lines.append("By Quadrant:")
        for q, q_stats in sorted(stats.get("by_quadrant", {}).items()):
            lines.append(f"  {q}: avg_loss={q_stats['avg_loss']:.4f} (n={q_stats['n_models']})")
        lines.append("")
        
        # By supervision
        lines.append("By Supervision Type:")
        for s, s_stats in sorted(stats.get("by_supervision", {}).items()):
            lines.append(f"  {s}: avg_loss={s_stats['avg_loss']:.4f} (n={s_stats['n_models']})")
        lines.append("")
    
    # Baselines
    baseline_stats = stats.get("baselines", {})
    if baseline_stats:
        lines.extend([
            "BASELINES",
            "-" * 40,
        ])
        for name, b_stats in baseline_stats.items():
            lines.append(f"  {name}: loss={b_stats['loss']:.4f}, n={b_stats['n_samples']}")
        lines.append("")
    
    # Model inventory
    lines.extend([
        "MODEL INVENTORY",
        "-" * 40,
    ])
    
    for m in models.get("per_quadrant", []):
        lines.append(f"  [{m['quadrant']}+{m['supervision']}] {m['checkpoint']}")
    
    for m in models.get("baselines", []):
        lines.append(f"  [baseline:{m['name']}] {m['checkpoint']}")
    
    lines.extend(["", "=" * 60])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate training summary")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Directory containing training results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Output directory",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Collect model information
    logger.info(f"Collecting model info from {args.input_dir}")
    models = collect_model_info(args.input_dir)
    
    n_pq = len(models.get("per_quadrant", []))
    n_bl = len(models.get("baselines", []))
    logger.info(f"Found {n_pq} per-quadrant models, {n_bl} baselines")
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_training_statistics(models)
    
    # Generate report
    report = generate_report(models, stats)
    
    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": get_timestamp(),
        "models": models,
        "statistics": stats,
    }
    
    save_json(summary, args.output_dir / "complete_training_summary.json")
    
    report_path = args.output_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Print report
    print(report)
    
    logger.info(f"Summary saved to {args.output_dir}")


if __name__ == "__main__":
    main()
