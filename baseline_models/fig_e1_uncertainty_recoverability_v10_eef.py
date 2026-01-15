#!/usr/bin/env python3
"""
fig_e1_uncertainty_recoverability_v10_eef.py - ADAPTED FOR EEF CSV FORMAT

E1: Uncertainty → Recoverability diagnostics.

Input:
  CSV from EEF pipeline (data_with_quadrants.csv)

This script produces:
- AUC (U predicts success)
- Enrichment curve: recoverable rate among top-fraction-by-U vs global rate
- Budget curve: recoverable yield vs resample budget b (1..Bmax)
  + optional bootstrap CI over snapshots

Example
-------
python fig_e1_uncertainty_recoverability_v10_eef.py \
  --recovery-trajectories data_with_quadrants.csv \
  --out-dir ./v10_analysis \
  --bootstrap 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _read_data(path: Path) -> List[Dict[str, Any]]:
    """Read CSV format from EEF pipeline."""
    df = pd.read_csv(path)
    
    print(f"Loaded CSV with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Determine success threshold (median L)
    l_threshold = df['L'].median()
    
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        # Map CSV columns to expected format
        u_val = row.get('U', row.get('normalized_entropy', row.get('true_entropy', 0)))
        
        # Success: either explicit success column or L > threshold
        success = row.get('success', False)
        if pd.isna(success) or success is None:
            success = row['L'] > l_threshold
        
        record = {
            'U': float(u_val),
            'success': bool(success),
            'attempts_used': int(row.get('num_attempts', 1)),  # Default to 1
            'L': float(row.get('L', row.get('improvement', 0))),
        }
        out.append(record)
    
    print(f"\nData summary:")
    print(f"  L threshold (median): {l_threshold:.2f}")
    print(f"  Success rate: {100*sum(r['success'] for r in out)/len(out):.1f}%")
    print(f"  Mean U: {sum(r['U'] for r in out)/len(out):.3f}")
    print(f"  Mean L: {sum(r['L'] for r in out)/len(out):.2f}")
    
    return out


def _auc_rank(y_true: List[int], y_score: List[float]) -> float:
    """
    Compute ROC AUC via rank statistic (equivalent to Mann–Whitney U).
    """
    pairs = list(zip(y_score, y_true))
    pairs.sort(key=lambda t: t[0])  # ascending
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # average ranks for ties
    ranks = [0.0] * len(pairs)
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        rank += (j - i)
        i = j

    sum_ranks_pos = sum(ranks[idx] for idx, (_, y) in enumerate(pairs) if y == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _bootstrap_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    vals = sorted(values)
    lo_idx = int((alpha / 2.0) * (len(vals) - 1))
    hi_idx = int((1 - alpha / 2.0) * (len(vals) - 1))
    return vals[lo_idx], vals[hi_idx]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E1: Uncertainty predicts recoverability (v10, EEF-adapted)")
    ap.add_argument("--recovery-trajectories", type=Path, required=True,
                   help="Path to data_with_quadrants.csv")
    ap.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for plots")
    ap.add_argument("--bootstrap", type=int, default=200, 
                   help="Bootstrap samples for CI (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_data(args.recovery_trajectories)
    if not rows:
        raise SystemExit("No rows found.")

    U: List[float] = []
    y: List[int] = []
    attempts_used: List[int] = []

    for r in rows:
        U.append(float(r["U"]))
        y.append(1 if bool(r["success"]) else 0)
        attempts_used.append(int(r["attempts_used"]))

    auc = _auc_rank(y, U)
    base_rate = sum(y) / float(len(y))

    print(f"\n{'='*70}")
    print(f"UNCERTAINTY → RECOVERABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"N samples: {len(y)}")
    print(f"Base recoverable rate: {100*base_rate:.1f}%")
    print(f"AUC (U predicts success): {auc:.3f}")
    print(f"{'='*70}\n")

    # Enrichment curve (top fraction by U)
    pairs = list(zip(U, y))
    pairs.sort(key=lambda t: t[0], reverse=True)

    fracs = [i / 50.0 for i in range(1, 51)]  # 0.02..1.0
    enrich: List[float] = []
    rates: List[float] = []
    for f in fracs:
        k = max(1, int(round(f * len(pairs))))
        rate = sum(lbl for _, lbl in pairs[:k]) / float(k)
        rates.append(rate)
        enrich.append(rate / base_rate if base_rate > 0 else float("nan"))

    # Budget curve: yield vs b
    Bmax = max(attempts_used) if attempts_used else 1
    budgets = list(range(1, Bmax + 1))
    yield_mean: List[float] = []
    yield_lo: List[float] = []
    yield_hi: List[float] = []

    rng = random.Random(args.seed)
    for b in budgets:
        # empirical yield: success within b attempts
        ok = [1 if (yy == 1 and au <= b) else 0 for yy, au in zip(y, attempts_used)]
        m = sum(ok) / float(len(ok))
        yield_mean.append(m)

        if args.bootstrap and args.bootstrap > 0:
            boot = []
            for _ in range(int(args.bootstrap)):
                idxs = [rng.randrange(len(ok)) for _ in range(len(ok))]
                boot_m = sum(ok[i] for i in idxs) / float(len(ok))
                boot.append(boot_m)
            lo, hi = _bootstrap_ci(boot, alpha=0.05)
            yield_lo.append(lo)
            yield_hi.append(hi)
        else:
            yield_lo.append(float("nan"))
            yield_hi.append(float("nan"))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "n": len(y),
        "base_recoverable_rate": base_rate,
        "auc_roc": auc,
        "Bmax": Bmax,
    }
    (out_dir / "e1_uncertainty_recoverability_stats.json").write_text(json.dumps(stats, indent=2))

    # Plot enrichment curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fracs, rates, linewidth=2.5, color='steelblue', label="Recoverable rate @ top-fraction(U)")
    ax.axhline(base_rate, linestyle="--", linewidth=2, color='red', label=f"Random baseline ({100*base_rate:.1f}%)")
    ax.set_xlabel("Top fraction selected by uncertainty (higher U first)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Recoverable fraction", fontsize=12, fontweight='bold')
    ax.set_title(f"Uncertainty Enriches for Recoverable States\nAUC = {auc:.3f}", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_e1_enrichment_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'fig_e1_enrichment_curve.png'}")

    # Plot enrichment ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fracs, enrich, linewidth=2.5, color='coral')
    ax.axhline(1.0, linestyle="--", linewidth=2, color='black', alpha=0.5, label="No enrichment")
    ax.set_xlabel("Top fraction selected by uncertainty", fontsize=12, fontweight='bold')
    ax.set_ylabel("Enrichment ratio (vs random)", fontsize=12, fontweight='bold')
    ax.set_title("Enrichment: Recovery Rate / Baseline Rate", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_e1_enrichment_ratio.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_dir / 'fig_e1_enrichment_ratio.png'}")

    # Plot budget curve
    if Bmax > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(budgets, yield_mean, linewidth=2.5, marker='o', markersize=8, 
               color='green', label="Recoverable yield")
        if args.bootstrap and args.bootstrap > 0:
            ax.fill_between(budgets, yield_lo, yield_hi, alpha=0.25, color='green',
                           label="95% bootstrap CI")
        ax.set_xlabel("Resample budget b (attempts per snapshot)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Fraction recoverable within b attempts", fontsize=12, fontweight='bold')
        ax.set_title("Recovery Yield vs Resample Budget", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_e1_budget_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {out_dir / 'fig_e1_budget_curve.png'}")
    else:
        print("⚠ Skipping budget curve (Bmax=1, no variation)")

    print(f"\n✓ Wrote E1 figs + stats to {out_dir}")
    print(f"\nKey results:")
    print(f"  AUC = {auc:.3f}")
    print(f"  Base rate = {100*base_rate:.1f}%")
    
    # Compute enrichment at key percentiles
    for pct_idx, pct in enumerate([0.1, 0.2, 0.5]):
        idx = int(pct * len(fracs))
        if idx < len(rates):
            lift = (rates[idx] / base_rate - 1) * 100 if base_rate > 0 else 0
            print(f"  Top {int(pct*100)}% by U: {100*rates[idx]:.1f}% recoverable (lift: {lift:+.1f}%)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())