"""
Quadrant assignment using U (uncertainty) and L_local (actionability).

The quadrant structure is the PRIMARY organizing principle for experiments:
- Q1 (high U, high L): Uncertain and fixable
- Q2 (high U, low L): Uncertain and stuck
- Q3 (low U, low L): Confident and stuck
- Q4 (low U, high L): Confident but wrong
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

from ..data.snapshot import LabeledSnapshot


QuadrantLabel = Literal["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]


QUADRANT_NAMES = {
    "Q1_highU_highL": "Q1: Uncertain & Fixable",
    "Q2_highU_lowL": "Q2: Uncertain & Stuck",
    "Q3_lowU_lowL": "Q3: Confident & Stuck",
    "Q4_lowU_highL": "Q4: Confident but Wrong",
}

QUADRANT_DESCRIPTIONS = {
    "Q1_highU_highL": "High uncertainty, high actionability. Agent is unsure but can be helped with a hint.",
    "Q2_highU_lowL": "High uncertainty, low actionability. Agent is unsure and stuck; needs trajectory-level help.",
    "Q3_lowU_lowL": "Low uncertainty, low actionability. Agent is confident but stuck; may be a dead-end.",
    "Q4_lowU_highL": "Low uncertainty, high actionability. Agent is confident but wrong; needs contrast training.",
}


@dataclass
class QuadrantConfig:
    """Configuration for quadrant assignment."""
    
    U_threshold: Optional[float] = None  # Will be set to median if None
    L_threshold: Optional[float] = None  # Will be set to median if None
    method: str = "median"  # "median" or "percentile_75"
    
    @classmethod
    def from_yaml(cls, path: str) -> "QuadrantConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("quadrant", {}))


def compute_thresholds(
    snapshots: list[dict],
    method: str = "median",
) -> tuple[float, float]:
    """
    Compute U and L thresholds from snapshot distribution.
    
    Args:
        snapshots: List of snapshot dicts with uncertainty and leverage data
        method: "median" or "percentile_75"
        
    Returns:
        Tuple of (U_threshold, L_threshold)
    """
    # Extract values, handling both dict and LabeledSnapshot formats
    U_values = []
    L_values = []
    
    for s in snapshots:
        if isinstance(s, dict):
            U_values.append(s.get("uncertainty", {}).get("entropy", s.get("U", 0)))
            L_values.append(s.get("leverage", {}).get("L_local", 0))
        elif hasattr(s, "U") and hasattr(s, "leverage"):
            U_values.append(s.U)
            L_values.append(s.leverage.L_local if s.leverage else 0)
    
    if not U_values or not L_values:
        raise ValueError("No valid uncertainty/leverage data found in snapshots")
    
    if method == "median":
        U_threshold = float(np.median(U_values))
        L_threshold = float(np.median(L_values))
    elif method == "percentile_75":
        U_threshold = float(np.percentile(U_values, 75))
        L_threshold = float(np.percentile(L_values, 75))
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    print(f"Computed thresholds: U={U_threshold:.3f}, L={L_threshold:.3f}")
    print(f"  U range: [{min(U_values):.3f}, {max(U_values):.3f}]")
    print(f"  L range: [{min(L_values):.3f}, {max(L_values):.3f}]")
    
    return U_threshold, L_threshold


def assign_quadrant(
    U: float,
    L_local: float,
    U_threshold: float,
    L_threshold: float,
) -> QuadrantLabel:
    """
    Assign snapshot to quadrant based on U and L_local.
    
    Quadrant interpretations:
    - Q1 (high U, high L): Uncertain and fixable - Hint supervision recommended
    - Q2 (high U, low L): Uncertain and stuck - Limited intervention possible
    - Q3 (low U, low L): Confident and stuck - Low ELP expected
    - Q4 (low U, high L): Confident but wrong - Contrast supervision recommended
    
    Args:
        U: Uncertainty value (higher = more uncertain)
        L_local: Single-step leverage (higher = more actionable)
        U_threshold: Threshold for high vs low uncertainty
        L_threshold: Threshold for high vs low leverage
        
    Returns:
        Quadrant label string
    """
    high_U = U > U_threshold
    high_L = L_local > L_threshold
    
    if high_U and high_L:
        return "Q1_highU_highL"
    elif high_U and not high_L:
        return "Q2_highU_lowL"
    elif not high_U and not high_L:
        return "Q3_lowU_lowL"
    else:  # low U, high L
        return "Q4_lowU_highL"


def partition_by_quadrant(
    snapshots: list[dict],
    U_threshold: float,
    L_threshold: float,
) -> dict[QuadrantLabel, list[dict]]:
    """
    Partition snapshots into quadrants.
    
    Args:
        snapshots: List of snapshot dicts with uncertainty and leverage data
        U_threshold: Threshold for uncertainty
        L_threshold: Threshold for leverage
        
    Returns:
        Dict mapping quadrant labels to lists of snapshots
    """
    partitions: dict[QuadrantLabel, list[dict]] = {
        "Q1_highU_highL": [],
        "Q2_highU_lowL": [],
        "Q3_lowU_lowL": [],
        "Q4_lowU_highL": [],
    }
    
    for snap in snapshots:
        # Extract U and L values
        if isinstance(snap, dict):
            U = snap.get("uncertainty", {}).get("entropy", snap.get("U", 0))
            L_local = snap.get("leverage", {}).get("L_local", 0)
        elif hasattr(snap, "U") and hasattr(snap, "leverage"):
            U = snap.U
            L_local = snap.leverage.L_local if snap.leverage else 0
        else:
            continue
        
        quadrant = assign_quadrant(U, L_local, U_threshold, L_threshold)
        
        # Add quadrant to snapshot
        if isinstance(snap, dict):
            snap["quadrant"] = quadrant
        elif hasattr(snap, "quadrant"):
            snap.quadrant = quadrant
        
        partitions[quadrant].append(snap)
    
    # Report distribution
    total = len(snapshots)
    print("Quadrant distribution:")
    for q, snaps in partitions.items():
        print(f"  {q}: {len(snaps)} ({100*len(snaps)/total:.1f}%)")
    
    return partitions


def assign_quadrants_to_labeled_snapshots(
    snapshots: list[LabeledSnapshot],
    U_threshold: Optional[float] = None,
    L_threshold: Optional[float] = None,
) -> list[LabeledSnapshot]:
    """
    Assign quadrants to a list of LabeledSnapshot objects.
    
    If thresholds are not provided, they are computed from the data.
    
    Args:
        snapshots: List of LabeledSnapshot objects
        U_threshold: Optional U threshold (computed if None)
        L_threshold: Optional L threshold (computed if None)
        
    Returns:
        Same list with quadrant fields updated
    """
    if U_threshold is None or L_threshold is None:
        # Convert to dict format for threshold computation
        snap_dicts = [s.to_dict() for s in snapshots]
        computed_U, computed_L = compute_thresholds(snap_dicts)
        
        U_threshold = U_threshold if U_threshold is not None else computed_U
        L_threshold = L_threshold if L_threshold is not None else computed_L
    
    for snap in snapshots:
        if snap.leverage is not None:
            snap.quadrant = assign_quadrant(
                snap.U, snap.leverage.L_local,
                U_threshold, L_threshold
            )
    
    return snapshots


def get_quadrant_statistics(
    partitions: dict[QuadrantLabel, list],
) -> dict:
    """
    Compute statistics for each quadrant.
    
    Args:
        partitions: Dict mapping quadrant labels to snapshot lists
        
    Returns:
        Dict with per-quadrant and aggregate statistics
    """
    stats = {}
    total = sum(len(snaps) for snaps in partitions.values())
    
    for quadrant, snaps in partitions.items():
        if not snaps:
            stats[quadrant] = {"count": 0, "percentage": 0.0}
            continue
        
        # Extract values
        U_values = []
        L_values = []
        elp_values = []
        
        for s in snaps:
            if isinstance(s, dict):
                U_values.append(s.get("uncertainty", {}).get("entropy", s.get("U", 0)))
                L_values.append(s.get("leverage", {}).get("L_local", 0))
                if "cpt" in s and s["cpt"]:
                    elp_values.append(s["cpt"].get("ELP_net", 0))
            elif hasattr(s, "U") and hasattr(s, "leverage"):
                U_values.append(s.U)
                L_values.append(s.leverage.L_local if s.leverage else 0)
                if s.cpt:
                    elp_values.append(s.cpt.ELP_net)
        
        stats[quadrant] = {
            "count": len(snaps),
            "percentage": 100 * len(snaps) / total if total > 0 else 0,
            "U_mean": float(np.mean(U_values)) if U_values else 0,
            "U_std": float(np.std(U_values)) if U_values else 0,
            "L_mean": float(np.mean(L_values)) if L_values else 0,
            "L_std": float(np.std(L_values)) if L_values else 0,
            "ELP_mean": float(np.mean(elp_values)) if elp_values else None,
            "ELP_std": float(np.std(elp_values)) if elp_values else None,
        }
    
    stats["total"] = total
    
    return stats


def get_recommended_supervision(quadrant: QuadrantLabel) -> str:
    """
    Get recommended supervision type for a quadrant.
    
    Based on hypotheses in research blueprint §3.6.2.
    
    Args:
        quadrant: Quadrant label
        
    Returns:
        Recommended supervision type
    """
    recommendations = {
        "Q1_highU_highL": "hint",      # Agent needs direction, not full demo
        "Q2_highU_lowL": "demo",       # State may be beyond local repair, try demo
        "Q3_lowU_lowL": "demo",        # Low ELP expected, standard demo
        "Q4_lowU_highL": "contrast",   # Agent needs to unlearn confident mistake
    }
    return recommendations.get(quadrant, "demo")
