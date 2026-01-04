"""
Recovery depth computation from leverage data.

Depth measures how many steps back one needs to go to enable recovery.
This is computed offline from already-computed leverage labels.
"""

from typing import Optional
from dataclasses import dataclass

from ..data.snapshot import LabeledSnapshot, DepthLabels


@dataclass
class DepthConfig:
    """Configuration for depth computation."""
    
    recovery_threshold: float = 0.5  # Consider recovered if p >= threshold
    
    @classmethod
    def from_yaml(cls, path: str) -> "DepthConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("depth", {}))


def compute_depth_from_leverage(
    trajectory_snapshots: list[LabeledSnapshot],
    recovery_threshold: float = 0.5,
) -> dict[str, DepthLabels]:
    """
    Given snapshots at fixed intervals along a trajectory,
    compute depth by walking backward until recovery threshold is met.
    
    No additional rollouts required - uses already-computed leverage.
    
    Args:
        trajectory_snapshots: Snapshots ordered by step_idx from same trajectory
        recovery_threshold: Success probability threshold for "recovered"
        
    Returns:
        Dict mapping snapshot IDs to DepthLabels
    """
    # Sort by step index
    sorted_snapshots = sorted(
        trajectory_snapshots,
        key=lambda s: s.snapshot.step_idx
    )
    
    depth_labels = {}
    
    for i, snap in enumerate(sorted_snapshots):
        if snap.leverage is None:
            continue
        
        d_expert = None
        d_force = None
        
        # Walk backward through earlier snapshots
        for j in range(i, -1, -1):
            earlier_snap = sorted_snapshots[j]
            if earlier_snap.leverage is None:
                continue
            
            steps_back = snap.snapshot.step_idx - earlier_snap.snapshot.step_idx
            
            # Check expert recovery condition
            if d_expert is None and earlier_snap.leverage.p_expert >= recovery_threshold:
                d_expert = steps_back
            
            # Check single-action recovery condition
            if d_force is None and earlier_snap.leverage.p_force >= recovery_threshold:
                d_force = steps_back
        
        # Default to trajectory length if never recoverable
        max_depth = snap.snapshot.step_idx
        d_expert = d_expert if d_expert is not None else max_depth
        d_force = d_force if d_force is not None else max_depth
        
        depth_labels[snap.snapshot.id] = DepthLabels(
            d_expert=d_expert,
            d_force=d_force,
        )
    
    return depth_labels


def compute_depth_for_all_trajectories(
    snapshots_by_trajectory: dict[str, list[LabeledSnapshot]],
    recovery_threshold: float = 0.5,
) -> dict[str, DepthLabels]:
    """
    Compute depth for all snapshots across multiple trajectories.
    
    Args:
        snapshots_by_trajectory: Dict mapping trajectory IDs to snapshot lists
        recovery_threshold: Success probability threshold for "recovered"
        
    Returns:
        Dict mapping snapshot IDs to DepthLabels
    """
    all_depths = {}
    
    for traj_id, snapshots in snapshots_by_trajectory.items():
        traj_depths = compute_depth_from_leverage(snapshots, recovery_threshold)
        all_depths.update(traj_depths)
    
    return all_depths


def group_snapshots_by_trajectory(
    snapshots: list[LabeledSnapshot],
) -> dict[str, list[LabeledSnapshot]]:
    """
    Group snapshots by their trajectory ID.
    
    Args:
        snapshots: List of labeled snapshots
        
    Returns:
        Dict mapping trajectory IDs to snapshot lists
    """
    grouped = {}
    
    for snap in snapshots:
        traj_id = snap.snapshot.trajectory_id
        if traj_id not in grouped:
            grouped[traj_id] = []
        grouped[traj_id].append(snap)
    
    return grouped


def compute_depth_statistics(depth_labels: list[DepthLabels]) -> dict:
    """
    Compute summary statistics for depth labels.
    
    Args:
        depth_labels: List of DepthLabels objects
        
    Returns:
        Dictionary with statistics
    """
    import numpy as np
    
    if not depth_labels:
        return {}
    
    d_expert = [l.d_expert for l in depth_labels]
    d_force = [l.d_force for l in depth_labels]
    depth_gap = [l.depth_gap for l in depth_labels]
    
    return {
        "d_expert": {
            "mean": float(np.mean(d_expert)),
            "std": float(np.std(d_expert)),
            "min": int(np.min(d_expert)),
            "max": int(np.max(d_expert)),
            "median": float(np.median(d_expert)),
        },
        "d_force": {
            "mean": float(np.mean(d_force)),
            "std": float(np.std(d_force)),
            "min": int(np.min(d_force)),
            "max": int(np.max(d_force)),
            "median": float(np.median(d_force)),
        },
        "depth_gap": {
            "mean": float(np.mean(depth_gap)),
            "std": float(np.std(depth_gap)),
        },
        "local_recoverable_rate": float(np.mean([d == 0 for d in d_force])),
        "expert_recoverable_rate": float(np.mean([d == 0 for d in d_expert])),
        "n_snapshots": len(depth_labels),
    }
