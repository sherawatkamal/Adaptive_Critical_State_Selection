"""
Transfer matrix evaluation.

Measures how training on one quadrant affects performance on other quadrants,
revealing cross-quadrant generalization patterns.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Protocol
from pathlib import Path
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result for a single source -> target transfer."""
    
    source_quadrant: str
    target_quadrant: str
    source_supervision: str
    baseline_success: float
    trained_success: float
    delta: float
    n_episodes: int
    
    @property
    def relative_improvement(self) -> float:
        if self.baseline_success == 0:
            return float('inf') if self.delta > 0 else 0.0
        return self.delta / self.baseline_success
    
    def to_dict(self) -> dict:
        return {
            "source_quadrant": self.source_quadrant,
            "target_quadrant": self.target_quadrant,
            "source_supervision": self.source_supervision,
            "baseline_success": self.baseline_success,
            "trained_success": self.trained_success,
            "delta": self.delta,
            "relative_improvement": self.relative_improvement,
            "n_episodes": self.n_episodes,
        }


@dataclass
class TransferMatrix:
    """Full 4x4 transfer matrix for a supervision type."""
    
    supervision: str
    results: list[TransferResult] = field(default_factory=list)
    
    def get_matrix(self) -> np.ndarray:
        """Return 4x4 numpy array of transfer deltas."""
        quadrants = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
        matrix = np.zeros((4, 4))
        
        for result in self.results:
            try:
                i = quadrants.index(result.source_quadrant)
                j = quadrants.index(result.target_quadrant)
                matrix[i, j] = result.delta
            except ValueError:
                continue
        
        return matrix
    
    def get_diagonal(self) -> list[float]:
        """Get diagonal (same-quadrant) improvements."""
        matrix = self.get_matrix()
        return list(np.diag(matrix))
    
    def get_off_diagonal_mean(self) -> float:
        """Get mean off-diagonal (cross-quadrant) transfer."""
        matrix = self.get_matrix()
        mask = ~np.eye(4, dtype=bool)
        return float(np.mean(matrix[mask]))
    
    def to_dict(self) -> dict:
        return {
            "supervision": self.supervision,
            "matrix": self.get_matrix().tolist(),
            "diagonal": self.get_diagonal(),
            "off_diagonal_mean": self.get_off_diagonal_mean(),
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PolicyProtocol(Protocol):
    """Protocol for policy models."""
    
    def get_action_probs(self, state: str) -> dict[str, float]: ...
    def select_action(self, state: str) -> str: ...


class EnvironmentProtocol(Protocol):
    """Protocol for environments."""
    
    def reset(self, task_id: str) -> str: ...
    def step(self, action: str) -> tuple[str, float, bool, dict]: ...


def evaluate_on_quadrant(
    policy: PolicyProtocol,
    env_factory: Callable[[], EnvironmentProtocol],
    quadrant_tasks: list[str],
    n_rollouts: int = 3,
    max_steps: int = 15,
) -> tuple[float, int]:
    """
    Evaluate policy success rate on tasks from a specific quadrant.
    
    Args:
        policy: Policy to evaluate
        env_factory: Factory to create environment instances
        quadrant_tasks: List of task IDs for this quadrant
        n_rollouts: Rollouts per task
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (success_rate, total_episodes)
    """
    successes = 0
    total = 0
    
    for task_id in quadrant_tasks:
        for _ in range(n_rollouts):
            env = env_factory()
            state = env.reset(task_id)
            
            success = False
            for _ in range(max_steps):
                action = policy.select_action(state)
                state, reward, done, info = env.step(action)
                
                if done:
                    success = reward > 0
                    break
            
            if success:
                successes += 1
            total += 1
    
    success_rate = successes / total if total > 0 else 0.0
    return success_rate, total


def compute_transfer_matrix(
    trained_models: dict[str, PolicyProtocol],
    baseline_policy: PolicyProtocol,
    quadrant_tasks: dict[str, list[str]],
    env_factory: Callable[[], EnvironmentProtocol],
    supervision_type: str = "demo",
    n_rollouts: int = 3,
    max_steps: int = 15,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TransferMatrix:
    """
    Compute full 4x4 transfer matrix.
    
    Args:
        trained_models: Dict mapping quadrant names to trained policies
        baseline_policy: Baseline (untrained) policy
        quadrant_tasks: Dict mapping quadrant names to task ID lists
        env_factory: Factory to create environment instances
        supervision_type: Supervision type used for training
        n_rollouts: Rollouts per task for evaluation
        max_steps: Maximum steps per episode
        progress_callback: Called with (completed, total) for progress
        
    Returns:
        TransferMatrix with all source -> target results
    """
    quadrants = list(quadrant_tasks.keys())
    results = []
    
    # First compute baselines for each target quadrant
    baseline_rates = {}
    for target_q in quadrants:
        rate, _ = evaluate_on_quadrant(
            baseline_policy,
            env_factory,
            quadrant_tasks[target_q],
            n_rollouts,
            max_steps,
        )
        baseline_rates[target_q] = rate
    
    total_pairs = len(quadrants) ** 2
    completed = 0
    
    # Compute transfer for each source -> target pair
    for source_q in quadrants:
        if source_q not in trained_models:
            logger.warning(f"No trained model for {source_q}")
            continue
        
        trained_policy = trained_models[source_q]
        
        for target_q in quadrants:
            trained_rate, n_episodes = evaluate_on_quadrant(
                trained_policy,
                env_factory,
                quadrant_tasks[target_q],
                n_rollouts,
                max_steps,
            )
            
            results.append(TransferResult(
                source_quadrant=source_q,
                target_quadrant=target_q,
                source_supervision=supervision_type,
                baseline_success=baseline_rates[target_q],
                trained_success=trained_rate,
                delta=trained_rate - baseline_rates[target_q],
                n_episodes=n_episodes,
            ))
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total_pairs)
    
    return TransferMatrix(supervision=supervision_type, results=results)


def analyze_transfer_patterns(matrices: list[TransferMatrix]) -> dict:
    """
    Analyze transfer patterns across supervision types.
    
    Args:
        matrices: List of TransferMatrix for different supervision types
        
    Returns:
        Dict with analysis results
    """
    analysis = {
        "supervision_comparison": {},
        "quadrant_patterns": {},
        "summary": {},
    }
    
    # Compare supervision types
    for matrix in matrices:
        analysis["supervision_comparison"][matrix.supervision] = {
            "diagonal_mean": np.mean(matrix.get_diagonal()),
            "off_diagonal_mean": matrix.get_off_diagonal_mean(),
            "best_source": None,
            "worst_source": None,
        }
        
        # Find best and worst source quadrants
        diag = matrix.get_diagonal()
        quadrants = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
        best_idx = int(np.argmax(diag))
        worst_idx = int(np.argmin(diag))
        
        analysis["supervision_comparison"][matrix.supervision]["best_source"] = quadrants[best_idx]
        analysis["supervision_comparison"][matrix.supervision]["worst_source"] = quadrants[worst_idx]
    
    # Aggregate across supervision types
    if matrices:
        all_diagonals = [np.mean(m.get_diagonal()) for m in matrices]
        all_off_diag = [m.get_off_diagonal_mean() for m in matrices]
        
        analysis["summary"] = {
            "mean_same_quadrant_improvement": float(np.mean(all_diagonals)),
            "mean_cross_quadrant_transfer": float(np.mean(all_off_diag)),
            "transfer_ratio": float(np.mean(all_off_diag) / np.mean(all_diagonals)) if np.mean(all_diagonals) != 0 else 0,
            "best_supervision_for_transfer": max(matrices, key=lambda m: m.get_off_diagonal_mean()).supervision,
        }
    
    return analysis


def generate_transfer_figure_data(matrices: list[TransferMatrix]) -> dict:
    """
    Generate data for transfer matrix heatmap visualization.
    
    Args:
        matrices: List of TransferMatrix objects
        
    Returns:
        Dict with data formatted for plotting
    """
    quadrant_labels = ["Q1\n(U+L+)", "Q2\n(U+L-)", "Q3\n(U-L-)", "Q4\n(U-L+)"]
    
    figure_data = {
        "quadrant_labels": quadrant_labels,
        "matrices": {},
    }
    
    for matrix in matrices:
        figure_data["matrices"][matrix.supervision] = {
            "values": matrix.get_matrix().tolist(),
            "diagonal": matrix.get_diagonal(),
            "off_diagonal_mean": matrix.get_off_diagonal_mean(),
        }
    
    return figure_data


def run_transfer_analysis(
    model_paths: dict[str, str],
    baseline_model_path: str,
    quadrant_tasks: dict[str, list[str]],
    env_factory: Callable[[], EnvironmentProtocol],
    model_loader: Callable[[str], PolicyProtocol],
    output_dir: str,
    n_rollouts: int = 3,
) -> dict:
    """
    Run complete transfer matrix analysis.
    
    Args:
        model_paths: Dict mapping run_id to model path
        baseline_model_path: Path to baseline model
        quadrant_tasks: Dict mapping quadrant to task list
        env_factory: Environment factory
        model_loader: Function to load policy from path
        output_dir: Output directory for results
        n_rollouts: Rollouts per task
        
    Returns:
        Analysis results dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline
    baseline = model_loader(baseline_model_path)
    
    # Group models by supervision type
    supervision_groups = {}
    for run_id, path in model_paths.items():
        parts = run_id.split("_")
        if len(parts) >= 3:
            supervision = parts[-1]
            quadrant = "_".join(parts[:-1])
            
            if supervision not in supervision_groups:
                supervision_groups[supervision] = {}
            
            supervision_groups[supervision][quadrant] = model_loader(path)
    
    # Compute transfer matrices
    matrices = []
    for supervision, models in supervision_groups.items():
        logger.info(f"Computing transfer matrix for {supervision}")
        
        matrix = compute_transfer_matrix(
            trained_models=models,
            baseline_policy=baseline,
            quadrant_tasks=quadrant_tasks,
            env_factory=env_factory,
            supervision_type=supervision,
            n_rollouts=n_rollouts,
        )
        
        matrix.save(str(output_path / f"transfer_matrix_{supervision}.json"))
        matrices.append(matrix)
    
    # Analyze patterns
    analysis = analyze_transfer_patterns(matrices)
    
    # Generate figure data
    figure_data = generate_transfer_figure_data(matrices)
    
    # Save results
    with open(output_path / "transfer_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    with open(output_path / "transfer_figure_data.json", "w") as f:
        json.dump(figure_data, f, indent=2)
    
    return {
        "matrices": [m.to_dict() for m in matrices],
        "analysis": analysis,
        "figure_data": figure_data,
    }
