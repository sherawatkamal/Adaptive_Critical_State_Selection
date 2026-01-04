"""
Retention evaluation.

Measures whether improvements persist after continued training,
detecting catastrophic forgetting and skill retention.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Protocol
from pathlib import Path
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetentionCheckpoint:
    """Performance at a specific training checkpoint."""
    
    checkpoint_step: int
    quadrant: str
    success_rate: float
    n_episodes: int
    timestamp: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "checkpoint_step": self.checkpoint_step,
            "quadrant": self.quadrant,
            "success_rate": self.success_rate,
            "n_episodes": self.n_episodes,
            "timestamp": self.timestamp,
        }


@dataclass
class RetentionCurve:
    """Retention curve for a single quadrant across checkpoints."""
    
    quadrant: str
    checkpoints: list[RetentionCheckpoint] = field(default_factory=list)
    
    @property
    def steps(self) -> list[int]:
        return [c.checkpoint_step for c in self.checkpoints]
    
    @property
    def success_rates(self) -> list[float]:
        return [c.success_rate for c in self.checkpoints]
    
    @property
    def peak_performance(self) -> float:
        if not self.checkpoints:
            return 0.0
        return max(c.success_rate for c in self.checkpoints)
    
    @property
    def final_performance(self) -> float:
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1].success_rate
    
    @property
    def retention_ratio(self) -> float:
        """Final / Peak performance ratio."""
        if self.peak_performance == 0:
            return 1.0
        return self.final_performance / self.peak_performance
    
    @property
    def forgetting_rate(self) -> float:
        """1 - retention_ratio."""
        return 1.0 - self.retention_ratio
    
    def to_dict(self) -> dict:
        return {
            "quadrant": self.quadrant,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "peak_performance": self.peak_performance,
            "final_performance": self.final_performance,
            "retention_ratio": self.retention_ratio,
            "forgetting_rate": self.forgetting_rate,
        }


@dataclass
class RetentionAnalysis:
    """Complete retention analysis for a training run."""
    
    run_id: str
    curves: dict[str, RetentionCurve] = field(default_factory=dict)
    
    def add_checkpoint(self, checkpoint: RetentionCheckpoint) -> None:
        """Add a checkpoint to the appropriate curve."""
        if checkpoint.quadrant not in self.curves:
            self.curves[checkpoint.quadrant] = RetentionCurve(quadrant=checkpoint.quadrant)
        self.curves[checkpoint.quadrant].checkpoints.append(checkpoint)
    
    def get_mean_retention(self) -> float:
        """Mean retention ratio across all quadrants."""
        if not self.curves:
            return 1.0
        return float(np.mean([c.retention_ratio for c in self.curves.values()]))
    
    def get_forgetting_summary(self) -> dict:
        """Summary of forgetting across quadrants."""
        return {
            quadrant: {
                "retention_ratio": curve.retention_ratio,
                "forgetting_rate": curve.forgetting_rate,
                "peak": curve.peak_performance,
                "final": curve.final_performance,
            }
            for quadrant, curve in self.curves.items()
        }
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "curves": {q: c.to_dict() for q, c in self.curves.items()},
            "mean_retention": self.get_mean_retention(),
            "forgetting_summary": self.get_forgetting_summary(),
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PolicyProtocol(Protocol):
    """Protocol for policy models."""
    
    def select_action(self, state: str) -> str: ...


class EnvironmentProtocol(Protocol):
    """Protocol for environments."""
    
    def reset(self, task_id: str) -> str: ...
    def step(self, action: str) -> tuple[str, float, bool, dict]: ...


def evaluate_checkpoint(
    checkpoint_path: str,
    model_loader: Callable[[str], PolicyProtocol],
    quadrant_tasks: dict[str, list[str]],
    env_factory: Callable[[], EnvironmentProtocol],
    checkpoint_step: int,
    n_rollouts: int = 3,
    max_steps: int = 15,
) -> list[RetentionCheckpoint]:
    """
    Evaluate a single checkpoint on all quadrants.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_loader: Function to load policy from path
        quadrant_tasks: Dict mapping quadrant to task list
        env_factory: Environment factory
        checkpoint_step: Training step of this checkpoint
        n_rollouts: Rollouts per task
        max_steps: Max steps per episode
        
    Returns:
        List of RetentionCheckpoint for each quadrant
    """
    policy = model_loader(checkpoint_path)
    checkpoints = []
    
    for quadrant, tasks in quadrant_tasks.items():
        successes = 0
        total = 0
        
        for task_id in tasks:
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
        
        checkpoints.append(RetentionCheckpoint(
            checkpoint_step=checkpoint_step,
            quadrant=quadrant,
            success_rate=success_rate,
            n_episodes=total,
        ))
    
    return checkpoints


def compute_retention_curves(
    checkpoint_dir: str,
    model_loader: Callable[[str], PolicyProtocol],
    quadrant_tasks: dict[str, list[str]],
    env_factory: Callable[[], EnvironmentProtocol],
    checkpoint_pattern: str = "checkpoint-*",
    n_rollouts: int = 3,
    max_steps: int = 15,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> RetentionAnalysis:
    """
    Compute retention curves across all checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_loader: Function to load policy
        quadrant_tasks: Dict mapping quadrant to tasks
        env_factory: Environment factory
        checkpoint_pattern: Glob pattern for checkpoint dirs
        n_rollouts: Rollouts per task
        max_steps: Max steps per episode
        progress_callback: Progress callback
        
    Returns:
        RetentionAnalysis with curves for all quadrants
    """
    from pathlib import Path
    import re
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(
        checkpoint_path.glob(checkpoint_pattern),
        key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else 0
    )
    
    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return RetentionAnalysis(run_id=checkpoint_dir)
    
    analysis = RetentionAnalysis(run_id=checkpoint_dir)
    total = len(checkpoints)
    
    for i, ckpt in enumerate(checkpoints):
        # Extract step number from checkpoint name
        match = re.search(r'(\d+)', ckpt.name)
        step = int(match.group(1)) if match else i
        
        results = evaluate_checkpoint(
            str(ckpt),
            model_loader,
            quadrant_tasks,
            env_factory,
            step,
            n_rollouts,
            max_steps,
        )
        
        for result in results:
            analysis.add_checkpoint(result)
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return analysis


def detect_catastrophic_forgetting(
    analysis: RetentionAnalysis,
    threshold: float = 0.2,
) -> dict:
    """
    Detect catastrophic forgetting in retention curves.
    
    Args:
        analysis: RetentionAnalysis with curves
        threshold: Forgetting rate threshold for "catastrophic"
        
    Returns:
        Dict with forgetting detection results
    """
    results = {
        "has_catastrophic_forgetting": False,
        "affected_quadrants": [],
        "severity": {},
    }
    
    for quadrant, curve in analysis.curves.items():
        if curve.forgetting_rate > threshold:
            results["has_catastrophic_forgetting"] = True
            results["affected_quadrants"].append(quadrant)
            results["severity"][quadrant] = curve.forgetting_rate
    
    return results


def compare_retention_across_runs(
    analyses: list[RetentionAnalysis],
) -> dict:
    """
    Compare retention patterns across training runs.
    
    Args:
        analyses: List of RetentionAnalysis from different runs
        
    Returns:
        Comparison results
    """
    comparison = {
        "runs": {},
        "best_retention": None,
        "worst_retention": None,
        "mean_retention": 0.0,
    }
    
    if not analyses:
        return comparison
    
    retentions = []
    
    for analysis in analyses:
        retention = analysis.get_mean_retention()
        retentions.append(retention)
        
        comparison["runs"][analysis.run_id] = {
            "mean_retention": retention,
            "forgetting_summary": analysis.get_forgetting_summary(),
        }
    
    comparison["mean_retention"] = float(np.mean(retentions))
    
    best_idx = int(np.argmax(retentions))
    worst_idx = int(np.argmin(retentions))
    
    comparison["best_retention"] = analyses[best_idx].run_id
    comparison["worst_retention"] = analyses[worst_idx].run_id
    
    return comparison


def run_retention_evaluation(
    training_output_dir: str,
    quadrant_tasks: dict[str, list[str]],
    env_factory: Callable[[], EnvironmentProtocol],
    model_loader: Callable[[str], PolicyProtocol],
    output_dir: str,
    n_rollouts: int = 3,
) -> dict:
    """
    Run complete retention evaluation for all trained models.
    
    Args:
        training_output_dir: Directory with training outputs
        quadrant_tasks: Dict mapping quadrant to tasks
        env_factory: Environment factory
        model_loader: Model loader function
        output_dir: Output directory
        n_rollouts: Rollouts per task
        
    Returns:
        Complete retention analysis results
    """
    from pathlib import Path
    
    training_path = Path(training_output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all run directories
    run_dirs = [d for d in training_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    analyses = []
    
    for run_dir in run_dirs:
        logger.info(f"Evaluating retention for {run_dir.name}")
        
        analysis = compute_retention_curves(
            str(run_dir),
            model_loader,
            quadrant_tasks,
            env_factory,
            n_rollouts=n_rollouts,
        )
        
        analysis.save(str(output_path / f"retention_{run_dir.name}.json"))
        analyses.append(analysis)
    
    # Compare across runs
    comparison = compare_retention_across_runs(analyses)
    
    # Check for catastrophic forgetting
    forgetting_results = {}
    for analysis in analyses:
        forgetting_results[analysis.run_id] = detect_catastrophic_forgetting(analysis)
    
    results = {
        "analyses": [a.to_dict() for a in analyses],
        "comparison": comparison,
        "forgetting_detection": forgetting_results,
    }
    
    with open(output_path / "retention_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_retention_figure_data(analyses: list[RetentionAnalysis]) -> dict:
    """
    Generate data for retention curve visualization.
    
    Args:
        analyses: List of retention analyses
        
    Returns:
        Dict formatted for plotting
    """
    figure_data = {
        "runs": {},
        "quadrant_labels": ["Q1 (U+L+)", "Q2 (U+L-)", "Q3 (U-L-)", "Q4 (U-L+)"],
    }
    
    for analysis in analyses:
        run_data = {}
        
        for quadrant, curve in analysis.curves.items():
            run_data[quadrant] = {
                "steps": curve.steps,
                "success_rates": curve.success_rates,
                "peak": curve.peak_performance,
                "final": curve.final_performance,
                "retention": curve.retention_ratio,
            }
        
        figure_data["runs"][analysis.run_id] = run_data
    
    return figure_data
