"""
Drift panel evaluation.

Fixed evaluation panels for detecting distribution drift and ensuring
consistent measurement across experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Protocol
from pathlib import Path
import json
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PanelTask:
    """A single task in an evaluation panel."""
    
    task_id: str
    quadrant: str
    difficulty: str  # easy, medium, hard
    expected_steps: int
    gold_actions: list[str]
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "quadrant": self.quadrant,
            "difficulty": self.difficulty,
            "expected_steps": self.expected_steps,
            "gold_actions": self.gold_actions,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PanelTask":
        return cls(
            task_id=data["task_id"],
            quadrant=data["quadrant"],
            difficulty=data.get("difficulty", "medium"),
            expected_steps=data.get("expected_steps", 5),
            gold_actions=data.get("gold_actions", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationPanel:
    """Fixed evaluation panel with stratified task sampling."""
    
    panel_id: str
    version: str
    created_at: str
    tasks: list[PanelTask] = field(default_factory=list)
    
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)
    
    def get_tasks_by_quadrant(self) -> dict[str, list[PanelTask]]:
        """Group tasks by quadrant."""
        groups = {}
        for task in self.tasks:
            if task.quadrant not in groups:
                groups[task.quadrant] = []
            groups[task.quadrant].append(task)
        return groups
    
    def get_tasks_by_difficulty(self) -> dict[str, list[PanelTask]]:
        """Group tasks by difficulty."""
        groups = {}
        for task in self.tasks:
            if task.difficulty not in groups:
                groups[task.difficulty] = []
            groups[task.difficulty].append(task)
        return groups
    
    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "version": self.version,
            "created_at": self.created_at,
            "n_tasks": self.n_tasks,
            "tasks": [t.to_dict() for t in self.tasks],
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EvaluationPanel":
        with open(path) as f:
            data = json.load(f)
        
        tasks = [PanelTask.from_dict(t) for t in data["tasks"]]
        return cls(
            panel_id=data["panel_id"],
            version=data["version"],
            created_at=data["created_at"],
            tasks=tasks,
        )


@dataclass
class PanelResult:
    """Result of evaluating a model on a panel."""
    
    panel_id: str
    model_id: str
    evaluated_at: str
    overall_success: float
    by_quadrant: dict[str, float]
    by_difficulty: dict[str, float]
    task_results: list[dict]
    
    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "model_id": self.model_id,
            "evaluated_at": self.evaluated_at,
            "overall_success": self.overall_success,
            "by_quadrant": self.by_quadrant,
            "by_difficulty": self.by_difficulty,
            "task_results": self.task_results,
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


def create_evaluation_panel(
    quadrant_tasks: dict[str, list[str]],
    n_per_quadrant: int = 50,
    difficulties: dict[str, int] = None,
    panel_id: str = "main",
    seed: int = 42,
) -> EvaluationPanel:
    """
    Create a fixed evaluation panel with stratified sampling.
    
    Args:
        quadrant_tasks: Dict mapping quadrant to available task IDs
        n_per_quadrant: Tasks to sample per quadrant
        difficulties: Dict mapping difficulty to count (default equal split)
        panel_id: Identifier for this panel
        seed: Random seed for reproducibility
        
    Returns:
        EvaluationPanel with selected tasks
    """
    import random
    random.seed(seed)
    
    if difficulties is None:
        difficulties = {"easy": n_per_quadrant // 3, "medium": n_per_quadrant // 3, "hard": n_per_quadrant // 3}
    
    tasks = []
    
    for quadrant, available_tasks in quadrant_tasks.items():
        sampled = random.sample(available_tasks, min(n_per_quadrant, len(available_tasks)))
        
        for i, task_id in enumerate(sampled):
            # Assign difficulty based on position
            if i < difficulties.get("easy", 0):
                difficulty = "easy"
            elif i < difficulties.get("easy", 0) + difficulties.get("medium", 0):
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            tasks.append(PanelTask(
                task_id=task_id,
                quadrant=quadrant,
                difficulty=difficulty,
                expected_steps=5,
                gold_actions=[],
            ))
    
    return EvaluationPanel(
        panel_id=panel_id,
        version="1.0",
        created_at=datetime.now().isoformat(),
        tasks=tasks,
    )


def evaluate_on_panel(
    panel: EvaluationPanel,
    policy: PolicyProtocol,
    env_factory: Callable[[], EnvironmentProtocol],
    model_id: str,
    n_rollouts: int = 3,
    max_steps: int = 15,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> PanelResult:
    """
    Evaluate a policy on an evaluation panel.
    
    Args:
        panel: Evaluation panel
        policy: Policy to evaluate
        env_factory: Environment factory
        model_id: Identifier for the model
        n_rollouts: Rollouts per task
        max_steps: Max steps per episode
        progress_callback: Progress callback
        
    Returns:
        PanelResult with detailed results
    """
    task_results = []
    total = len(panel.tasks)
    
    for i, task in enumerate(panel.tasks):
        successes = 0
        total_steps = 0
        
        for _ in range(n_rollouts):
            env = env_factory()
            state = env.reset(task.task_id)
            
            success = False
            steps = 0
            
            for step in range(max_steps):
                action = policy.select_action(state)
                state, reward, done, info = env.step(action)
                steps += 1
                
                if done:
                    success = reward > 0
                    break
            
            if success:
                successes += 1
            total_steps += steps
        
        task_results.append({
            "task_id": task.task_id,
            "quadrant": task.quadrant,
            "difficulty": task.difficulty,
            "success_rate": successes / n_rollouts,
            "mean_steps": total_steps / n_rollouts,
        })
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    # Aggregate by quadrant
    by_quadrant = {}
    for quadrant in panel.get_tasks_by_quadrant().keys():
        q_results = [r for r in task_results if r["quadrant"] == quadrant]
        by_quadrant[quadrant] = float(np.mean([r["success_rate"] for r in q_results]))
    
    # Aggregate by difficulty
    by_difficulty = {}
    for difficulty in panel.get_tasks_by_difficulty().keys():
        d_results = [r for r in task_results if r["difficulty"] == difficulty]
        by_difficulty[difficulty] = float(np.mean([r["success_rate"] for r in d_results]))
    
    overall = float(np.mean([r["success_rate"] for r in task_results]))
    
    return PanelResult(
        panel_id=panel.panel_id,
        model_id=model_id,
        evaluated_at=datetime.now().isoformat(),
        overall_success=overall,
        by_quadrant=by_quadrant,
        by_difficulty=by_difficulty,
        task_results=task_results,
    )


def detect_drift(
    baseline_result: PanelResult,
    current_result: PanelResult,
    threshold: float = 0.1,
) -> dict:
    """
    Detect distribution drift between baseline and current results.
    
    Args:
        baseline_result: Baseline panel result
        current_result: Current panel result
        threshold: Change threshold for drift detection
        
    Returns:
        Dict with drift detection results
    """
    drift_results = {
        "has_drift": False,
        "overall_change": current_result.overall_success - baseline_result.overall_success,
        "quadrant_drift": {},
        "difficulty_drift": {},
    }
    
    # Check quadrant drift
    for quadrant in baseline_result.by_quadrant.keys():
        baseline = baseline_result.by_quadrant.get(quadrant, 0)
        current = current_result.by_quadrant.get(quadrant, 0)
        change = current - baseline
        
        drift_results["quadrant_drift"][quadrant] = {
            "baseline": baseline,
            "current": current,
            "change": change,
            "is_drift": abs(change) > threshold,
        }
        
        if abs(change) > threshold:
            drift_results["has_drift"] = True
    
    # Check difficulty drift
    for difficulty in baseline_result.by_difficulty.keys():
        baseline = baseline_result.by_difficulty.get(difficulty, 0)
        current = current_result.by_difficulty.get(difficulty, 0)
        change = current - baseline
        
        drift_results["difficulty_drift"][difficulty] = {
            "baseline": baseline,
            "current": current,
            "change": change,
            "is_drift": abs(change) > threshold,
        }
        
        if abs(change) > threshold:
            drift_results["has_drift"] = True
    
    return drift_results


def run_drift_monitoring(
    panel_path: str,
    model_paths: dict[str, str],
    baseline_model_path: str,
    model_loader: Callable[[str], PolicyProtocol],
    env_factory: Callable[[], EnvironmentProtocol],
    output_dir: str,
    n_rollouts: int = 3,
) -> dict:
    """
    Run drift monitoring across all trained models.
    
    Args:
        panel_path: Path to evaluation panel JSON
        model_paths: Dict mapping model_id to path
        baseline_model_path: Path to baseline model
        model_loader: Model loader function
        env_factory: Environment factory
        output_dir: Output directory
        n_rollouts: Rollouts per task
        
    Returns:
        Drift monitoring results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load panel
    panel = EvaluationPanel.load(panel_path)
    
    # Evaluate baseline
    baseline_policy = model_loader(baseline_model_path)
    baseline_result = evaluate_on_panel(
        panel, baseline_policy, env_factory, "baseline", n_rollouts
    )
    baseline_result.save(str(output_path / "baseline_result.json"))
    
    # Evaluate all models and check for drift
    all_results = {"baseline": baseline_result.to_dict()}
    drift_summary = {}
    
    for model_id, model_path in model_paths.items():
        logger.info(f"Evaluating {model_id} on drift panel")
        
        policy = model_loader(model_path)
        result = evaluate_on_panel(panel, policy, env_factory, model_id, n_rollouts)
        result.save(str(output_path / f"result_{model_id}.json"))
        
        drift = detect_drift(baseline_result, result)
        
        all_results[model_id] = result.to_dict()
        drift_summary[model_id] = drift
    
    # Save summary
    summary = {
        "panel_id": panel.panel_id,
        "n_models": len(model_paths),
        "results": all_results,
        "drift_summary": drift_summary,
        "models_with_drift": [
            model_id for model_id, drift in drift_summary.items()
            if drift["has_drift"]
        ],
    }
    
    with open(output_path / "drift_monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def generate_panel_comparison_data(results: list[PanelResult]) -> dict:
    """
    Generate data for panel comparison visualization.
    
    Args:
        results: List of panel results
        
    Returns:
        Dict formatted for plotting
    """
    figure_data = {
        "models": [r.model_id for r in results],
        "overall": [r.overall_success for r in results],
        "by_quadrant": {
            quadrant: [r.by_quadrant.get(quadrant, 0) for r in results]
            for quadrant in results[0].by_quadrant.keys() if results
        },
        "by_difficulty": {
            diff: [r.by_difficulty.get(diff, 0) for r in results]
            for diff in results[0].by_difficulty.keys() if results
        },
    }
    
    return figure_data
