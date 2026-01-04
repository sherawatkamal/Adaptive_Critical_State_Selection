"""Mechanistic evaluation: per-quadrant improvement analysis.

This module implements the core evaluation logic for measuring how well
interventions improve agent behavior within each teachability quadrant.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import numpy as np
from pathlib import Path


class Quadrant(Enum):
    """Teachability quadrants based on uncertainty x leverage."""
    Q1_UNCERTAIN_FIXABLE = "Q1"  # High U, high L_local: teachable
    Q2_UNCERTAIN_STUCK = "Q2"   # High U, low L_local: needs deeper help  
    Q3_CONFIDENT_STUCK = "Q3"   # Low U, low L_local: blind spots
    Q4_CONFIDENT_WRONG = "Q4"   # Low U, high L_local: confident errors


@dataclass
class QuadrantMetrics:
    """Metrics for a single quadrant."""
    quadrant: str
    num_snapshots: int
    
    # Pre/post success rates
    pre_success_rate: float
    post_success_rate: float
    improvement: float  # post - pre
    relative_improvement: float  # (post - pre) / pre if pre > 0
    
    # Action-level metrics
    action_match_rate: float  # How often agent takes expected action
    confidence_delta: float  # Change in agent confidence
    
    # Statistical significance
    p_value: Optional[float] = None
    is_significant: bool = False
    
    # Per-supervision breakdown
    per_supervision: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quadrant": self.quadrant,
            "num_snapshots": self.num_snapshots,
            "pre_success_rate": self.pre_success_rate,
            "post_success_rate": self.post_success_rate,
            "improvement": self.improvement,
            "relative_improvement": self.relative_improvement,
            "action_match_rate": self.action_match_rate,
            "confidence_delta": self.confidence_delta,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "per_supervision": self.per_supervision,
        }


@dataclass
class QuadrantEvaluationResult:
    """Complete per-quadrant evaluation results."""
    model_name: str
    quadrant_metrics: Dict[str, QuadrantMetrics]
    overall_improvement: float
    best_quadrant: str
    worst_quadrant: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quadrant_metrics": {k: v.to_dict() for k, v in self.quadrant_metrics.items()},
            "overall_improvement": self.overall_improvement,
            "best_quadrant": self.best_quadrant,
            "worst_quadrant": self.worst_quadrant,
        }
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "QuadrantEvaluationResult":
        with open(path) as f:
            data = json.load(f)
        
        quadrant_metrics = {}
        for k, v in data["quadrant_metrics"].items():
            quadrant_metrics[k] = QuadrantMetrics(
                quadrant=v["quadrant"],
                num_snapshots=v["num_snapshots"],
                pre_success_rate=v["pre_success_rate"],
                post_success_rate=v["post_success_rate"],
                improvement=v["improvement"],
                relative_improvement=v["relative_improvement"],
                action_match_rate=v["action_match_rate"],
                confidence_delta=v["confidence_delta"],
                p_value=v.get("p_value"),
                is_significant=v.get("is_significant", False),
                per_supervision=v.get("per_supervision", {}),
            )
        
        return cls(
            model_name=data["model_name"],
            quadrant_metrics=quadrant_metrics,
            overall_improvement=data["overall_improvement"],
            best_quadrant=data["best_quadrant"],
            worst_quadrant=data["worst_quadrant"],
        )


def compute_pre_post_success(
    snapshots: List[Dict[str, Any]],
    pre_results: List[bool],
    post_results: List[bool],
) -> Tuple[float, float]:
    """Compute pre and post intervention success rates.
    
    Args:
        snapshots: List of labeled snapshots
        pre_results: Boolean success for each snapshot before intervention
        post_results: Boolean success for each snapshot after intervention
        
    Returns:
        (pre_success_rate, post_success_rate)
    """
    if not snapshots:
        return 0.0, 0.0
    
    pre_rate = sum(pre_results) / len(pre_results)
    post_rate = sum(post_results) / len(post_results)
    
    return pre_rate, post_rate


def compute_action_match_rate(
    agent_actions: List[str],
    expected_actions: List[str],
) -> float:
    """Compute how often agent takes the expected action.
    
    Args:
        agent_actions: Actions taken by the agent
        expected_actions: Expected correct actions
        
    Returns:
        Match rate in [0, 1]
    """
    if not agent_actions:
        return 0.0
    
    matches = sum(1 for a, e in zip(agent_actions, expected_actions) if a == e)
    return matches / len(agent_actions)


def compute_confidence_delta(
    pre_confidences: List[float],
    post_confidences: List[float],
) -> float:
    """Compute mean change in agent confidence.
    
    Args:
        pre_confidences: Confidence scores before intervention
        post_confidences: Confidence scores after intervention
        
    Returns:
        Mean delta (post - pre)
    """
    if not pre_confidences or not post_confidences:
        return 0.0
    
    deltas = [post - pre for pre, post in zip(pre_confidences, post_confidences)]
    return float(np.mean(deltas))


def mcnemar_test(
    pre_success: List[bool],
    post_success: List[bool],
) -> Tuple[float, bool]:
    """Perform McNemar's test for paired binary outcomes.
    
    Tests whether the intervention significantly changes success rate.
    
    Args:
        pre_success: Success indicators before intervention
        post_success: Success indicators after intervention
        
    Returns:
        (p_value, is_significant at alpha=0.05)
    """
    if len(pre_success) != len(post_success):
        raise ValueError("Pre and post lists must have same length")
    
    # Count discordant pairs
    # b: pre=0, post=1 (improvement)
    # c: pre=1, post=0 (regression)
    b = sum(1 for pre, post in zip(pre_success, post_success) if not pre and post)
    c = sum(1 for pre, post in zip(pre_success, post_success) if pre and not post)
    
    # McNemar's test with continuity correction
    if b + c == 0:
        return 1.0, False
    
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    
    # Chi-square CDF approximation for 1 degree of freedom
    # p-value = 1 - chi2_cdf(chi2, df=1)
    # Using scipy would be better, but avoiding dependency here
    # Approximate using normal distribution
    z = np.sqrt(chi2)
    p_value = 2 * (1 - 0.5 * (1 + np.math.erf(z / np.sqrt(2))))
    
    return float(p_value), p_value < 0.05


def evaluate_per_quadrant(
    labeled_snapshots: List[Dict[str, Any]],
    pre_results: Dict[str, bool],
    post_results: Dict[str, bool],
    pre_actions: Dict[str, str],
    post_actions: Dict[str, str],
    expected_actions: Dict[str, str],
    pre_confidences: Dict[str, float],
    post_confidences: Dict[str, float],
    model_name: str = "model",
) -> QuadrantEvaluationResult:
    """Evaluate model improvement per quadrant.
    
    Args:
        labeled_snapshots: Snapshots with quadrant assignments
        pre_results: Snapshot ID -> success before intervention
        post_results: Snapshot ID -> success after intervention
        pre_actions: Snapshot ID -> action before intervention
        post_actions: Snapshot ID -> action after intervention
        expected_actions: Snapshot ID -> expected correct action
        pre_confidences: Snapshot ID -> confidence before intervention
        post_confidences: Snapshot ID -> confidence after intervention
        model_name: Name of model being evaluated
        
    Returns:
        QuadrantEvaluationResult with per-quadrant metrics
    """
    # Group snapshots by quadrant
    quadrant_groups: Dict[str, List[Dict[str, Any]]] = {
        "Q1": [], "Q2": [], "Q3": [], "Q4": []
    }
    
    for snapshot in labeled_snapshots:
        quadrant = snapshot.get("quadrant", "Q1")
        if quadrant in quadrant_groups:
            quadrant_groups[quadrant].append(snapshot)
    
    # Compute metrics per quadrant
    quadrant_metrics = {}
    
    for quadrant, snapshots in quadrant_groups.items():
        if not snapshots:
            quadrant_metrics[quadrant] = QuadrantMetrics(
                quadrant=quadrant,
                num_snapshots=0,
                pre_success_rate=0.0,
                post_success_rate=0.0,
                improvement=0.0,
                relative_improvement=0.0,
                action_match_rate=0.0,
                confidence_delta=0.0,
            )
            continue
        
        snapshot_ids = [s["snapshot_id"] for s in snapshots]
        
        # Get results for this quadrant
        q_pre = [pre_results.get(sid, False) for sid in snapshot_ids]
        q_post = [post_results.get(sid, False) for sid in snapshot_ids]
        
        pre_rate, post_rate = compute_pre_post_success(snapshots, q_pre, q_post)
        improvement = post_rate - pre_rate
        relative_improvement = improvement / pre_rate if pre_rate > 0 else 0.0
        
        # Action match rate
        q_actions = [post_actions.get(sid, "") for sid in snapshot_ids]
        q_expected = [expected_actions.get(sid, "") for sid in snapshot_ids]
        action_match = compute_action_match_rate(q_actions, q_expected)
        
        # Confidence delta
        q_pre_conf = [pre_confidences.get(sid, 0.0) for sid in snapshot_ids]
        q_post_conf = [post_confidences.get(sid, 0.0) for sid in snapshot_ids]
        conf_delta = compute_confidence_delta(q_pre_conf, q_post_conf)
        
        # Statistical significance
        p_value, is_sig = mcnemar_test(q_pre, q_post)
        
        quadrant_metrics[quadrant] = QuadrantMetrics(
            quadrant=quadrant,
            num_snapshots=len(snapshots),
            pre_success_rate=pre_rate,
            post_success_rate=post_rate,
            improvement=improvement,
            relative_improvement=relative_improvement,
            action_match_rate=action_match,
            confidence_delta=conf_delta,
            p_value=p_value,
            is_significant=is_sig,
        )
    
    # Overall improvement (weighted by quadrant size)
    total_snapshots = sum(m.num_snapshots for m in quadrant_metrics.values())
    if total_snapshots > 0:
        overall_improvement = sum(
            m.improvement * m.num_snapshots 
            for m in quadrant_metrics.values()
        ) / total_snapshots
    else:
        overall_improvement = 0.0
    
    # Find best and worst quadrants
    improvements = [(q, m.improvement) for q, m in quadrant_metrics.items() if m.num_snapshots > 0]
    if improvements:
        best_quadrant = max(improvements, key=lambda x: x[1])[0]
        worst_quadrant = min(improvements, key=lambda x: x[1])[0]
    else:
        best_quadrant = "Q1"
        worst_quadrant = "Q1"
    
    return QuadrantEvaluationResult(
        model_name=model_name,
        quadrant_metrics=quadrant_metrics,
        overall_improvement=overall_improvement,
        best_quadrant=best_quadrant,
        worst_quadrant=worst_quadrant,
    )


def compute_quadrant_improvement(
    baseline_results: QuadrantEvaluationResult,
    trained_results: QuadrantEvaluationResult,
) -> Dict[str, Dict[str, float]]:
    """Compare trained model to baseline per quadrant.
    
    Args:
        baseline_results: Evaluation results for baseline model
        trained_results: Evaluation results for trained model
        
    Returns:
        Dictionary with improvement metrics per quadrant
    """
    comparison = {}
    
    for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
        baseline = baseline_results.quadrant_metrics.get(quadrant)
        trained = trained_results.quadrant_metrics.get(quadrant)
        
        if baseline is None or trained is None:
            continue
        
        comparison[quadrant] = {
            "baseline_success": baseline.post_success_rate,
            "trained_success": trained.post_success_rate,
            "absolute_gain": trained.post_success_rate - baseline.post_success_rate,
            "baseline_action_match": baseline.action_match_rate,
            "trained_action_match": trained.action_match_rate,
            "action_match_gain": trained.action_match_rate - baseline.action_match_rate,
        }
    
    return comparison


def generate_quadrant_report(
    results: QuadrantEvaluationResult,
) -> str:
    """Generate human-readable report of per-quadrant results.
    
    Args:
        results: Quadrant evaluation results
        
    Returns:
        Formatted report string
    """
    lines = [
        f"Per-Quadrant Evaluation Report: {results.model_name}",
        "=" * 60,
        "",
    ]
    
    for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
        metrics = results.quadrant_metrics.get(quadrant)
        if metrics is None:
            continue
        
        quadrant_names = {
            "Q1": "Uncertain + Fixable (Teachable)",
            "Q2": "Uncertain + Stuck (Needs Deeper Help)",
            "Q3": "Confident + Stuck (Blind Spots)",
            "Q4": "Confident + Wrong (Confident Errors)",
        }
        
        lines.extend([
            f"{quadrant}: {quadrant_names[quadrant]}",
            "-" * 40,
            f"  Snapshots: {metrics.num_snapshots}",
            f"  Pre-success:  {metrics.pre_success_rate:.1%}",
            f"  Post-success: {metrics.post_success_rate:.1%}",
            f"  Improvement:  {metrics.improvement:+.1%}",
            f"  Action match: {metrics.action_match_rate:.1%}",
            f"  Significant:  {'Yes' if metrics.is_significant else 'No'} (p={metrics.p_value:.3f})" if metrics.p_value else "",
            "",
        ])
    
    lines.extend([
        "-" * 60,
        f"Overall Improvement: {results.overall_improvement:+.1%}",
        f"Best Quadrant:  {results.best_quadrant}",
        f"Worst Quadrant: {results.worst_quadrant}",
    ])
    
    return "\n".join(lines)
