"""
Failure detector for analyzing student failures and identifying teachable moments.

This module provides sophisticated failure analysis including:
- Failure pattern detection
- Root cause analysis
- Teachability scoring (how easy is this failure to correct?)
- Gap analysis (student vs teacher comparison)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import Counter
import re

from .student_rollout import FailureEvent, FailureType, RolloutResult

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """A detected pattern of failures across trajectories."""
    
    pattern_id: str
    failure_type: FailureType
    description: str
    
    # Occurrences
    count: int
    trajectories: list[str]  # trajectory_ids
    
    # Common characteristics
    common_states: list[str] = field(default_factory=list)
    common_actions: list[str] = field(default_factory=list)
    
    # Teachability assessment
    teachability_score: float = 0.5  # 0=hard to teach, 1=easy to teach
    suggested_intervention: str = ""
    
    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "failure_type": self.failure_type.value,
            "description": self.description,
            "count": self.count,
            "trajectories": self.trajectories,
            "common_states": self.common_states,
            "common_actions": self.common_actions,
            "teachability_score": self.teachability_score,
            "suggested_intervention": self.suggested_intervention,
        }


@dataclass 
class TeachableGap:
    """A gap where teacher succeeds but student fails."""
    
    task_id: str
    
    # Student failure info
    student_failure_type: FailureType
    student_failure_step: int
    student_state_at_failure: str
    student_action_at_failure: str
    
    # Teacher success info
    teacher_action_at_same_point: str
    teacher_total_steps: int
    teacher_reasoning: Optional[str] = None
    
    # Gap analysis
    action_mismatch: bool = True
    strategy_difference: str = ""
    
    # Teaching potential
    teachability_score: float = 0.5
    supervision_recommendation: str = "hint"  # hint, demo, or contrast
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "student_failure_type": self.student_failure_type.value,
            "student_failure_step": self.student_failure_step,
            "student_state_at_failure": self.student_state_at_failure,
            "student_action_at_failure": self.student_action_at_failure,
            "teacher_action_at_same_point": self.teacher_action_at_same_point,
            "teacher_total_steps": self.teacher_total_steps,
            "teacher_reasoning": self.teacher_reasoning,
            "action_mismatch": self.action_mismatch,
            "strategy_difference": self.strategy_difference,
            "teachability_score": self.teachability_score,
            "supervision_recommendation": self.supervision_recommendation,
        }


@dataclass
class FailureDetectorConfig:
    """Configuration for failure detector."""
    
    # Pattern detection
    min_pattern_count: int = 3          # Minimum occurrences to form pattern
    state_similarity_threshold: float = 0.7
    
    # Teachability scoring
    weight_teacher_success: float = 0.4  # Teacher succeeds → more teachable
    weight_low_steps: float = 0.3        # Few steps to fix → more teachable
    weight_common_failure: float = 0.3   # Common failure → higher priority
    
    # Intervention mapping
    intervention_rules: dict = field(default_factory=lambda: {
        "stuck_loop": "contrast",       # Show good vs bad to break loop
        "wrong_action": "demo",         # Show correct action
        "timeout": "hint",              # Guide toward goal
        "confusion": "hint",            # Reduce uncertainty
        "suboptimal": "contrast",       # Show efficient alternative
    })


class FailureDetector:
    """
    Analyzes failures to identify teachable moments.
    
    Main capabilities:
    1. Cluster similar failures into patterns
    2. Compare student failures with teacher success
    3. Score teachability of each failure
    4. Recommend intervention type
    """
    
    def __init__(self, config: Optional[FailureDetectorConfig] = None):
        self.config = config or FailureDetectorConfig()
        
    def analyze_failures(
        self,
        student_results: list[RolloutResult],
        teacher_results: Optional[list[dict]] = None,
    ) -> dict:
        """
        Comprehensive failure analysis.
        
        Args:
            student_results: Student rollout results
            teacher_results: Optional teacher results for comparison
            
        Returns:
            Analysis dict with patterns, gaps, and recommendations
        """
        # Collect all failures
        all_failures = []
        for result in student_results:
            all_failures.extend(result.failures)
        
        # Detect patterns
        patterns = self._detect_patterns(all_failures)
        
        # Compare with teacher if available
        gaps = []
        if teacher_results:
            gaps = self._find_teachable_gaps(student_results, teacher_results)
        
        # Score teachability
        for failure in all_failures:
            failure_dict = failure.to_dict()
            failure_dict["teachability_score"] = self._score_teachability(
                failure, patterns, gaps
            )
            failure_dict["supervision_recommendation"] = self._recommend_supervision(
                failure, gaps
            )
        
        # Aggregate statistics
        stats = self._compute_statistics(all_failures, patterns, gaps)
        
        return {
            "summary": stats,
            "patterns": [p.to_dict() for p in patterns],
            "teachable_gaps": [g.to_dict() for g in gaps],
            "all_failures": [f.to_dict() for f in all_failures],
            "recommendations": self._generate_recommendations(patterns, gaps),
        }
    
    def _detect_patterns(self, failures: list[FailureEvent]) -> list[FailurePattern]:
        """Cluster failures into patterns."""
        patterns = []
        
        # Group by failure type first
        by_type = {}
        for f in failures:
            ft = f.failure_type
            if ft not in by_type:
                by_type[ft] = []
            by_type[ft].append(f)
        
        pattern_id = 0
        for failure_type, type_failures in by_type.items():
            if len(type_failures) < self.config.min_pattern_count:
                # Still create pattern but mark as rare
                pass
            
            # Further cluster by common actions/states
            action_clusters = self._cluster_by_actions(type_failures)
            
            for cluster_key, cluster_failures in action_clusters.items():
                pattern_id += 1
                
                # Find common characteristics
                common_states = self._find_common_states(cluster_failures)
                common_actions = self._find_common_actions(cluster_failures)
                
                # Generate description
                description = self._generate_pattern_description(
                    failure_type, common_states, common_actions
                )
                
                pattern = FailurePattern(
                    pattern_id=f"pattern_{pattern_id:03d}",
                    failure_type=failure_type,
                    description=description,
                    count=len(cluster_failures),
                    trajectories=[f.trajectory_id for f in cluster_failures],
                    common_states=common_states[:5],
                    common_actions=common_actions[:5],
                    teachability_score=self._pattern_teachability(cluster_failures),
                    suggested_intervention=self.config.intervention_rules.get(
                        failure_type.value, "hint"
                    ),
                )
                patterns.append(pattern)
        
        # Sort by count (most common first)
        patterns.sort(key=lambda p: p.count, reverse=True)
        
        return patterns
    
    def _cluster_by_actions(
        self, failures: list[FailureEvent]
    ) -> dict[str, list[FailureEvent]]:
        """Cluster failures by similar actions."""
        clusters = {}
        
        for f in failures:
            # Create cluster key from action pattern
            action = f.action_taken.lower()
            
            # Normalize action
            action_type = action.split("[")[0] if "[" in action else action
            
            if action_type not in clusters:
                clusters[action_type] = []
            clusters[action_type].append(f)
        
        return clusters
    
    def _find_common_states(self, failures: list[FailureEvent]) -> list[str]:
        """Find common state patterns across failures."""
        # Extract keywords from states
        all_keywords = []
        for f in failures:
            keywords = self._extract_keywords(f.state)
            all_keywords.extend(keywords)
        
        # Find most common
        counter = Counter(all_keywords)
        return [kw for kw, count in counter.most_common(10)]
    
    def _find_common_actions(self, failures: list[FailureEvent]) -> list[str]:
        """Find common action patterns."""
        actions = [f.action_taken for f in failures]
        counter = Counter(actions)
        return [a for a, count in counter.most_common(5)]
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _generate_pattern_description(
        self,
        failure_type: FailureType,
        common_states: list[str],
        common_actions: list[str],
    ) -> str:
        """Generate human-readable pattern description."""
        base_descriptions = {
            FailureType.STUCK_LOOP: "Student gets stuck repeating",
            FailureType.WRONG_ACTION: "Student takes incorrect action",
            FailureType.TIMEOUT: "Student runs out of steps",
            FailureType.CONFUSION: "Student shows high uncertainty",
            FailureType.SUBOPTIMAL: "Student succeeds inefficiently",
            FailureType.EARLY_TERMINATION: "Student stops prematurely",
        }
        
        base = base_descriptions.get(failure_type, "Student fails")
        
        if common_actions:
            action_part = f" ({common_actions[0]})"
        else:
            action_part = ""
        
        if common_states:
            state_part = f" when: {', '.join(common_states[:3])}"
        else:
            state_part = ""
        
        return f"{base}{action_part}{state_part}"
    
    def _pattern_teachability(self, failures: list[FailureEvent]) -> float:
        """Score how teachable a pattern is."""
        if not failures:
            return 0.5
        
        # Higher teachability if:
        # 1. Failures happen at similar points (consistent)
        steps = [f.step_idx for f in failures]
        step_variance = self._variance(steps)
        consistency_score = 1.0 / (1.0 + step_variance)
        
        # 2. Model confidence was low (student knows it's unsure)
        confidences = [f.model_confidence for f in failures if f.model_confidence]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            uncertainty_score = 1.0 - avg_confidence  # Low confidence = more teachable
        else:
            uncertainty_score = 0.5
        
        # 3. Enough examples to learn from
        count_score = min(len(failures) / 10, 1.0)
        
        return (consistency_score + uncertainty_score + count_score) / 3
    
    def _variance(self, values: list) -> float:
        """Compute variance of a list."""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _find_teachable_gaps(
        self,
        student_results: list[RolloutResult],
        teacher_results: list[dict],
    ) -> list[TeachableGap]:
        """Find gaps where teacher succeeds but student fails."""
        gaps = []
        
        # Index teacher results by task_id
        teacher_by_task = {r.get("task_id"): r for r in teacher_results}
        
        for student in student_results:
            if student.success:
                continue  # No gap if student succeeded
            
            teacher = teacher_by_task.get(student.task_id)
            if not teacher or not teacher.get("success"):
                continue  # No gap if teacher also failed
            
            # Found a teachable gap
            for failure in student.failures:
                # Find what teacher did at same step
                teacher_action = ""
                teacher_reasoning = ""
                
                if failure.step_idx < len(teacher.get("actions", [])):
                    teacher_action = teacher["actions"][failure.step_idx]
                if failure.step_idx < len(teacher.get("reasoning", [])):
                    teacher_reasoning = teacher["reasoning"][failure.step_idx]
                
                gap = TeachableGap(
                    task_id=student.task_id,
                    student_failure_type=failure.failure_type,
                    student_failure_step=failure.step_idx,
                    student_state_at_failure=failure.state,
                    student_action_at_failure=failure.action_taken,
                    teacher_action_at_same_point=teacher_action,
                    teacher_total_steps=teacher.get("n_steps", 0),
                    teacher_reasoning=teacher_reasoning,
                    action_mismatch=failure.action_taken != teacher_action,
                    strategy_difference=self._analyze_strategy_diff(
                        failure.action_taken, teacher_action
                    ),
                    teachability_score=self._score_gap_teachability(failure, teacher),
                    supervision_recommendation=self._recommend_for_gap(failure, teacher),
                )
                gaps.append(gap)
        
        return gaps
    
    def _analyze_strategy_diff(self, student_action: str, teacher_action: str) -> str:
        """Analyze difference in strategy between student and teacher."""
        if not student_action or not teacher_action:
            return "missing_action"
        
        student_type = student_action.split("[")[0] if "[" in student_action else student_action
        teacher_type = teacher_action.split("[")[0] if "[" in teacher_action else teacher_action
        
        if student_type == teacher_type:
            return "same_type_different_target"
        else:
            return f"different_strategy_{student_type}_vs_{teacher_type}"
    
    def _score_gap_teachability(self, failure: FailureEvent, teacher: dict) -> float:
        """Score how teachable a specific gap is."""
        score = 0.5
        
        # Teacher succeeds quickly → easier to teach
        teacher_steps = teacher.get("n_steps", 30)
        if teacher_steps < 10:
            score += 0.2
        elif teacher_steps < 20:
            score += 0.1
        
        # Student was uncertain → more receptive to teaching
        if failure.model_confidence and failure.model_confidence < 0.5:
            score += 0.15
        
        # Early failure → more room to improve
        if failure.step_idx < 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _recommend_for_gap(self, failure: FailureEvent, teacher: dict) -> str:
        """Recommend supervision type for a gap."""
        failure_type = failure.failure_type
        
        # Default based on failure type
        default = self.config.intervention_rules.get(failure_type.value, "hint")
        
        # Override based on gap characteristics
        if failure.model_confidence and failure.model_confidence < 0.3:
            # Very uncertain → hint to guide
            return "hint"
        
        if failure_type == FailureType.STUCK_LOOP:
            # Show contrast to break loop
            return "contrast"
        
        # If teacher reasoning available and clear, demo is good
        if teacher.get("reasoning") and len(teacher["reasoning"]) > 0:
            return "demo"
        
        return default
    
    def _score_teachability(
        self,
        failure: FailureEvent,
        patterns: list[FailurePattern],
        gaps: list[TeachableGap],
    ) -> float:
        """Score overall teachability of a failure."""
        score = 0.5
        
        # Check if part of a pattern
        for pattern in patterns:
            if failure.trajectory_id in pattern.trajectories:
                score = max(score, pattern.teachability_score)
                break
        
        # Check if has a teachable gap
        for gap in gaps:
            if gap.task_id == failure.trajectory_id.split("_")[-1]:
                score = max(score, gap.teachability_score)
                break
        
        return score
    
    def _recommend_supervision(
        self,
        failure: FailureEvent,
        gaps: list[TeachableGap],
    ) -> str:
        """Recommend supervision type for a failure."""
        # Check gaps first
        for gap in gaps:
            if gap.student_failure_step == failure.step_idx:
                return gap.supervision_recommendation
        
        # Default based on failure type
        return self.config.intervention_rules.get(
            failure.failure_type.value, "hint"
        )
    
    def _compute_statistics(
        self,
        failures: list[FailureEvent],
        patterns: list[FailurePattern],
        gaps: list[TeachableGap],
    ) -> dict:
        """Compute summary statistics."""
        if not failures:
            return {"total_failures": 0}
        
        # Failure type distribution
        type_dist = Counter(f.failure_type.value for f in failures)
        
        # Teachability distribution
        teachable_high = sum(1 for f in failures if hasattr(f, 'teachability_score') and f.teachability_score > 0.7)
        teachable_medium = sum(1 for f in failures if hasattr(f, 'teachability_score') and 0.4 <= f.teachability_score <= 0.7)
        teachable_low = sum(1 for f in failures if hasattr(f, 'teachability_score') and f.teachability_score < 0.4)
        
        return {
            "total_failures": len(failures),
            "unique_patterns": len(patterns),
            "teachable_gaps": len(gaps),
            "failure_type_distribution": dict(type_dist),
            "teachability_distribution": {
                "high": teachable_high,
                "medium": teachable_medium,
                "low": teachable_low,
            },
            "avg_failure_step": sum(f.step_idx for f in failures) / len(failures),
            "avg_confidence_at_failure": sum(
                f.model_confidence for f in failures if f.model_confidence
            ) / max(1, sum(1 for f in failures if f.model_confidence)),
        }
    
    def _generate_recommendations(
        self,
        patterns: list[FailurePattern],
        gaps: list[TeachableGap],
    ) -> list[dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Top patterns to address
        for pattern in patterns[:5]:
            rec = {
                "priority": "high" if pattern.count >= 10 else "medium",
                "pattern": pattern.pattern_id,
                "description": pattern.description,
                "count": pattern.count,
                "suggested_intervention": pattern.suggested_intervention,
                "teachability": pattern.teachability_score,
                "action": f"Create {pattern.suggested_intervention} supervision for {pattern.count} instances",
            }
            recommendations.append(rec)
        
        # Gap-based recommendations
        gap_types = Counter(g.supervision_recommendation for g in gaps)
        for supervision_type, count in gap_types.most_common():
            rec = {
                "priority": "high",
                "type": "teachable_gap",
                "supervision": supervision_type,
                "count": count,
                "action": f"Generate {count} {supervision_type} examples from teacher-student gaps",
            }
            recommendations.append(rec)
        
        return recommendations
    
    def save_analysis(self, analysis: dict, output_path: str):
        """Save failure analysis to JSON."""
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved failure analysis to {output_path}")
