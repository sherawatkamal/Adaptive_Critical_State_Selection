"""Stuckness detection and metrics.

Identifies episodes where agents get stuck in loops or make no progress,
and measures how well interventions reduce stuckness.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import json
import numpy as np
from pathlib import Path


@dataclass
class StuckPattern:
    """A detected stuckness pattern in an episode."""
    pattern_type: str  # "action_loop", "state_loop", "no_progress"
    start_step: int
    end_step: int
    duration: int
    repeated_element: Optional[str] = None  # The action or state being repeated
    repetition_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration": self.duration,
            "repeated_element": self.repeated_element,
            "repetition_count": self.repetition_count,
        }


@dataclass
class EpisodeStuckness:
    """Stuckness analysis for a single episode."""
    episode_id: str
    is_stuck: bool
    total_stuck_steps: int
    stuck_ratio: float  # stuck_steps / total_steps
    patterns: List[StuckPattern] = field(default_factory=list)
    
    # Severity metrics
    max_loop_length: int = 0
    num_distinct_patterns: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "is_stuck": self.is_stuck,
            "total_stuck_steps": self.total_stuck_steps,
            "stuck_ratio": self.stuck_ratio,
            "patterns": [p.to_dict() for p in self.patterns],
            "max_loop_length": self.max_loop_length,
            "num_distinct_patterns": self.num_distinct_patterns,
        }


@dataclass
class StucknessMetrics:
    """Aggregated stuckness metrics across episodes."""
    num_episodes: int
    stuck_episode_count: int
    stuck_episode_rate: float
    mean_stuck_ratio: float
    mean_stuck_steps: float
    
    # Pattern breakdown
    action_loop_count: int = 0
    state_loop_count: int = 0
    no_progress_count: int = 0
    
    # Per-episode details
    episode_results: List[EpisodeStuckness] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "stuck_episode_count": self.stuck_episode_count,
            "stuck_episode_rate": self.stuck_episode_rate,
            "mean_stuck_ratio": self.mean_stuck_ratio,
            "mean_stuck_steps": self.mean_stuck_steps,
            "action_loop_count": self.action_loop_count,
            "state_loop_count": self.state_loop_count,
            "no_progress_count": self.no_progress_count,
        }
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["episode_results"] = [e.to_dict() for e in self.episode_results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "StucknessMetrics":
        with open(path) as f:
            data = json.load(f)
        
        episode_results = []
        for e in data.get("episode_results", []):
            patterns = [StuckPattern(**p) for p in e.get("patterns", [])]
            episode_results.append(EpisodeStuckness(
                episode_id=e["episode_id"],
                is_stuck=e["is_stuck"],
                total_stuck_steps=e["total_stuck_steps"],
                stuck_ratio=e["stuck_ratio"],
                patterns=patterns,
                max_loop_length=e.get("max_loop_length", 0),
                num_distinct_patterns=e.get("num_distinct_patterns", 0),
            ))
        
        return cls(
            num_episodes=data["num_episodes"],
            stuck_episode_count=data["stuck_episode_count"],
            stuck_episode_rate=data["stuck_episode_rate"],
            mean_stuck_ratio=data["mean_stuck_ratio"],
            mean_stuck_steps=data["mean_stuck_steps"],
            action_loop_count=data.get("action_loop_count", 0),
            state_loop_count=data.get("state_loop_count", 0),
            no_progress_count=data.get("no_progress_count", 0),
            episode_results=episode_results,
        )


def detect_action_loops(
    actions: List[str],
    min_loop_length: int = 2,
    min_repetitions: int = 3,
) -> List[StuckPattern]:
    """Detect repeated action sequences.
    
    Args:
        actions: Sequence of actions taken
        min_loop_length: Minimum length of repeated sequence
        min_repetitions: Minimum number of repetitions to count as stuck
        
    Returns:
        List of detected action loop patterns
    """
    patterns = []
    n = len(actions)
    
    # Check for simple single-action repetition
    for i in range(n):
        if i + min_repetitions > n:
            break
        
        action = actions[i]
        count = 1
        j = i + 1
        while j < n and actions[j] == action:
            count += 1
            j += 1
        
        if count >= min_repetitions:
            patterns.append(StuckPattern(
                pattern_type="action_loop",
                start_step=i,
                end_step=j - 1,
                duration=count,
                repeated_element=action,
                repetition_count=count,
            ))
    
    # Check for multi-action sequence repetition
    for loop_len in range(min_loop_length, min(n // min_repetitions + 1, 5)):
        for start in range(n - loop_len * min_repetitions + 1):
            pattern = tuple(actions[start:start + loop_len])
            
            # Count how many times this pattern repeats
            reps = 1
            pos = start + loop_len
            while pos + loop_len <= n:
                if tuple(actions[pos:pos + loop_len]) == pattern:
                    reps += 1
                    pos += loop_len
                else:
                    break
            
            if reps >= min_repetitions:
                patterns.append(StuckPattern(
                    pattern_type="action_loop",
                    start_step=start,
                    end_step=start + reps * loop_len - 1,
                    duration=reps * loop_len,
                    repeated_element=" -> ".join(pattern),
                    repetition_count=reps,
                ))
    
    return patterns


def detect_state_loops(
    observations: List[str],
    min_repetitions: int = 3,
) -> List[StuckPattern]:
    """Detect repeated observation states.
    
    Args:
        observations: Sequence of observations
        min_repetitions: Minimum repetitions to count as stuck
        
    Returns:
        List of detected state loop patterns
    """
    patterns = []
    
    # Use observation hash for comparison (observations can be long)
    obs_hashes = [hash(obs) for obs in observations]
    
    for i in range(len(obs_hashes)):
        if i + min_repetitions > len(obs_hashes):
            break
        
        obs_hash = obs_hashes[i]
        count = 1
        j = i + 1
        while j < len(obs_hashes) and obs_hashes[j] == obs_hash:
            count += 1
            j += 1
        
        if count >= min_repetitions:
            # Use truncated observation as repeated element
            truncated = observations[i][:100] + "..." if len(observations[i]) > 100 else observations[i]
            patterns.append(StuckPattern(
                pattern_type="state_loop",
                start_step=i,
                end_step=j - 1,
                duration=count,
                repeated_element=truncated,
                repetition_count=count,
            ))
    
    return patterns


def detect_no_progress(
    rewards: List[float],
    window_size: int = 5,
    min_duration: int = 5,
) -> List[StuckPattern]:
    """Detect periods of zero progress (no positive rewards).
    
    Args:
        rewards: Sequence of step rewards
        window_size: Window for checking progress
        min_duration: Minimum steps of no progress
        
    Returns:
        List of detected no-progress patterns
    """
    patterns = []
    n = len(rewards)
    
    # Find runs of zero or negative rewards
    start = None
    for i in range(n):
        if rewards[i] <= 0:
            if start is None:
                start = i
        else:
            if start is not None:
                duration = i - start
                if duration >= min_duration:
                    patterns.append(StuckPattern(
                        pattern_type="no_progress",
                        start_step=start,
                        end_step=i - 1,
                        duration=duration,
                        repeated_element="zero_reward",
                        repetition_count=duration,
                    ))
                start = None
    
    # Check if stuck at the end
    if start is not None:
        duration = n - start
        if duration >= min_duration:
            patterns.append(StuckPattern(
                pattern_type="no_progress",
                start_step=start,
                end_step=n - 1,
                duration=duration,
                repeated_element="zero_reward",
                repetition_count=duration,
            ))
    
    return patterns


def analyze_episode_stuckness(
    episode_id: str,
    actions: List[str],
    observations: List[str],
    rewards: List[float],
    stuck_threshold: float = 0.3,  # Ratio of stuck steps to consider episode stuck
) -> EpisodeStuckness:
    """Analyze stuckness for a single episode.
    
    Args:
        episode_id: Unique episode identifier
        actions: Sequence of actions
        observations: Sequence of observations
        rewards: Sequence of rewards
        stuck_threshold: Minimum stuck ratio to mark episode as stuck
        
    Returns:
        EpisodeStuckness with detected patterns
    """
    all_patterns = []
    
    # Detect different types of stuckness
    action_patterns = detect_action_loops(actions)
    state_patterns = detect_state_loops(observations)
    progress_patterns = detect_no_progress(rewards)
    
    all_patterns.extend(action_patterns)
    all_patterns.extend(state_patterns)
    all_patterns.extend(progress_patterns)
    
    # Compute stuck steps (union of all pattern spans)
    total_steps = len(actions)
    stuck_steps = set()
    
    for pattern in all_patterns:
        for step in range(pattern.start_step, pattern.end_step + 1):
            stuck_steps.add(step)
    
    total_stuck_steps = len(stuck_steps)
    stuck_ratio = total_stuck_steps / total_steps if total_steps > 0 else 0.0
    
    # Compute severity metrics
    max_loop_length = max((p.duration for p in all_patterns), default=0)
    num_distinct_patterns = len(all_patterns)
    
    return EpisodeStuckness(
        episode_id=episode_id,
        is_stuck=stuck_ratio >= stuck_threshold,
        total_stuck_steps=total_stuck_steps,
        stuck_ratio=stuck_ratio,
        patterns=all_patterns,
        max_loop_length=max_loop_length,
        num_distinct_patterns=num_distinct_patterns,
    )


def detect_stuck_episodes(
    episodes: List[Dict[str, Any]],
    stuck_threshold: float = 0.3,
) -> List[EpisodeStuckness]:
    """Detect stuckness across multiple episodes.
    
    Args:
        episodes: List of episodes, each with keys:
            - episode_id: str
            - actions: List[str]
            - observations: List[str]
            - rewards: List[float]
        stuck_threshold: Minimum stuck ratio to mark as stuck
        
    Returns:
        List of EpisodeStuckness results
    """
    results = []
    
    for episode in episodes:
        result = analyze_episode_stuckness(
            episode_id=episode["episode_id"],
            actions=episode["actions"],
            observations=episode["observations"],
            rewards=episode["rewards"],
            stuck_threshold=stuck_threshold,
        )
        results.append(result)
    
    return results


def compute_stuckness_metrics(
    episode_results: List[EpisodeStuckness],
) -> StucknessMetrics:
    """Compute aggregated stuckness metrics.
    
    Args:
        episode_results: List of per-episode stuckness analysis
        
    Returns:
        StucknessMetrics with aggregated statistics
    """
    if not episode_results:
        return StucknessMetrics(
            num_episodes=0,
            stuck_episode_count=0,
            stuck_episode_rate=0.0,
            mean_stuck_ratio=0.0,
            mean_stuck_steps=0.0,
        )
    
    num_episodes = len(episode_results)
    stuck_episodes = [e for e in episode_results if e.is_stuck]
    stuck_episode_count = len(stuck_episodes)
    stuck_episode_rate = stuck_episode_count / num_episodes
    
    mean_stuck_ratio = np.mean([e.stuck_ratio for e in episode_results])
    mean_stuck_steps = np.mean([e.total_stuck_steps for e in episode_results])
    
    # Count pattern types
    action_loop_count = 0
    state_loop_count = 0
    no_progress_count = 0
    
    for episode in episode_results:
        for pattern in episode.patterns:
            if pattern.pattern_type == "action_loop":
                action_loop_count += 1
            elif pattern.pattern_type == "state_loop":
                state_loop_count += 1
            elif pattern.pattern_type == "no_progress":
                no_progress_count += 1
    
    return StucknessMetrics(
        num_episodes=num_episodes,
        stuck_episode_count=stuck_episode_count,
        stuck_episode_rate=stuck_episode_rate,
        mean_stuck_ratio=float(mean_stuck_ratio),
        mean_stuck_steps=float(mean_stuck_steps),
        action_loop_count=action_loop_count,
        state_loop_count=state_loop_count,
        no_progress_count=no_progress_count,
        episode_results=episode_results,
    )


def compare_stuckness(
    baseline_metrics: StucknessMetrics,
    trained_metrics: StucknessMetrics,
) -> Dict[str, float]:
    """Compare stuckness metrics between baseline and trained model.
    
    Args:
        baseline_metrics: Stuckness metrics for baseline
        trained_metrics: Stuckness metrics for trained model
        
    Returns:
        Dictionary with comparison metrics
    """
    return {
        "baseline_stuck_rate": baseline_metrics.stuck_episode_rate,
        "trained_stuck_rate": trained_metrics.stuck_episode_rate,
        "stuck_rate_reduction": baseline_metrics.stuck_episode_rate - trained_metrics.stuck_episode_rate,
        "stuck_rate_reduction_pct": (
            (baseline_metrics.stuck_episode_rate - trained_metrics.stuck_episode_rate) 
            / baseline_metrics.stuck_episode_rate * 100
            if baseline_metrics.stuck_episode_rate > 0 else 0.0
        ),
        "baseline_mean_stuck_ratio": baseline_metrics.mean_stuck_ratio,
        "trained_mean_stuck_ratio": trained_metrics.mean_stuck_ratio,
        "stuck_ratio_reduction": baseline_metrics.mean_stuck_ratio - trained_metrics.mean_stuck_ratio,
    }
