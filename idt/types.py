"""
Data types for Intervention-Defined Teachability (IDT).

Trajectory representation, StepContext, and PatchSearchResult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Trajectory:
    """A single trajectory (e.g. failed rollout)."""

    traj_id: str
    task_id: Any  # goal index or string id
    instruction: str = ""
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observations and not self.actions:
            pass
        elif len(self.observations) != len(self.actions) + 1:
            # Allow observations[0] = initial obs, then obs after each action
            if len(self.observations) != len(self.actions):
                pass  # still valid if we only have actions

    @property
    def length(self) -> int:
        return len(self.actions)

    def observation_at(self, step_t: int) -> str:
        """Observation at step t (after action t-1, before action t)."""
        if step_t < len(self.observations):
            return self.observations[step_t]
        return ""

    def action_at(self, step_t: int) -> str:
        """Action taken at step t."""
        if step_t < len(self.actions):
            return self.actions[step_t]
        return ""


@dataclass
class StepContext:
    """Context at a single step t for proposer/verifier."""

    step_t: int
    instruction: str
    observation: str
    action_history: List[str]
    action_at_t: str = ""
    valid_actions: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trajectory(cls, traj: Trajectory, step_t: int) -> "StepContext":
        obs = traj.observation_at(step_t)
        history = traj.actions[:step_t] if step_t <= traj.length else traj.actions
        action_t = traj.action_at(step_t) if step_t < traj.length else ""
        valid = []
        if "steps" in traj.info and step_t < len(traj.info["steps"]):
            valid = traj.info["steps"][step_t].get("valid_actions", [])
        return cls(
            step_t=step_t,
            instruction=traj.instruction,
            observation=obs,
            action_history=history,
            action_at_t=action_t,
            valid_actions=valid,
            extra={},
        )


@dataclass
class PatchSearchResult:
    """Result of minimal patch search for one trajectory."""

    traj_id: str
    task_id: Any
    found_patch: bool
    best_patch: Optional[Dict[str, Any]] = None
    best_step: Optional[int] = None
    patch_type: Optional[str] = None
    R1: float = 0.0
    R3: float = 0.0
    R5: float = 0.0
    teachable_label: bool = False
    total_env_steps: int = 0
    total_model_calls: int = 0
    compute_counters: Dict[str, int] = field(default_factory=dict)


def trajectory_from_failure_dict(d: Dict[str, Any], traj_id: Optional[str] = None) -> Trajectory:
    """Build Trajectory from repo failure format (task_id, goal, steps with action_taken)."""
    tid = traj_id or str(d.get("task_id", ""))
    goal = d.get("goal", "")
    steps = d.get("steps", [])
    observations = []
    actions = []
    rewards: List[float] = []
    for s in steps:
        observations.append(s.get("observation", ""))
        actions.append(s.get("action_taken", ""))
        r = s.get("reward")
        if r is not None:
            rewards.append(float(r))
    if not rewards:
        rewards = [0.0] * len(actions)
    return Trajectory(
        traj_id=tid,
        task_id=d.get("task_id"),
        instruction=goal,
        observations=observations,
        actions=actions,
        rewards=rewards,
        done=True,
        info={"steps": steps, "raw": d},
    )


def load_trajectories(path: str | Path) -> List[Trajectory]:
    """Load trajectories from JSONL (one JSON object per line) or JSON array."""
    path = Path(path)
    if not path.exists():
        return []
    trajectories: List[Trajectory] = []
    with open(path, "r") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        data = json.loads(content)
        for i, item in enumerate(data):
            traj = trajectory_from_failure_dict(item, traj_id=str(i))
            trajectories.append(traj)
    else:
        for i, line in enumerate(content.splitlines()):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            traj = trajectory_from_failure_dict(item, traj_id=str(i))
            trajectories.append(traj)
    return trajectories


def _trajectory_to_dict(traj: Trajectory) -> Dict[str, Any]:
    return {
        "traj_id": traj.traj_id,
        "task_id": traj.task_id,
        "instruction": traj.instruction,
        "observations": traj.observations,
        "actions": traj.actions,
        "rewards": traj.rewards,
        "done": traj.done,
        "info": {k: v for k, v in traj.info.items() if k != "raw"},
    }


def save_trajectories(path: str | Path, trajectories: List[Trajectory]) -> None:
    """Save trajectories to JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(_trajectory_to_dict(traj)) + "\n")
