#!/usr/bin/env python3
"""
PatchSimulator: replay a failed trajectory up to a target step, then
(1) run baseline rollouts (no forced patch), and/or
(2) run patched rollouts (force a specific first action at that step).

This is designed to sit next to your existing `eef_detailed_with_diagnosis.py`.

Key difference vs EEFSimulator:
- EEFSimulator starts at a chosen step and samples actions from the agent.
- PatchSimulator can *override* the first action, which turns "step selection"
  into a fuller "step + patch" intervention.

This makes it possible to measure:
- The leverage of a step (if we change the action here, can we recover?)
- The leverage of an action patch (which alternative action yields recovery?)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence
import time


@dataclass
class ReplayOutcome:
    ok: bool
    obs: str
    info: Dict[str, Any]
    full_traj: List[Dict[str, Any]]
    reward: float = 0.0
    done: bool = False
    error: Optional[str] = None


@dataclass
class RolloutOutcome:
    success: bool
    reward: float
    sim_traj: List[Dict[str, Any]]
    full_traj: List[Dict[str, Any]]
    forced_action: Optional[str] = None
    error: Optional[str] = None
    # Compute counters
    env_steps: int = 0
    model_calls: int = 0
    # When forced action was provided but not used
    forced_skip_reason: Optional[str] = None  # 'invalid', 'forced_fallback', 'exception_fallback', None if forced was used


class PatchSimulator:
    def __init__(self, env, agent, *, max_steps: int = 50, debug: bool = False):
        self.env = env
        self.agent = agent
        self.max_steps = int(max_steps)
        self.debug = bool(debug)
        self.stats = {
            "total_rollouts": 0,
            "replay_failures": 0,
            "successful_replays": 0,
            "recoveries": 0,
            "goal_mismatches": 0,
        }

    # --------------------------
    # Replay (prefix) utilities
    # --------------------------
    def replay_prefix(self, trajectory: Dict[str, Any], target_step: int) -> ReplayOutcome:
        """
        Reset env for trajectory['task_id'] and step through actions [0..target_step-1].

        Returns:
            ReplayOutcome where (obs, info) correspond to the environment state *at* target_step
            (i.e., right before action at target_step would be taken).
        """
        task_id = trajectory.get("task_id")
        steps = trajectory.get("steps", [])
        target_step = int(target_step)

        if task_id is None:
            return ReplayOutcome(
                ok=False, obs="", info={}, full_traj=[], error="trajectory missing task_id"
            )

        obs, info = self.env.reset(task_id)
        full_traj: List[Dict[str, Any]] = []
        full_traj.append({
            "step": -1,
            "is_replay": True,
            "observation": obs,
            "action_taken": None,
            "reward": 0.0,
            "done": False,
            "info": info,
        })

        # Goal check: you already track goal mismatches in EEF; keep the same behavior.
        expected_goal = trajectory.get("goal")
        env_goal = info.get("goal")
        if expected_goal and env_goal and expected_goal != env_goal:
            self.stats["goal_mismatches"] += 1
            if self.debug:
                print("⚠️  Goal mismatch during replay:")
                print("  Expected:", expected_goal[:120])
                print("  Env:", env_goal[:120])

        # Replay prefix
        for i in range(target_step):
            if i >= len(steps):
                break
            # Support a few common schema variants.
            action = steps[i].get("action_taken")
            if action is None:
                action = steps[i].get("action")
            if action is None:
                return ReplayOutcome(
                    ok=False,
                    obs=obs,
                    info=info,
                    full_traj=full_traj,
                    error=f"missing action_taken/action at step {i}",
                )

            obs, reward, done, info = self.env.step(action)
            full_traj.append({
                "step": i,
                "is_replay": True,
                "observation": obs,
                "action_taken": action,
                "reward": float(reward),
                "done": bool(done),
                "info": info,
            })

            if done:
                # We hit terminal before target_step; can't start from target_step.
                self.stats["successful_replays"] += 1
                scaled = float(reward) * 10.0
                return ReplayOutcome(ok=False, obs=obs, info=info, full_traj=full_traj, reward=scaled, done=True,
                                    error=f"trajectory ended early during replay at step {i}")

        self.stats["successful_replays"] += 1
        return ReplayOutcome(ok=True, obs=obs, info=info, full_traj=full_traj, reward=0.0, done=False)

    # --------------------------
    # Rollout utilities
    # --------------------------
    def _step_env(self, obs: str, info: Dict[str, Any], action: str, *, step_idx: int, is_replay: bool) -> Tuple[str, float, bool, Dict[str, Any], Dict[str, Any]]:
        """
        Single env step with logging dict, returns updated (obs, reward, done, info, step_record).
        """
        next_obs, reward, done, next_info = self.env.step(action)
        step_record = {
            "step": step_idx,
            "is_replay": is_replay,
            "observation": next_obs,
            "action_taken": action,
            "reward": float(reward),
            "done": bool(done),
            "info": next_info,
        }
        return next_obs, float(reward), bool(done), next_info, step_record

    def rollout_from_state(
        self,
        trajectory: Dict[str, Any],
        target_step: int,
        *,
        method: str = "softmax",
        forced_first_action: Optional[str] = None,
        allow_any_search: bool = False,
    ) -> RolloutOutcome:
        """
        Replay to target_step, then simulate up to max_steps.

        Args:
            forced_first_action:
                If provided, we will attempt to take this as the FIRST action after replay.
                If it's invalid (not in info['valid']), we fall back to agent.get_action unless:
                    - allow_any_search=True and action starts with "search[" (then we try anyway)

        Returns:
            RolloutOutcome with sim_traj (only post-replay steps) and full_traj (replay + sim).
        """
        self.stats["total_rollouts"] += 1

        replay = self.replay_prefix(trajectory, target_step)
        if not replay.ok:
            self.stats["replay_failures"] += 1
            return RolloutOutcome(
                success=False,
                reward=float(replay.reward),
                sim_traj=[],
                full_traj=replay.full_traj,
                forced_action=forced_first_action,
                error=replay.error,
            )

        obs, info = replay.obs, replay.info
        full_traj = list(replay.full_traj)
        sim_traj: List[Dict[str, Any]] = []

        # Compute counters
        env_steps_sim = 0
        model_calls_sim = 0
        forced_skip_reason: Optional[str] = None

        # Simulate from the target state
        done = False
        total_reward = 0.0

        # Step index in the *simulation* segment (not global env step)
        for sim_step in range(self.max_steps):
            valid_acts = info.get("valid", []) or []

            if sim_step == 0 and forced_first_action is not None:
                action = forced_first_action
                # Validate action unless it is a free-form search and allow_any_search=True
                if (action not in valid_acts):
                    if allow_any_search and isinstance(action, str) and action.startswith("search["):
                        pass
                    else:
                        # Fall back to agent
                        action, _ = self.agent.get_action(obs, info, method=method)
                        model_calls_sim += 1
                        forced_skip_reason = "invalid"  # patch not in valid_acts
                action_info = {"type": "forced" if action == forced_first_action else "forced_fallback"}
                if action != forced_first_action and forced_skip_reason is None:
                    forced_skip_reason = "forced_fallback"
            else:
                action, action_info = self.agent.get_action(obs, info, method=method)
                model_calls_sim += 1

            try:
                obs, reward, done, info, record = self._step_env(obs, info, action, step_idx=sim_step, is_replay=False)
                env_steps_sim += 1
            except Exception as e:
                if forced_first_action is not None and forced_skip_reason is None:
                    forced_skip_reason = "exception_fallback"
                return RolloutOutcome(
                    success=False,
                    reward=0.0,
                    sim_traj=sim_traj,
                    full_traj=full_traj,
                    forced_action=forced_first_action,
                    error=f"env.step exception: {e}",
                    env_steps=env_steps_sim,
                    model_calls=model_calls_sim,
                    forced_skip_reason=forced_skip_reason,
                )

            record["action_info"] = action_info
            record["valid_actions"] = valid_acts
            sim_traj.append(record)
            full_traj.append(record)

            total_reward = reward  # WebShop returns terminal reward; keep last reward.

            if done:
                break

        success = bool(done and total_reward == 10.0)
        scaled_reward = float(total_reward) * 10.0

        if success:
            self.stats["recoveries"] += 1

        return RolloutOutcome(
            success=success,
            reward=scaled_reward,
            sim_traj=sim_traj,
            full_traj=full_traj,
            forced_action=forced_first_action,
            error=None,
            env_steps=env_steps_sim,
            model_calls=model_calls_sim,
            forced_skip_reason=forced_skip_reason,
        )
