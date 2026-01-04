"""
Hint generation for teachable moments.

Generates diagnostic hints using GPT-4o to explain why an action
was wrong and what the agent should do instead.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import json
import logging

logger = logging.getLogger(__name__)


# System prompt for hint generation
HINT_SYSTEM_PROMPT = """You are an expert teaching assistant for an AI agent learning to shop on an e-commerce website.

Your task is to analyze the agent's mistake and provide a concise, actionable hint that explains:
1. What error pattern the agent exhibited (e.g., premature action, wrong interpretation, missed information)
2. What the agent should have noticed or done differently
3. A brief principle the agent can apply to similar situations

Keep your hint under 100 words. Focus on the reasoning process, not just the correct action.
Do not be condescending. Be direct and helpful."""


# Template for hint request
HINT_REQUEST_TEMPLATE = """## Task
{task_description}

## Current State
{state_text}

## Agent's Wrong Action
{wrong_action}

## Correct Action
{correct_action}

## Available Actions
{available_actions}

Please provide a teaching hint that explains the agent's error and how to improve."""


@dataclass
class HintRequest:
    """Request for hint generation."""
    
    task_description: str
    state_text: str
    wrong_action: str
    correct_action: str
    available_actions: list[str]
    snapshot_id: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert to prompt string."""
        return HINT_REQUEST_TEMPLATE.format(
            task_description=self.task_description,
            state_text=self.state_text,
            wrong_action=self.wrong_action,
            correct_action=self.correct_action,
            available_actions=", ".join(self.available_actions),
        )
    
    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "state_text": self.state_text,
            "wrong_action": self.wrong_action,
            "correct_action": self.correct_action,
            "available_actions": self.available_actions,
            "snapshot_id": self.snapshot_id,
        }


@dataclass
class GeneratedHint:
    """Generated teaching hint."""
    
    hint_text: str
    error_type: Optional[str] = None
    principle: Optional[str] = None
    snapshot_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "hint_text": self.hint_text,
            "error_type": self.error_type,
            "principle": self.principle,
            "snapshot_id": self.snapshot_id,
        }


class HintGenerator:
    """Generates teaching hints using teacher model."""
    
    def __init__(self, client):
        """
        Initialize hint generator.
        
        Args:
            client: TeacherClient instance
        """
        self.client = client
        self.generated_count = 0
    
    def generate_hint(self, request: HintRequest) -> GeneratedHint:
        """
        Generate a teaching hint for a snapshot.
        
        Args:
            request: HintRequest with snapshot details
            
        Returns:
            GeneratedHint with teaching content
        """
        prompt = request.to_prompt()
        
        response = self.client.generate(
            prompt=prompt,
            system_prompt=HINT_SYSTEM_PROMPT,
        )
        
        self.generated_count += 1
        
        # Parse response for structured fields
        hint = GeneratedHint(
            hint_text=response,
            snapshot_id=request.snapshot_id,
        )
        
        # Try to extract error type from hint
        hint.error_type = self._extract_error_type(response)
        hint.principle = self._extract_principle(response)
        
        return hint
    
    def _extract_error_type(self, text: str) -> Optional[str]:
        """Extract error type from hint text."""
        error_patterns = [
            "premature",
            "wrong interpretation",
            "missed information",
            "incorrect assumption",
            "oversight",
            "misread",
        ]
        
        text_lower = text.lower()
        for pattern in error_patterns:
            if pattern in text_lower:
                return pattern.replace(" ", "_")
        
        return None
    
    def _extract_principle(self, text: str) -> Optional[str]:
        """Extract teaching principle from hint text."""
        # Look for sentences with principle-like keywords
        sentences = text.split(".")
        
        principle_keywords = ["always", "should", "remember", "principle", "rule", "before"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in principle_keywords):
                return sentence.strip() + "."
        
        return None
    
    def get_stats(self) -> dict:
        """Get generation statistics."""
        cache_stats = self.client.get_cache_stats()
        return {
            "generated_count": self.generated_count,
            "cache_stats": cache_stats,
        }


def generate_hints_batch(
    snapshots: list[dict],
    client,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """
    Generate hints for a batch of snapshots.
    
    Args:
        snapshots: List of snapshot dicts
        client: TeacherClient instance
        progress_callback: Progress callback
        
    Returns:
        List of snapshots with hints added
    """
    generator = HintGenerator(client)
    results = []
    total = len(snapshots)
    
    for i, snap in enumerate(snapshots):
        try:
            # Build hint request from snapshot
            request = HintRequest(
                task_description=snap.get("task", {}).get("description", ""),
                state_text=snap.get("state", ""),
                wrong_action=snap.get("policy_action", ""),
                correct_action=snap.get("expert_action", ""),
                available_actions=snap.get("available_actions", []),
                snapshot_id=snap.get("id", str(i)),
            )
            
            hint = generator.generate_hint(request)
            
            # Add hint to snapshot
            snap_with_hint = snap.copy()
            snap_with_hint["teacher_hint"] = hint.to_dict()
            results.append(snap_with_hint)
            
        except Exception as e:
            logger.error(f"Failed to generate hint for snapshot {i}: {e}")
            results.append(snap)  # Keep original without hint
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    logger.info(f"Generated {generator.generated_count} hints")
    logger.info(f"Cache stats: {generator.get_stats()['cache_stats']}")
    
    return results


def create_hint_request_from_snapshot(snapshot: dict) -> HintRequest:
    """
    Create HintRequest from a labeled snapshot.
    
    Args:
        snapshot: Labeled snapshot dict
        
    Returns:
        HintRequest ready for hint generation
    """
    return HintRequest(
        task_description=snapshot.get("task", {}).get("description", ""),
        state_text=snapshot.get("state", ""),
        wrong_action=snapshot.get("policy_action", ""),
        correct_action=snapshot.get("expert_action", ""),
        available_actions=snapshot.get("available_actions", []),
        snapshot_id=snapshot.get("id"),
    )


def run_hint_generation(
    input_path: str,
    output_path: str,
    client,
    filter_fn: Optional[Callable[[dict], bool]] = None,
) -> dict:
    """
    Run hint generation on a set of snapshots.
    
    Args:
        input_path: Path to input JSON with snapshots
        output_path: Path to save snapshots with hints
        client: TeacherClient instance
        filter_fn: Optional filter to select which snapshots get hints
        
    Returns:
        Summary statistics
    """
    with open(input_path) as f:
        data = json.load(f)
    
    snapshots = data.get("snapshots", data)
    
    if filter_fn:
        snapshots_to_process = [s for s in snapshots if filter_fn(s)]
        logger.info(f"Filtered to {len(snapshots_to_process)} snapshots")
    else:
        snapshots_to_process = snapshots
    
    results = generate_hints_batch(snapshots_to_process, client)
    
    # Save results
    output_data = {"snapshots": results}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Compute summary
    hints_generated = sum(1 for r in results if "teacher_hint" in r)
    
    return {
        "input_count": len(snapshots),
        "processed_count": len(snapshots_to_process),
        "hints_generated": hints_generated,
        "output_path": output_path,
    }
