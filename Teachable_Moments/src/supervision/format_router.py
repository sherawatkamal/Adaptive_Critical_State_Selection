"""
Generate supervision data in route-matched format.

Supports three supervision formats:
- Demo: Shows correct action with rationale
- Contrast: Explicitly contrasts wrong vs correct action
- Hint: Provides diagnostic insight without direct action
"""

from dataclasses import dataclass
from typing import Literal, Optional
from enum import Enum
import json


class SupervisionFormat(str, Enum):
    """Supervision format types."""
    DEMO = "demo"
    CONTRAST = "contrast"
    HINT = "hint"
    DEMO_RATIONALE = "demo_rationale"


@dataclass
class SupervisionExample:
    """A single SFT training example."""
    
    input_text: str
    output_text: str
    snapshot_id: str
    format: SupervisionFormat
    quadrant: str = ""
    metadata: dict = None
    
    def to_dict(self) -> dict:
        return {
            "input": self.input_text,
            "output": self.output_text,
            "snapshot_id": self.snapshot_id,
            "format": self.format.value,
            "quadrant": self.quadrant,
            "metadata": self.metadata or {},
        }


def generate_diagnosis_text(teacher_hint: dict) -> str:
    """
    Generate diagnostic text for hint format.
    
    Args:
        teacher_hint: Teacher hint dict with error_type and rationale
        
    Returns:
        Diagnostic insight string
    """
    error_type = teacher_hint.get("error_type", "unknown")
    rationale = teacher_hint.get("rationale", "")
    
    # Map error types to diagnostic insights
    error_diagnostics = {
        "affordance_miss": "Look carefully at all available options on the page.",
        "attribute_confusion": "Double-check that the item matches all required attributes.",
        "planning_error": "Consider the sequence of steps needed to complete the goal.",
        "exploration_failure": "More search or navigation may be needed to find the right item.",
    }
    
    base_diagnostic = error_diagnostics.get(error_type, "Analyze the current situation carefully.")
    
    # Combine with rationale if available
    if rationale:
        return f"{base_diagnostic} {rationale}"
    return base_diagnostic


from typing import Literal, Optional, Union

def generate_supervision_single(
    snapshot: dict,
    format: Union[SupervisionFormat, str],
) -> SupervisionExample:
    """
    Generate SFT training example with specified format.
    
    Args:
        snapshot: Snapshot dict with observation, agent_prefix, last_action, teacher_hint
        format: Supervision format (demo, contrast, or hint)
        
    Returns:
        SupervisionExample with input and output text
    """
    if isinstance(format, str):
        format = SupervisionFormat(format)
    
    observation = snapshot.get("observation", snapshot.get("snapshot", {}).get("observation", ""))

    # Include task/instruction context if available (important for WebShop)
    instruction_text = snapshot.get("instruction_text", "") or snapshot.get("task_description", "")
    if not instruction_text:
        # Some schemas nest instruction under snapshot dict; try there too
        instruction_text = snapshot.get("snapshot", {}).get("instruction_text", "") or snapshot.get("snapshot", {}).get("task_description", "")
    if instruction_text and not (observation or "").lstrip().lower().startswith("task:"):
        observation = f"Task: {instruction_text}\n\n{observation}".strip()
    agent_prefix = snapshot.get("agent_prefix", snapshot.get("snapshot", {}).get("agent_prefix", ""))
    last_action = snapshot.get("last_action", snapshot.get("snapshot", {}).get("last_action", ""))
    
    # Teacher hints live either at the top-level (Snapshot.to_dict) or nested under
    # `snapshot` (LabeledSnapshot.to_dict). Support both.
    teacher_hint = snapshot.get("teacher_hint") or snapshot.get("snapshot", {}).get("teacher_hint") or {}
    if isinstance(teacher_hint, dict) is False and hasattr(teacher_hint, "to_dict"):
        teacher_hint = teacher_hint.to_dict()
    
    suggested_action = teacher_hint.get("suggested_action", "")
    rationale = teacher_hint.get("rationale", "")
    
    snapshot_id = snapshot.get("id", snapshot.get("snapshot", {}).get("id", ""))
    quadrant = snapshot.get("quadrant", "")
    
    if format == SupervisionFormat.DEMO:
        input_text = f"{observation}\n{agent_prefix}".strip()
        output_text = suggested_action
        
    elif format == SupervisionFormat.CONTRAST:
        contrast_prefix = (
            f"The action '{last_action}' was suboptimal. "
            f"The better choice is '{suggested_action}' because: {rationale}"
        )
        input_text = f"{contrast_prefix}\n\n{observation}\n{agent_prefix}".strip()
        output_text = suggested_action
        
    elif format == SupervisionFormat.HINT:
        diagnosis = generate_diagnosis_text(teacher_hint)
        input_text = f"{diagnosis}\n\n{observation}\n{agent_prefix}".strip()
        output_text = suggested_action
        
    elif format == SupervisionFormat.DEMO_RATIONALE:
        input_text = f"{observation}\n{agent_prefix}".strip()
        output_text = json.dumps({
            "action": suggested_action,
            "rationale": rationale,
        })
        
    else:
        raise ValueError(f"Unknown supervision format: {format}")
    
    return SupervisionExample(
        input_text=input_text,
        output_text=output_text,
        snapshot_id=snapshot_id,
        format=format,
        quadrant=quadrant,
        metadata={
            "teacher_action": suggested_action,
            "teacher_rationale": rationale,
            "last_action": last_action,
        },
    )


def generate_supervision(
    snapshots: list[dict],
    format: Union[SupervisionFormat, str],
) -> list[SupervisionExample]:
    """
    Generate supervision examples for all snapshots in specified format.
    
    Args:
        snapshots: List of snapshot dicts
        format: Supervision format
        
    Returns:
        List of SupervisionExample objects
    """
    return [generate_supervision_single(snap, format) for snap in snapshots]


def generate_mixed_supervision(
    snapshots_by_quadrant: dict[str, list[dict]],
    format_by_quadrant: dict[str, Union[SupervisionFormat, str]],
) -> list[SupervisionExample]:
    """
    Generate supervision with different formats for different quadrants.
    
    Args:
        snapshots_by_quadrant: Dict mapping quadrant labels to snapshot lists
        format_by_quadrant: Dict mapping quadrant labels to supervision formats
        
    Returns:
        Combined list of SupervisionExample objects
    """
    all_examples = []
    
    for quadrant, snapshots in snapshots_by_quadrant.items():
        format_str = format_by_quadrant.get(quadrant, SupervisionFormat.DEMO)
        examples = generate_supervision(snapshots, format_str)
        all_examples.extend(examples)
    
    return all_examples


def create_training_prompt(
    observation: str,
    agent_prefix: str = "",
    system_prompt: str = "",
) -> str:
    """
    Create a training prompt in chat format.
    
    Args:
        observation: Current observation
        agent_prefix: Agent context/history
        system_prompt: Optional system prompt
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    if system_prompt:
        parts.append(f"<system>\n{system_prompt}\n</system>")
    
    if agent_prefix:
        parts.append(f"{agent_prefix}")
    
    parts.append(f"<observation>\n{observation}\n</observation>")
    parts.append("<action>")
    
    return "\n".join(parts)
