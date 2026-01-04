"""
Structured teacher hint generation for CPT patches.

Generates TeacherHint objects with JSON schema that matches
the expected format for CPT patch generation.
"""

import json
import logging
import re
from typing import Optional, Sequence, Any

from ..data.snapshot import TeacherHint, ErrorType

logger = logging.getLogger(__name__)


# JSON schema for teacher hint responses
HINT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "suggested_action": {"type": "string", "description": "The correct action to take"},
        "rationale": {"type": "string", "description": "Why this action is correct"},
        "error_type": {
            "type": "string",
            "enum": ["affordance_miss", "attribute_confusion", "planning_error", "exploration_failure"],
            "description": "Type of error the student made"
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence in this recommendation"
        }
    },
    "required": ["suggested_action", "rationale", "error_type", "confidence"]
}


def build_teacher_hint_prompt(
    instruction_text: str,
    observation: str,
    valid_actions: Sequence[str],
    student_action: Optional[str] = None,
) -> str:
    """Build a prompt that asks the teacher model to return JSON only.
    
    Args:
        instruction_text: The task instruction/goal
        observation: Current observation text
        valid_actions: List of valid action strings
        student_action: Optional action the student took (if analyzing an error)
        
    Returns:
        Prompt string that requests JSON-formatted hint
    """
    actions_str = "\n".join(f"  - {a}" for a in valid_actions[:15])  # Limit for context
    
    student_context = ""
    if student_action:
        student_context = f"""
The student chose: {student_action}
Analyze whether this was correct and provide the recommended action.
"""
    
    prompt = f"""You are an expert teacher analyzing a web shopping task.

TASK INSTRUCTION: {instruction_text}

CURRENT OBSERVATION:
{observation}

VALID ACTIONS:
{actions_str}
{student_context}
Provide a teaching hint as JSON with exactly these keys:
- suggested_action: The correct action to take (must be one of the valid actions, or a search query in format search[query])
- rationale: A brief explanation of why this action is correct
- error_type: One of "affordance_miss", "attribute_confusion", "planning_error", "exploration_failure"
- confidence: One of "low", "medium", "high"

Respond with ONLY valid JSON, no other text:"""
    
    return prompt


def _extract_json_obj(raw_text: str) -> dict:
    """Extract JSON object from raw text response.
    
    Handles cases where the model returns extra text around the JSON.
    
    Args:
        raw_text: Raw text from model
        
    Returns:
        Parsed JSON dict
        
    Raises:
        ValueError: If no valid JSON found
    """
    # Try direct parsing first
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON with nested objects (more permissive)
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not extract JSON from text: {raw_text[:200]}...")


def _parse_error_type(error_str: str) -> ErrorType:
    """Parse error type string to ErrorType enum.
    
    Args:
        error_str: String representation of error type
        
    Returns:
        ErrorType enum value
    """
    error_map = {
        "affordance_miss": ErrorType.AFFORDANCE_MISS,
        "attribute_confusion": ErrorType.ATTRIBUTE_CONFUSION,
        "planning_error": ErrorType.PLANNING_ERROR,
        "exploration_failure": ErrorType.EXPLORATION_FAILURE,
    }
    
    error_lower = error_str.lower().strip()
    
    # Direct match
    if error_lower in error_map:
        return error_map[error_lower]
    
    # Fuzzy match
    for key, value in error_map.items():
        if key in error_lower or error_lower in key:
            return value
    
    # Default
    logger.warning(f"Unknown error type: {error_str}, defaulting to PLANNING_ERROR")
    return ErrorType.PLANNING_ERROR


def generate_teacher_hint(
    teacher_client: Any,
    instruction_text: str,
    observation: str,
    valid_actions: Sequence[str],
    student_action: Optional[str] = None,
    max_retries: int = 2,
) -> TeacherHint:
    """Generate a structured TeacherHint using an LLM client.
    
    Args:
        teacher_client: LLM client with generate_text() method
        instruction_text: Task instruction/goal
        observation: Current observation
        valid_actions: List of valid actions
        student_action: Optional student's action to analyze
        max_retries: Number of retries on parse failure
        
    Returns:
        TeacherHint with structured fields
        
    Raises:
        ValueError: If hint generation fails after retries
    """
    prompt = build_teacher_hint_prompt(
        instruction_text=instruction_text,
        observation=observation,
        valid_actions=valid_actions,
        student_action=student_action,
    )
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if hasattr(teacher_client, "generate_text"):
                raw = teacher_client.generate_text(prompt)
            elif hasattr(teacher_client, "generate"):
                raw = teacher_client.generate(prompt)
            else:
                raise AttributeError(f"Teacher client {type(teacher_client)} has no generate or generate_text method")
            obj = _extract_json_obj(raw)
            
            # Extract and validate fields
            suggested_action = obj.get("suggested_action", "")
            if not suggested_action:
                raise ValueError("Missing suggested_action")
            
            rationale = obj.get("rationale", "")
            if not rationale:
                rationale = "Action advances the task."
            
            error_type_str = obj.get("error_type", "planning_error")
            error_type = _parse_error_type(error_type_str)
            
            confidence = obj.get("confidence", "medium")
            if confidence not in ("low", "medium", "high"):
                confidence = "medium"
            
            return TeacherHint(
                suggested_action=suggested_action,
                rationale=rationale,
                error_type=error_type,
                confidence=confidence,
            )
            
        except Exception as e:
            last_error = e
            logger.warning(f"Hint generation attempt {attempt + 1} failed: {e}")
            continue
    
    raise ValueError(f"Failed to generate teacher hint after {max_retries + 1} attempts: {last_error}")


def hint_to_patch_text(hint: TeacherHint) -> str:
    """Convert a TeacherHint into a patch string to inject for CPT.
    
    Args:
        hint: TeacherHint object
        
    Returns:
        Formatted patch string for CPT injection
    """
    parts = [
        "TEACHER TIP:",
        f"Correct next action: {hint.suggested_action}",
        f"Why: {hint.rationale}",
    ]
    return "\n".join([p for p in parts if p])


def hint_to_contrast_patch(
    hint: TeacherHint,
    student_action: str,
    observation_summary: str = "",
) -> str:
    """Convert hint to contrastive patch format.
    
    Args:
        hint: TeacherHint object
        student_action: The action the student took
        observation_summary: Brief summary of observation
        
    Returns:
        Contrastive patch string
    """
    error_explanations = {
        ErrorType.AFFORDANCE_MISS: "The action doesn't match available options.",
        ErrorType.ATTRIBUTE_CONFUSION: "The action targets the wrong attribute.",
        ErrorType.PLANNING_ERROR: "The action is out of sequence for the goal.",
        ErrorType.EXPLORATION_FAILURE: "More exploration was needed first.",
    }
    
    why_bad = error_explanations.get(hint.error_type, "This action doesn't advance the task.")
    
    parts = [
        "[Feedback on action choice]",
    ]
    if observation_summary:
        parts.append(f'In a situation like this, where: "{observation_summary}"')
    
    parts.extend([
        f"- Avoid: {student_action} — This fails because: {why_bad}",
        f"- Instead: {hint.suggested_action} — This works because: {hint.rationale}",
        "[End of feedback]",
        "",
        "Now continue with your task:",
    ])
    
    return "\n".join(parts)


def batch_generate_hints(
    teacher_client: Any,
    snapshots: Sequence[dict],
    n_parallel: int = 4,
    progress_callback: Optional[Any] = None,
) -> dict[str, TeacherHint]:
    """Generate hints for a batch of snapshots.
    
    Args:
        teacher_client: LLM client with generate_text() method
        snapshots: Sequence of snapshot dicts with id, instruction_text, observation, valid_actions
        n_parallel: Number of parallel workers (currently sequential due to API limits)
        progress_callback: Optional callback(completed_count)
        
    Returns:
        Dict mapping snapshot ID to TeacherHint
    """
    results = {}
    
    for i, snap in enumerate(snapshots):
        snap_id = snap.get("id", f"snap_{i}")
        
        try:
            hint = generate_teacher_hint(
                teacher_client=teacher_client,
                instruction_text=snap.get("instruction_text", ""),
                observation=snap.get("observation", ""),
                valid_actions=snap.get("valid_actions", []),
                student_action=snap.get("last_action"),
            )
            results[snap_id] = hint
            
        except Exception as e:
            logger.error(f"Failed to generate hint for {snap_id}: {e}")
            # Create fallback hint
            results[snap_id] = TeacherHint(
                suggested_action=snap.get("valid_actions", ["none"])[0],
                rationale="Fallback hint due to generation failure.",
                error_type=ErrorType.PLANNING_ERROR,
                confidence="low",
            )
        
        if progress_callback:
            progress_callback(i + 1)
    
    return results
