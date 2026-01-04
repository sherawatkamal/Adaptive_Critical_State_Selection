"""
DPO data preparation module.
Constructs preference pairs (chosen/rejected) from labeled snapshots.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random

@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str
    snapshot_id: str
    metadata: Dict[str, Any] = None


def build_dpo_dataset(
    snapshots: List[Dict[str, Any]],
    use_random_rejected: bool = False,
) -> List[DPOExample]:
    """
    Build DPO dataset from snapshots.
    
    Strategy:
    - Chosen: Teacher's suggested action
    - Rejected: Student's original action (which led to failure)
                OR random other valid action if use_random_rejected is True
    """
    examples = []
    
    for snap in snapshots:
        # Access nested snapshot if present (structure variance)
        data = snap.get("snapshot", snap)
        
        # 1. Get Prompt
        observation = data.get("observation", "")
        agent_prefix = data.get("agent_prefix", "")
        prompt = f"{observation}\n{agent_prefix}".strip()
        
        # 2. Get Chosen (Teacher)
        teacher_hint = snap.get("teacher_hint", {})
        if hasattr(teacher_hint, "to_dict"):
            teacher_hint = teacher_hint.to_dict()
            
        chosen_action = teacher_hint.get("suggested_action", "")
        if not chosen_action:
            continue
            
        # 3. Get Rejected (Student)
        if use_random_rejected:
            # Pick random valid action != chosen
            valid_actions = data.get("valid_actions", data.get("available_actions", []))
            candidates = [a for a in valid_actions if a != chosen_action]
            if not candidates:
                continue
            rejected_action = random.choice(candidates)
        else:
            rejected_action = snap.get("student_action") or data.get("policy_action", data.get("last_action", ""))
            
        if not rejected_action or rejected_action == chosen_action:
            continue
            
        examples.append(DPOExample(
            prompt=prompt,
            chosen=chosen_action,
            rejected=rejected_action,
            snapshot_id=data.get("id", ""),
            metadata={
                "quadrant": snap.get("quadrant", ""),
                "dataset_source": "failure_correction"
            }
        ))
        
    return examples
