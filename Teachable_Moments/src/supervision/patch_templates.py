"""
Patch templates for CPT and supervision generation.
"""

# Demo format: Shows correct action with rationale
DEMO_FORMAT = """{observation}

[Example from a similar situation]
The correct action is: {teacher_action}
Reason: {rationale}
[End of example]

Your action:"""

# Contrast format: Explicitly contrasts wrong vs correct
CONTRAST_FORMAT = """{observation}

[Feedback on action choice]
- Avoid: {bad_action} - This fails because: {why_bad}
- Instead: {teacher_action} - This works because: {rationale}
[End of feedback]

Your action:"""

# Hint format: Provides diagnostic insight
HINT_FORMAT = """{observation}

[Observation from a similar situation]
A key insight: {diagnosis}
[End of observation]

Your action:"""


def format_demo(
    observation: str,
    teacher_action: str,
    rationale: str,
) -> str:
    """Format demo supervision."""
    return DEMO_FORMAT.format(
        observation=observation,
        teacher_action=teacher_action,
        rationale=rationale,
    )


def format_contrast(
    observation: str,
    bad_action: str,
    why_bad: str,
    teacher_action: str,
    rationale: str,
) -> str:
    """Format contrast supervision."""
    return CONTRAST_FORMAT.format(
        observation=observation,
        bad_action=bad_action,
        why_bad=why_bad,
        teacher_action=teacher_action,
        rationale=rationale,
    )


def format_hint(
    observation: str,
    diagnosis: str,
) -> str:
    """Format hint supervision."""
    return HINT_FORMAT.format(
        observation=observation,
        diagnosis=diagnosis,
    )
