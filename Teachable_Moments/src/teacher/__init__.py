"""
Teacher module for hint generation.

Uses GPT-4o to generate diagnostic teaching hints that explain
agent errors and suggest improvements.
"""

from .client import (
    TeacherConfig,
    TeacherCache,
    TeacherClient,
    MockTeacherClient,
    create_teacher_client,
)

from .hint_generator import (
    HintRequest,
    GeneratedHint,
    HintGenerator,
    generate_hints_batch,
    create_hint_request_from_snapshot,
    run_hint_generation,
    HINT_SYSTEM_PROMPT,
)

__all__ = [
    # Client
    "TeacherConfig",
    "TeacherCache",
    "TeacherClient",
    "MockTeacherClient",
    "create_teacher_client",
    # Hint generator
    "HintRequest",
    "GeneratedHint",
    "HintGenerator",
    "generate_hints_batch",
    "create_hint_request_from_snapshot",
    "run_hint_generation",
    "HINT_SYSTEM_PROMPT",
]
