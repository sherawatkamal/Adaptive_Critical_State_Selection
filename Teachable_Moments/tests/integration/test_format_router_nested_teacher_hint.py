import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.supervision.format_router import generate_supervision_single


def test_generate_supervision_reads_teacher_hint_from_nested_snapshot():
    labeled = {
        "snapshot_id": "s1",
        "quadrant": "Q1_highU_highL",
        "snapshot": {
            "observation": "Instruction: ...\nObservation: page ...",
            "valid_actions": ["click[a]", "search[b]"],
            "last_action": "search[b]",
            "teacher_hint": {
                "suggested_action": "click[a]",
                "rationale": "Because it leads to the product page.",
            },
        },
    }

    ex = generate_supervision_single(labeled, format="demo")
    assert "click[a]" in ex.output_text, "teacher action should be the target"

    ex2 = generate_supervision_single(labeled, format="contrast")
    assert "click[a]" in ex2.output_text
    assert "search[b]" in ex2.input_text, "contrast prompt should include last_action"
