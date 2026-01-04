"""Tests for supervision modules."""

import pytest

from src.supervision.format_router import SupervisionFormat, generate_supervision_single
from src.supervision.patch_templates import format_contrast, format_demo, format_hint
from src.supervision.sft_data import create_sft_dataset, save_sft_dataset


class TestGenerateSupervisionSingle:
    """Tests for supervision generation."""

    def test_generate_demo(self):
        snapshot = {
            "id": "snap0",
            "observation": "You are on a product page.",
            "agent_prefix": "",
            "last_action": "click[Back]",
            "teacher_hint": {
                "suggested_action": "click[Add to Cart]",
                "rationale": "Add the item before checkout",
                "error_type": "planning_error",
            },
            "quadrant": "Q1_highU_highL",
        }
        ex = generate_supervision_single(snapshot, SupervisionFormat.DEMO)
        assert ex.output_text == "click[Add to Cart]"
        assert "product page" in ex.input_text.lower()

    def test_generate_contrast(self):
        snapshot = {
            "id": "snap0",
            "observation": "You are on a product page.",
            "agent_prefix": "",
            "last_action": "click[Back]",
            "teacher_hint": {
                "suggested_action": "click[Add to Cart]",
                "rationale": "Add the item before checkout",
                "error_type": "planning_error",
            },
            "quadrant": "Q1_highU_highL",
        }
        ex = generate_supervision_single(snapshot, SupervisionFormat.CONTRAST)
        assert ex.output_text == "click[Add to Cart]"
        assert "suboptimal" in ex.input_text.lower()

    def test_generate_hint(self):
        snapshot = {
            "id": "snap0",
            "observation": "You are on a product page.",
            "agent_prefix": "",
            "last_action": "click[Back]",
            "teacher_hint": {
                "suggested_action": "click[Add to Cart]",
                "rationale": "Add the item before checkout",
                "error_type": "planning_error",
            },
            "quadrant": "Q1_highU_highL",
        }
        ex = generate_supervision_single(snapshot, SupervisionFormat.HINT)
        assert ex.output_text == "click[Add to Cart]"
        assert "product page" in ex.input_text.lower()
        assert "add the item" in ex.input_text.lower() or "consider" in ex.input_text.lower()


class TestPatchTemplates:
    """Tests for patch templates."""

    def test_demo_template_format(self):
        """Test demo template includes action and rationale."""
        patch = format_demo(
            observation="viewing product page",
            teacher_action="click[add to cart]",
            rationale="Adding item before checkout",
        )
        
        assert "add to cart" in patch.lower()
        assert "action" in patch.lower() or "click" in patch.lower()

    def test_hint_template_format(self):
        """Test hint template includes guidance without answer."""
        patch = format_hint(
            observation="viewing search results",
            diagnosis="Consider which product matches the requirements",
        )
        
        # Hint should not contain explicit action
        assert "click[" not in patch.lower()
        assert "consider" in patch.lower() or "hint" in patch.lower()

    def test_contrast_template_format(self):
        """Test contrast template shows good vs bad actions."""
        patch = format_contrast(
            observation="at checkout",
            bad_action="click[back to shopping]",
            why_bad="This abandons the cart",
            teacher_action="click[place order]",
            rationale="Complete purchase rather than abandoning cart",
        )
        
        assert "place order" in patch.lower() or "good" in patch.lower()
        assert "back" in patch.lower() or "bad" in patch.lower()


class TestSFTDataset:
    """Tests for SFT dataset creation."""

    def test_create_sft_dataset(self):
        snapshots = [
            {
                "id": "snap0",
                "observation": "You are on a product page.",
                "agent_prefix": "",
                "last_action": "click[Back]",
                "teacher_hint": {
                    "suggested_action": "click[Add to Cart]",
                    "rationale": "Add the item before checkout",
                    "error_type": "planning_error",
                },
                "quadrant": "Q1_highU_highL",
            }
        ]
        ds = create_sft_dataset(snapshots, format=SupervisionFormat.DEMO, name="test")
        assert len(ds) == 1
        assert ds.examples[0].output_text == "click[Add to Cart]"

    def test_save_sft_dataset(self, tmp_path):
        snapshots = [
            {
                "id": "snap0",
                "observation": "You are on a product page.",
                "agent_prefix": "",
                "last_action": "click[Back]",
                "teacher_hint": {
                    "suggested_action": "click[Add to Cart]",
                    "rationale": "Add the item before checkout",
                    "error_type": "planning_error",
                },
                "quadrant": "Q1_highU_highL",
            }
        ]
        ds = create_sft_dataset(snapshots, format=SupervisionFormat.DEMO, name="test")
        paths = save_sft_dataset(ds, tmp_path, splits=False)
        assert list(paths.values())[0].exists()
