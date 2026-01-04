import unittest
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.teacher.structured_hint import generate_teacher_hint

class MockClientOnlyGenerate:
    """Mock client that ONLY has generate() and NOT generate_text()."""
    def generate(self, prompt, **kwargs):
        return '{"suggested_action": "search[test]", "rationale": "test", "error_type": "planning_error", "confidence": "high"}'

class TestTeacherClientCompat(unittest.TestCase):
    def test_generate_compatibility(self):
        """Test that generate_teacher_hint works with a client that only has generate()."""
        client = MockClientOnlyGenerate()
        hint = generate_teacher_hint(
            teacher_client=client,
            instruction_text="task",
            observation="obs",
            valid_actions=["search[test]"]
        )
        self.assertEqual(hint.suggested_action, "search[test]")

if __name__ == "__main__":
    unittest.main()
