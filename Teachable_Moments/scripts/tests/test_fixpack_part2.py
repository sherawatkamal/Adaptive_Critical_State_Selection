import unittest
import sys
import json
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.model_factory import ModelFactory, ModelConfig
from scripts.phase0.mine_snapshots import mine_snapshots

class TestFixpackPart2(unittest.TestCase):
    
    def test_model_factory_decode_action_signature(self):
        """Test that decode_action returns 3 values."""
        # Mock configs
        config = ModelConfig(model_path="test-model")
        factory = ModelFactory(config)
        
        # Mock load() to return mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        factory.load = MagicMock(return_value=(mock_model, mock_tokenizer))
        
        # Mock generate output
        mock_outputs = MagicMock()
        mock_outputs.sequences = [[1, 2, 3, 4, 5]]
        mock_model.generate.return_value = mock_outputs
        
        # Mock tokenizer methods
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.return_value["input_ids"].shape = [1, 2] # prompt len 2
        mock_tokenizer.decode.return_value = "search[query]"
        
        # Mock action scoring
        factory._compute_action_probabilities = MagicMock(return_value={"search[query]": 0.9, "click[x]": 0.1})
        
        # Test decode_action
        action, probs, raw_text = factory.decode_action(
            observation="test obs",
            valid_actions=["search[query]", "click[x]"],
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        self.assertEqual(action, "search[query]")
        self.assertEqual(raw_text, "search[query]")
        self.assertIsInstance(probs, dict)
        
    def test_mine_snapshots(self):
        """Test snapshot mining logic."""
        # Create a dummy trajectory file
        dummy_traj = [{
            "trajectory_id": "traj_1",
            "task_id": "task_1",
            "success": True,
            "expert_model": "test-expert",
            "steps": [
                {
                    "step_idx": 0,
                    "env_state_b64": base64.b64encode(b"state0").decode("ascii"),
                    "observation": "obs0",
                    "valid_actions": ["a1", "a2"],
                    "action_taken": "a1"
                },
                {
                    "step_idx": 1,
                    # No state, should be skipped
                    "observation": "obs1"
                }
            ]
        }]
        
        traj_file = "tmp_traj.json"
        out_file = "tmp_snapshots.json"
        
        with open(traj_file, "w") as f:
            json.dump(dummy_traj, f)
            
        try:
            # Run mining
            count = mine_snapshots(traj_file, out_file)
            
            # Verify
            self.assertEqual(count, 1)
            
            with open(out_file) as f:
                snapshots = json.load(f)
                
            self.assertEqual(len(snapshots), 1)
            snap = snapshots[0]
            self.assertEqual(snap["id"], "traj_1_step0")
            self.assertEqual(snap["env_state_b64"], base64.b64encode(b"state0").decode("ascii"))
            
        finally:
            # Cleanup
            if Path(traj_file).exists():
                Path(traj_file).unlink()
            if Path(out_file).exists():
                Path(out_file).unlink()

if __name__ == "__main__":
    unittest.main()
