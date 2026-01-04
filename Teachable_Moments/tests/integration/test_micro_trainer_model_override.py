import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.micro_trainer import run_policy_rollout_from_state


class DummyEnv:
    def __init__(self):
        self._step = 0

    def set_state(self, _bytes):
        self._step = 0
        return {"observation": "obs0", "valid_actions": ["a1"], "instruction_text": "inst"}

    def step(self, action):
        self._step += 1
        return {
            "observation": f"obs{self._step}",
            "valid_actions": ["a1"],
            "instruction_text": "inst",
        }, 1.0, True, {}

    def is_success(self, total_reward: float) -> bool:
        return total_reward >= 0.8


class DummyModelFactory:
    def __init__(self):
        self.last_model = None
        self.last_tokenizer = None

    def decode_action(self, observation, valid_actions, task_description="", model=None, tokenizer=None):
        self.last_model = model
        self.last_tokenizer = tokenizer
        return valid_actions[0], 0.0, {valid_actions[0]: 0.0}


def test_run_policy_rollout_passes_model_override():
    env = DummyEnv()
    mf = DummyModelFactory()

    sentinel_model = object()
    sentinel_tok = object()

    out = run_policy_rollout_from_state(
        env=env,
        model_factory=mf,
        model=sentinel_model,
        tokenizer=sentinel_tok,
        env_state_bytes=b"dummy",
        max_steps=3,
    )

    assert out["success"] is True
    assert mf.last_model is sentinel_model
    assert mf.last_tokenizer is sentinel_tok
