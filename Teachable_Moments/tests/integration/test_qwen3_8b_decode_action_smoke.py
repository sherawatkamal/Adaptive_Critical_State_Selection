import os
from pathlib import Path

import pytest


def test_qwen3_8b_decode_action_smoke():
    if not os.environ.get("RUN_QWEN_SMOKE"):
        pytest.skip("Set RUN_QWEN_SMOKE=1 to enable local Qwen3-8B smoke test")

    try:
        import torch
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch not available: {e}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Qwen3-8B smoke test")

    total_mem = torch.cuda.get_device_properties(0).total_memory
    if total_mem < 22 * 1024**3:
        pytest.skip("Insufficient GPU memory for Qwen3-8B fp16 load (need ~22GB+)")

    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "models" / "Qwen3-8B"
    if not model_dir.exists():
        pytest.skip(f"Model directory not found: {model_dir}")

    from src.utils.model_factory import ModelConfig, ModelFactory

    cfg = ModelConfig(
        model_path=str(model_dir),
        device_map="auto",
        torch_dtype="float16",
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False,
    )
    factory = ModelFactory(cfg)

    observation = "You are on a page with items."
    valid_actions = ["search[shoes]", "click[item]", "back"]

    action, probs, raw = factory.decode_action(observation=observation, valid_actions=valid_actions)

    assert action in valid_actions
    assert isinstance(raw, str)
    assert set(probs.keys()) == set(valid_actions)
    assert abs(sum(probs.values()) - 1.0) < 1e-3
