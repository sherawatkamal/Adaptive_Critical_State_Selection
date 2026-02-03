"""
Create WebShop env and agent from baseline_models for IDT pipeline.
Uses baseline_models WebEnv + EEF Agent; requires running from repo root with baseline_models available.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from idt.agent_adapter import WebShopAgentAdapter
from idt.env_adapter import WebShopWebEnvAdapter


def create_webshop_env_and_agent(
    model_path: str = "baseline_models/ckpts/web_click/epoch_9/model.pth",
    split: str = "test",
) -> Tuple[WebShopWebEnvAdapter, WebShopAgentAdapter]:
    """
    Create WebShop env and agent from baseline_models.
    Temporarily chdir to baseline_models and clear argv so setup_environment/setup_model work.
    model_path can be relative to repo root (e.g. baseline_models/ckpts/.../model.pth) or
    relative to baseline_models (e.g. ckpts/web_click/epoch_9/model.pth).
    Uses BM25 search fallback (no Java/Lucene) so the pipeline runs without JVM.
    Returns (env_adapter, agent_adapter).
    """
    # Use BM25 search instead of Lucene so we avoid Java/jdk.incubator.vector errors
    os.environ['IDT_USE_BM25'] = '1'
    # Limit products for faster env init (avoid loading ~1.2M products)
    os.environ['IDT_NUM_PRODUCTS'] = os.environ.get('IDT_NUM_PRODUCTS', '1000')
    # idt may live at repo_root/idt or baseline_models/idt
    repo_root = Path(__file__).resolve().parents[1]
    baseline_dir = repo_root / "baseline_models" if (repo_root / "baseline_models").is_dir() else repo_root
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"baseline_models not found at {baseline_dir}")

    # When we chdir to baseline_models, model_path must be relative to baseline_models
    path_obj = Path(model_path)
    if not path_obj.is_absolute():
        if "baseline_models" in path_obj.parts:
            # strip baseline_models/ prefix
            parts = list(path_obj.parts)
            if parts[0] == "baseline_models":
                model_path_inside = str(Path(*parts[1:]))
            else:
                model_path_inside = model_path
        else:
            model_path_inside = model_path
    else:
        model_path_inside = model_path

    old_cwd = os.getcwd()
    old_argv = list(sys.argv)
    try:
        sys.path.insert(0, str(baseline_dir))
        os.chdir(baseline_dir)
        sys.argv = [sys.argv[0]]

        from eef_detailed_with_diagnosis_parallel import (
            setup_environment,
            setup_model,
            Agent as EEFAgent,
        )

        env = setup_environment(split=split)
        models = setup_model(model_path_inside)
        agent = EEFAgent(models)
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        if sys.path and sys.path[0] == str(baseline_dir):
            sys.path.pop(0)

    env_adapter = WebShopWebEnvAdapter(env)
    agent_adapter = WebShopAgentAdapter(agent)
    return env_adapter, agent_adapter
