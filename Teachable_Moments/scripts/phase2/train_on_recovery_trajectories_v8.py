#!/usr/bin/env python3
"""Train SFT on exported recovery trajectories.

This consumes the output of:
  scripts/phase1/export_lupper_recovery_trajectories_v8.py

and turns each step (observation, valid_actions, expert_action) into a standard
"demo" supervision example.

This is useful as:
  - A fallback ablation if quadrants are not cleanly separable
  - A direct comparison against single-step teacher hints

The key property is that it uses *expert continuation* actions beyond the
original failure state.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from supervision.format_router import generate_supervision, SupervisionFormat
from training.sft_trainer import train_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train on recovery trajectories (v8)")
    ap.add_argument("--recovery-trajectories", type=str, required=True, help="JSONL from export_lupper_recovery_trajectories_v8.py")
    ap.add_argument("--base-model", type=str, required=True, help="Base model name/path for loading")
    ap.add_argument("--output-dir", type=str, default="results/phase2/recovery_sft_v8", help="Where to write the trained model")

    ap.add_argument("--max-examples", type=int, default=2000, help="Total step-examples to use (fixed budget)")
    ap.add_argument("--only-success", action="store_true", help="Only include trajectories that reached success")
    ap.add_argument("--seed", type=int, default=42)

    # Training hyperparams (keep modest defaults)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    traj_path = Path(args.recovery_trajectories)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = _load_jsonl(traj_path)
    if args.only_success:
        trajs = [t for t in trajs if t.get("success") is True]
    if not trajs:
        raise SystemExit("No trajectories to train on (check filters)")

    # Flatten steps
    step_snaps: List[Dict[str, Any]] = []
    for t in trajs:
        quadrant = t.get("quadrant", "")
        snap_id = t.get("snapshot_id", "")
        for i, st in enumerate(t.get("steps", []) or []):
            step_snaps.append(
                {
                    "id": f"{snap_id}:recov:{i}",
                    "quadrant": quadrant,
                    "observation": st.get("observation", ""),
                    "valid_actions": st.get("valid_actions", []) or [],
                    "last_action": st.get("prev_action", ""),
                    "teacher_hint": {
                        "suggested_action": st.get("action", ""),
                        "rationale": st.get("reason", ""),
                        "error_type": "recovery_demo",
                        "confidence": 1.0,
                    },
                }
            )

    if not step_snaps:
        raise SystemExit("No steps found in trajectories")

    random.shuffle(step_snaps)
    if args.max_examples and len(step_snaps) > args.max_examples:
        step_snaps = step_snaps[: args.max_examples]

    logger.info(f"Training on {len(step_snaps)} step examples from {len(trajs)} trajectories")

    examples = generate_supervision(step_snaps, SupervisionFormat.DEMO)
    training_data = [{"input": ex.input_text, "output": ex.output_text} for ex in examples]

    train_sft(
        base_model=args.base_model,
        training_data=training_data,
        output_dir=str(out_dir),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    logger.info(f"Saved recovery-SFT model to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
