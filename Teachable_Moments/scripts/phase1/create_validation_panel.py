#!/usr/bin/env python3
"""Create a fixed evaluation panel (stratified sample) from labeled snapshots.

The panel is used for:
  - retention evaluation (catastrophic forgetting)
  - drift evaluation
  - consistent qualitative error analysis

Input must be in current `LabeledSnapshot.to_dict()` schema.

The output JSON stores a list of references (snapshot_id/task_id/trajectory_id/step_idx) and, optionally, a light payload (observation/valid_actions/hint). This keeps panels reproducible without copying massive env_state blobs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import setup_logging, set_seed

from scripts.phase1._labeling_io import load_labeled_snapshots


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a stratified evaluation panel")
    p.add_argument("--input", type=Path, required=True, help="Labeled snapshots (.json/.jsonl)")
    p.add_argument("--output", type=Path, required=True)

    p.add_argument("--panel-size", type=int, default=200)
    p.add_argument("--per-quadrant", type=int, default=None)
    p.add_argument(
        "--stratify-by",
        type=str,
        default="quadrant",
        choices=["quadrant", "U", "L_local", "depth"],
        help="Variable used to stratify sampling",
    )
    p.add_argument("--include_payload", action="store_true", help="Include observation/actions/hint in panel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    items = load_labeled_snapshots(args.input)
    logger.info("Loaded %d labeled snapshots", len(items))

    if args.per_quadrant is not None:
        panel_size = int(args.per_quadrant) * 4
    else:
        panel_size = int(args.panel_size)

    # Determine strata key per item
    def stratum_key(ls):
        if args.stratify_by == "quadrant":
            return ls.quadrant
        if args.stratify_by == "U":
            return "highU" if ls.U >= float(np.median([x.U for x in items])) else "lowU"
        if args.stratify_by == "L_local":
            vals = [x.leverage.L_local for x in items if x.leverage is not None]
            thr = float(np.median(vals)) if vals else 0.0
            L = float(ls.leverage.L_local) if ls.leverage is not None else 0.0
            return "highL" if L >= thr else "lowL"
        if args.stratify_by == "depth":
            d = ls.depth.d_force if ls.depth is not None else None
            if d is None:
                return "unknown"
            if d <= 1:
                return "shallow"
            if d <= 4:
                return "medium"
            return "deep"
        return "unknown"

    strata: dict[str, list] = {}
    for ls in items:
        k = stratum_key(ls)
        strata.setdefault(k, []).append(ls)

    logger.info("Strata (%s): %s", args.stratify_by, {k: len(v) for k, v in strata.items()})

    # Decide how many per stratum
    if args.per_quadrant is not None and args.stratify_by == "quadrant":
        per = int(args.per_quadrant)
        per_stratum = {k: per for k in ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]}
    else:
        # uniform across existing strata
        keys = list(strata.keys())
        if not keys:
            raise SystemExit("No strata found")
        base = panel_size // len(keys)
        per_stratum = {k: base for k in keys}

    panel_refs: list[dict] = []
    for k, group in strata.items():
        n = min(per_stratum.get(k, 0), len(group))
        if n <= 0:
            continue
        sampled = random.sample(group, n)
        for ls in sampled:
            ref = {
                "snapshot_id": ls.snapshot.id,
                "task_id": ls.snapshot.task_id,
                "trajectory_id": ls.snapshot.trajectory_id,
                "step_idx": ls.snapshot.step_idx,
                "stratum": k,
            }
            if args.include_payload:
                ref["observation"] = ls.snapshot.observation
                ref["valid_actions"] = ls.snapshot.valid_actions
                ref["teacher_hint"] = ls.snapshot.teacher_hint.to_dict() if ls.snapshot.teacher_hint else None
            panel_refs.append(ref)

    random.shuffle(panel_refs)
    logger.info("Panel size: %d", len(panel_refs))

    out = {
        "version": "v8",
        "seed": args.seed,
        "input": str(args.input),
        "stratify_by": args.stratify_by,
        "panel_size": len(panel_refs),
        "tasks": panel_refs,
        "strata_counts": {k: len(v) for k, v in strata.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
