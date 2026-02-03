#!/usr/bin/env python3
"""
Convert repo failure JSON/JSONL to IDT trajectory JSONL.
  python -m idt.scripts.convert_failures_to_trajectories --input path/to/failures.json --output path/to/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idt.types import trajectory_from_failure_dict, save_trajectories, load_trajectories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Failures JSON or JSONL")
    ap.add_argument("--output", "-o", required=True, help="Output trajectory JSONL")
    args = ap.parse_args()
    trajectories = load_trajectories(args.input)
    save_trajectories(args.output, trajectories)
    print(f"Converted {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
