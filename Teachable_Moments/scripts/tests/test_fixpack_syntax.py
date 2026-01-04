#!/usr/bin/env python3
"""Basic syntax/import sanity checks for the fixpack.

Run:
  python scripts/tests/test_fixpack_syntax.py
"""

from __future__ import annotations

import py_compile
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = [
        root / "src" / "simulation" / "student_rollout.py",
        root / "src" / "policies" / "model_factory_policy.py",
        root / "src" / "policies" / "random_policy.py",
        root / "src" / "eval" / "webshop_evaluator.py",
        root / "scripts" / "phase0" / "mine_failure_snapshots.py",
        root / "scripts" / "phase1" / "build_dataset.py",
        root / "scripts" / "phase1" / "select_stratified_panel.py",
    ]

    missing = [str(p) for p in targets if not p.exists()]
    if missing:
        print("Missing expected files:")
        for m in missing:
            print("  -", m)
        sys.exit(2)

    for p in targets:
        py_compile.compile(str(p), doraise=True)
        print("OK:", p.relative_to(root))

    print("\nAll fixpack syntax checks passed.")


if __name__ == "__main__":
    main()
