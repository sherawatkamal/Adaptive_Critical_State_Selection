#!/usr/bin/env python3
"""Smoke test: select_stratified_panel produces a panel of the requested size.

Run:
  python scripts/tests/test_select_stratified_panel.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "tmp" / "panel_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / "labeled.json"
    panel_path = out_dir / "panel.json"

    # Create tiny labeled file across 4 quadrants
    labeled = []
    quads = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_highL", "Q4_lowU_lowL"]
    for i in range(40):
        labeled.append({"id": f"snap_{i:03d}", "quadrant": quads[i % 4]})
    labeled_path.write_text(json.dumps(labeled, indent=2))

    script = root / "scripts" / "phase1" / "select_stratified_panel.py"
    subprocess.check_call([sys.executable, str(script), "--labeled", str(labeled_path), "--n", "20", "--seed", "7", "--output", str(panel_path)], cwd=str(root))

    panel = json.loads(panel_path.read_text())
    assert panel["n"] == 20, f"Expected 20, got {panel['n']}"
    assert len(panel["ids"]) == 20, "Panel ids length mismatch"

    print("PASS: select_stratified_panel")


if __name__ == "__main__":
    main()
