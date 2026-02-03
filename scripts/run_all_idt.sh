#!/bin/bash
# Run IDT pipeline: tests, build dataset (toy N=20), train predictor, eval plots.
# Usage: from repo root, run: bash scripts/run_all_idt.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "=== 1) Unit tests ==="
python -m pytest tests/test_toy_env.py tests/test_patches.py tests/test_replay.py tests/test_verifier.py tests/test_search_min_patch.py -v

echo "=== 2) Build IDT dataset (toy N=20) ==="
python -m idt.scripts.build_idt_dataset --toy --num_trajectories 20 --out_path idt_dataset.jsonl --K 5 --threshold 0.6

echo "=== 3) Train teachability predictor ==="
python -m idt.scripts.train_teachability --dataset_path idt_dataset.jsonl --model_out teachability_model.joblib

echo "=== 4) Evaluation plots ==="
python -m idt.scripts.eval_landscape --dataset_path idt_dataset.jsonl --out_dir idt_eval_out

echo "Done. Check idt_eval_out/ for metrics and plots."
