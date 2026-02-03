#!/usr/bin/env python3
"""
CLI wrapper for training teachability predictor.
  python -m idt.scripts.train_teachability_cli --dataset_path ... --model_out ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idt.train_teachability import load_records_with_features, train_baseline, save_model

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--model_out", type=str, default="teachability_model.joblib")
    ap.add_argument("--model_type", type=str, default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = load_records_with_features(args.dataset_path)
    if X.size == 0:
        raise SystemExit("No records or empty dataset")
    model, scaler = train_baseline(
        X, y, model_type=args.model_type, test_size=args.test_size, random_state=args.seed
    )
    save_model(model, scaler, args.model_out)
    print("Saved model to", args.model_out)


if __name__ == "__main__":
    main()
