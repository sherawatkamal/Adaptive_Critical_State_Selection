#!/usr/bin/env python3
"""
CLI entrypoint for training teachability predictor.
  python -m idt.scripts.train_teachability --dataset_path ... --model_out ...
"""
from idt.scripts.train_teachability_cli import main

if __name__ == "__main__":
    main()
