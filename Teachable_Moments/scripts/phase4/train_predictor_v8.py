"""Phase 4 (v8): Train teachability predictor (wrapper).

The repo already contains a v8-capable predictor trainer at
`scripts/phase4/train_predictor.py`, but several orchestration scripts (pilots,
execution plan) expect a stable CLI with `--labeled-snapshots`.

This wrapper:
- Loads labeled snapshots from `.jsonl` or `.json`
- Adds Tier-1 structural features (cheap, deterministic)
- Writes an augmented JSON list to <output-dir>/predictor_input.json
- Invokes `scripts/phase4/train_predictor.py` on that input

You can still run the underlying trainer directly if you prefer.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def maybe_add_structural_features(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add structural_features if missing (Tier1)."""
    # Import lazily to keep wrapper light.
    sys.path.insert(0, str(REPO_ROOT))
    from src.features.tier1_structural import extract_structural_features

    augmented: List[Dict[str, Any]] = []
    for r in records:
        if r.get("structural_features") is not None:
            augmented.append(r)
            continue
        snap = r.get("snapshot")
        if not isinstance(snap, dict):
            augmented.append(r)
            continue
        flat = dict(snap)
        # make sure extractor can see teachability fields
        flat["quadrant"] = r.get("quadrant")
        flat["U"] = r.get("U")
        flat["uncertainty_features"] = r.get("uncertainty_features")
        flat["leverage"] = r.get("leverage")
        try:
            sf = extract_structural_features(flat)
            r = dict(r)
            r["structural_features"] = sf.to_dict()
        except Exception:
            pass
        augmented.append(r)
    return augmented


def main() -> None:
    ap = argparse.ArgumentParser(description="Train teachability predictor (v8 wrapper)")
    ap.add_argument("--labeled-snapshots", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/phase4/v8"))

    # Pass-through options to the underlying trainer
    ap.add_argument("--embedder", type=str, default="hashing", choices=["hashing", "sentence_transformer", "mock"])
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--use-only-tier1", action="store_true")
    ap.add_argument("--mock-embeddings", action="store_true")

    # Training knobs
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    ap.add_argument("--dropout", type=float, default=0.2)

    args = ap.parse_args()
    setup_logging()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.labeled_snapshots)
    logger.info(f"Loaded {len(records)} labeled records")

    records = maybe_add_structural_features(records)

    input_path = args.output_dir / "predictor_input.json"
    input_path.write_text(json.dumps(records))

    # Build subprocess call
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "phase4" / "train_predictor.py"),
        "--input",
        str(input_path),
        "--output-dir",
        str(args.output_dir),
        "--embedder",
        args.embedder if not args.mock_embeddings else "mock",
        "--embedding-dim",
        str(args.embedding_dim),
        "--embedding-model",
        args.embedding_model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--dropout",
        str(args.dropout),
    ]

    cmd.extend(["--hidden-dims"] + [str(d) for d in args.hidden_dims])
    if args.use_only_tier1:
        cmd.append("--use-only-tier1")

    logger.info("Running underlying trainer: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))

    logger.info(f"Predictor artifacts saved under {args.output_dir}")


if __name__ == "__main__":
    main()
