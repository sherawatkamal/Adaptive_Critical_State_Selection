#!/usr/bin/env python3
"""Phase 5 (v8): Selection Experiment (H7).

This script runs the selection-method comparison experiment to validate
that predictor-selected training data yields better reliability gains than
random or entropy-only selection at fixed training budget K.

Selection methods compared:
1) RANDOM: Uniform sampling from labeled pool
2) ENTROPY: Top-K by uncertainty U(s)
3) PREDICTOR: Top-K by predicted ELP from trained predictor
4) ORACLE: Top-K by true ELP_net (upper bound)

For each method, we:
1. Select K snapshots
2. Train a LoRA SFT model (same hyperparams for all)
3. Evaluate with run_eval_suite_v8.py-compatible outputs

Outputs (in --output-dir):
- selection_config.json
- per-method subdirs: {random,entropy,predictor,oracle}/
  - selected_snapshots.json
  - training_summary.json
  - training_result.json
- selection_comparison.csv (aggregated)
- selection_summary.json

Usage:
    python scripts/phase5/run_selection_experiment_v8.py \
        --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
        --predictor-path results/phase4/predictor/model.pt \
        --base-model <BASE_MODEL> \
        --budget 500 \
        --output-dir results/phase5/selection
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random as py_random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.snapshot import LabeledSnapshot, Snapshot
from src.supervision.format_router import FormatRouter
from src.training.sft_trainer import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class SelectionExperimentConfig:
    """Configuration for selection experiment."""
    
    labeled_snapshots_path: str
    predictor_path: Optional[str]
    base_model: str
    output_dir: str
    
    # Selection budget
    budget_k: int = 500
    
    # Which methods to run
    methods: List[str] = None  # default: ["random", "entropy", "predictor", "oracle"]
    
    # Training config (same for all methods)
    supervision_format: str = "demo"  # Use same format across methods
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    
    seed: int = 42
    
    # Mock mode (for testing)
    mock_training: bool = False
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["random", "entropy", "predictor", "oracle"]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_labeled_records(path: Path) -> List[Dict[str, Any]]:
    """Load labeled snapshots from JSONL or JSON."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    else:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected list in {path}")


def get_field(record: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get a field from record, trying multiple key names."""
    for key in keys:
        if key in record:
            return record[key]
        # Try nested under "snapshot"
        snap = record.get("snapshot", {})
        if isinstance(snap, dict) and key in snap:
            return snap[key]
    return default


# -----------------------------------------------------------------------------
# Selection methods
# -----------------------------------------------------------------------------

def select_random(
    records: List[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Random uniform selection."""
    rng = py_random.Random(seed)
    if len(records) <= k:
        return records
    return rng.sample(records, k)


def select_entropy(
    records: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Select top-K by uncertainty (entropy)."""
    scored = []
    for r in records:
        u = get_field(r, "uncertainty", "entropy", "U", default=None)
        if u is not None:
            scored.append((float(u), r))
    
    if not scored:
        logger.warning("No uncertainty values found; falling back to random")
        return select_random(records, k, seed=42)
    
    # Sort descending by uncertainty
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]


def load_predictor(path: Path) -> Any:
    """Load a trained predictor model."""
    try:
        import torch
        from src.predictor.model import TeachabilityPredictor
        
        # Try to load model
        if path.is_dir():
            model_path = path / "model.pt"
        else:
            model_path = path
        
        if not model_path.exists():
            logger.warning(f"Predictor not found at {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Infer model config from checkpoint
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        
        # Simple heuristic to infer input dim
        for key in state:
            if "input" in key and "weight" in key:
                input_dim = state[key].shape[1]
                break
        else:
            input_dim = 128  # default
        
        model = TeachabilityPredictor(input_dim=input_dim)
        model.load_state_dict(state)
        model.eval()
        return model
        
    except Exception as e:
        logger.warning(f"Failed to load predictor: {e}")
        return None


def select_predictor(
    records: List[Dict[str, Any]],
    k: int,
    predictor: Any,
) -> List[Dict[str, Any]]:
    """Select top-K by predicted ELP from predictor."""
    if predictor is None:
        logger.warning("No predictor available; falling back to entropy selection")
        return select_entropy(records, k)
    
    try:
        import torch
        from src.features.extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        scored = []
        
        for r in records:
            try:
                snap_dict = r.get("snapshot", r)
                features = extractor.extract(snap_dict)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    pred = predictor(features_tensor)
                    if isinstance(pred, dict):
                        elp_pred = pred.get("elp", pred.get("elp_net", 0.0))
                    else:
                        elp_pred = float(pred[0])
                
                scored.append((float(elp_pred), r))
            except Exception:
                continue
        
        if not scored:
            logger.warning("Could not score any records; falling back to entropy")
            return select_entropy(records, k)
        
        # Sort descending by predicted ELP
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]
        
    except Exception as e:
        logger.warning(f"Predictor selection failed: {e}; falling back to entropy")
        return select_entropy(records, k)


def select_oracle(
    records: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Select top-K by true ELP_net (oracle upper bound)."""
    scored = []
    for r in records:
        elp = get_field(r, "elp_net", "ELP_net", "elp", "cpt_net", default=None)
        if elp is not None:
            scored.append((float(elp), r))
    
    if not scored:
        logger.warning("No ELP values found; falling back to entropy")
        return select_entropy(records, k)
    
    # Sort descending by ELP
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def prepare_training_examples(
    records: List[Dict[str, Any]],
    supervision_format: str,
) -> List[Dict[str, str]]:
    """Convert selected records to training examples."""
    router = FormatRouter()
    examples = []
    
    for r in records:
        try:
            snap_dict = r.get("snapshot", r)
            snap = Snapshot.from_dict(snap_dict)
            
            # Get teacher hint if available
            hint = get_field(r, "teacher_hint", "hint", default=None)
            
            # Format based on supervision type
            result = router.format(
                snapshot=snap,
                supervision_type=supervision_format,
                teacher_hint=hint,
            )
            
            if result and "input" in result and "completion" in result:
                examples.append({
                    "input": result["input"],
                    "output": result["completion"],
                })
        except Exception as e:
            logger.debug(f"Failed to format example: {e}")
            continue
    
    return examples


def train_model(
    examples: List[Dict[str, str]],
    output_dir: Path,
    config: SelectionExperimentConfig,
) -> Dict[str, Any]:
    """Train a LoRA SFT model on selected examples."""
    if config.mock_training:
        # Mock result for testing
        return {
            "status": "mock",
            "n_examples": len(examples),
            "output_dir": str(output_dir),
        }
    
    sft_config = SFTConfig(
        base_model=config.base_model,
        output_dir=str(output_dir / "checkpoints"),
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
    )
    
    trainer = SFTTrainer(sft_config)
    result = trainer.train(examples)
    
    return {
        "status": "trained",
        "n_examples": len(examples),
        "output_dir": str(output_dir),
        "training_result": result,
    }


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_selection_method(
    method: str,
    records: List[Dict[str, Any]],
    config: SelectionExperimentConfig,
    predictor: Any = None,
) -> Dict[str, Any]:
    """Run one selection method: select, train, return metadata."""
    logger.info(f"Running selection method: {method}")
    
    method_dir = Path(config.output_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # Select snapshots
    if method == "random":
        selected = select_random(records, config.budget_k, config.seed)
    elif method == "entropy":
        selected = select_entropy(records, config.budget_k)
    elif method == "predictor":
        selected = select_predictor(records, config.budget_k, predictor)
    elif method == "oracle":
        selected = select_oracle(records, config.budget_k)
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    logger.info(f"  Selected {len(selected)} snapshots")
    
    # Analyze selected set
    stats = analyze_selected(selected)
    
    # Save selected snapshot IDs
    selected_ids = [
        {
            "snapshot_id": get_field(r, "snapshot_id", default=str(i)),
            "quadrant": get_field(r, "quadrant", default=None),
            "uncertainty": get_field(r, "uncertainty", "U", default=None),
            "elp_net": get_field(r, "elp_net", "ELP_net", default=None),
        }
        for i, r in enumerate(selected)
    ]
    with open(method_dir / "selected_snapshots.json", "w") as f:
        json.dump(selected_ids, f, indent=2)
    
    # Prepare training examples
    examples = prepare_training_examples(selected, config.supervision_format)
    logger.info(f"  Prepared {len(examples)} training examples")
    
    # Save training data
    with open(method_dir / "training_data.json", "w") as f:
        json.dump(examples, f, indent=2)
    
    # Train model
    train_result = train_model(examples, method_dir, config)
    
    # Save training result
    with open(method_dir / "training_result.json", "w") as f:
        json.dump(train_result, f, indent=2)
    
    # Save training summary (for eval suite compatibility)
    training_summary = {
        "model_paths": {
            f"selection_{method}": str(method_dir / "checkpoints")
        },
        "base_model": config.base_model,
        "method": method,
        "budget_k": config.budget_k,
    }
    with open(method_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    return {
        "method": method,
        "n_selected": len(selected),
        "n_examples": len(examples),
        "stats": stats,
        "output_dir": str(method_dir),
        "training_summary_path": str(method_dir / "training_summary.json"),
    }


def analyze_selected(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for selected records."""
    stats: Dict[str, Any] = {"n": len(records)}
    
    # Quadrant distribution
    quadrants = [get_field(r, "quadrant", default="unknown") for r in records]
    q_counts = {}
    for q in quadrants:
        q_counts[str(q)] = q_counts.get(str(q), 0) + 1
    stats["quadrant_distribution"] = q_counts
    
    # Uncertainty stats
    u_vals = [get_field(r, "uncertainty", "U", default=None) for r in records]
    u_vals = [v for v in u_vals if v is not None]
    if u_vals:
        stats["uncertainty_mean"] = float(np.mean(u_vals))
        stats["uncertainty_std"] = float(np.std(u_vals))
    
    # ELP stats
    elp_vals = [get_field(r, "elp_net", "ELP_net", default=None) for r in records]
    elp_vals = [v for v in elp_vals if v is not None]
    if elp_vals:
        stats["elp_mean"] = float(np.mean(elp_vals))
        stats["elp_std"] = float(np.std(elp_vals))
    
    return stats


def write_comparison_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write comparison CSV for all methods."""
    fieldnames = [
        "method",
        "n_selected",
        "n_examples",
        "uncertainty_mean",
        "elp_mean",
        "training_summary_path",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "method": r["method"],
                "n_selected": r["n_selected"],
                "n_examples": r["n_examples"],
                "uncertainty_mean": r["stats"].get("uncertainty_mean", ""),
                "elp_mean": r["stats"].get("elp_mean", ""),
                "training_summary_path": r["training_summary_path"],
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H7 selection experiment")
    parser.add_argument("--labeled-snapshots", type=Path, required=True)
    parser.add_argument("--predictor-path", type=Path, default=None)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase5/selection"))
    
    parser.add_argument("--budget", type=int, default=500, help="Selection budget K")
    parser.add_argument("--methods", nargs="+", default=["random", "entropy", "predictor", "oracle"])
    parser.add_argument("--supervision-format", type=str, default="demo")
    
    # Training hyperparams
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-training", action="store_true", help="Skip actual training (for testing)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Set seeds
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config
    config = SelectionExperimentConfig(
        labeled_snapshots_path=str(args.labeled_snapshots),
        predictor_path=str(args.predictor_path) if args.predictor_path else None,
        base_model=args.base_model,
        output_dir=str(output_dir),
        budget_k=args.budget,
        methods=args.methods,
        supervision_format=args.supervision_format,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        mock_training=args.mock_training,
    )
    
    # Save config
    with open(output_dir / "selection_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load data
    logger.info(f"Loading labeled snapshots from {args.labeled_snapshots}")
    records = load_labeled_records(args.labeled_snapshots)
    logger.info(f"Loaded {len(records)} labeled records")
    
    # Load predictor if needed
    predictor = None
    if "predictor" in config.methods and args.predictor_path:
        predictor = load_predictor(args.predictor_path)
    
    # Run each selection method
    results = []
    for method in config.methods:
        try:
            result = run_selection_method(method, records, config, predictor)
            results.append(result)
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            continue
    
    # Write comparison CSV
    write_comparison_csv(results, output_dir / "selection_comparison.csv")
    
    # Write summary JSON
    summary = {
        "config": asdict(config),
        "n_input_records": len(records),
        "methods": results,
    }
    with open(output_dir / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Selection experiment complete. Results in {output_dir}")
    logger.info("To evaluate each trained model, run:")
    for r in results:
        path = r["training_summary_path"]
        logger.info(f"  python scripts/phase3/run_eval_suite_v8.py --training-summary {path} ...")


if __name__ == "__main__":
    main()
