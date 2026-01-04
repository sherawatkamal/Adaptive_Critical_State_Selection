#!/usr/bin/env python3
"""Validate CPT methodology end-to-end.

This script orchestrates the full CPT validation pipeline:
1. Run micro-training experiments
2. Analyze CPT-ELP correlation
3. Generate validation report and go/no-go decision
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.utils import setup_logging, save_json, get_timestamp


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    logger.info(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            return False
        
        logger.info(f"  Completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Command timed out")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def check_correlation_results(results_path: Path) -> dict:
    """Check correlation analysis results and determine if validation passed."""
    if not results_path.exists():
        return {"passed": False, "reason": "Results file not found"}
    
    with open(results_path) as f:
        results = json.load(f)
    
    overall = results.get("overall_correlation", {})
    pearson_r = overall.get("pearson_r", 0)
    pearson_p = overall.get("pearson_p", 1)
    
    # Validation criteria
    min_correlation = 0.5
    max_p_value = 0.05
    
    if pearson_r >= min_correlation and pearson_p < max_p_value:
        return {
            "passed": True,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "reason": f"Correlation r={pearson_r:.3f} meets threshold of {min_correlation}",
        }
    else:
        reasons = []
        if pearson_r < min_correlation:
            reasons.append(f"correlation {pearson_r:.3f} < {min_correlation}")
        if pearson_p >= max_p_value:
            reasons.append(f"p-value {pearson_p:.4f} >= {max_p_value}")
        
        return {
            "passed": False,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "reason": "; ".join(reasons),
        }


def main():
    parser = argparse.ArgumentParser(description="Validate CPT methodology")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase1/labeled_snapshots.json"),
        help="Path to labeled snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1b"),
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model for micro-training",
    )
    parser.add_argument(
        "--n-per-quadrant",
        type=int,
        default=10,
        help="Snapshots per quadrant for validation",
    )
    parser.add_argument(
        "--micro-steps",
        type=int,
        default=50,
        help="Micro-training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-micro-training",
        action="store_true",
        help="Skip micro-training if results exist",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    micro_results_path = args.output_dir / "micro_training_results.json"
    correlation_results_path = args.output_dir / "cpt_correlation_analysis.json"
    
    # Step 1: Run micro-training
    if args.skip_micro_training and micro_results_path.exists():
        logger.info("Skipping micro-training (results exist)")
    else:
        success = run_command(
            [
                sys.executable,
                "scripts/phase1b/run_micro_training.py",
                "--input", str(args.input),
                "--output-dir", str(args.output_dir),
                "--model", args.model,
                "--n-per-quadrant", str(args.n_per_quadrant),
                "--micro-steps", str(args.micro_steps),
                "--seed", str(args.seed),
            ],
            "Micro-training experiments",
        )
        
        if not success:
            logger.error("Micro-training failed")
            sys.exit(1)
    
    # Step 2: Analyze correlation
    success = run_command(
        [
            sys.executable,
            "scripts/phase1b/analyze_cpt_correlation.py",
            "--input", str(micro_results_path),
            "--output-dir", str(args.output_dir),
        ],
        "CPT-ELP correlation analysis",
    )
    
    if not success:
        logger.error("Correlation analysis failed")
        sys.exit(1)
    
    # Step 3: Check results and generate final report
    validation = check_correlation_results(correlation_results_path)
    
    final_report = {
        "timestamp": get_timestamp(),
        "config": {
            "input": str(args.input),
            "model": args.model,
            "n_per_quadrant": args.n_per_quadrant,
            "micro_steps": args.micro_steps,
            "seed": args.seed,
        },
        "validation": validation,
    }
    
    save_json(final_report, args.output_dir / "validation_report.json")
    
    # Print final result
    print("\n" + "=" * 60)
    print("CPT VALIDATION RESULT")
    print("=" * 60)
    
    if validation["passed"]:
        print(f"\n[PASS] CPT validation successful")
        print(f"  Pearson r: {validation['pearson_r']:.4f}")
        print(f"  P-value: {validation['pearson_p']:.4e}")
        print(f"\nCPT is a valid proxy for ELP. Proceed to Phase 2.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] CPT validation failed")
        print(f"  Reason: {validation['reason']}")
        print(f"\nReview CPT methodology before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
