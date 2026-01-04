#!/usr/bin/env python3
"""
Full Pipeline Orchestration Script.

Runs the complete teachable moments experiment pipeline:
1. Phase 0: Trajectory collection
2. Phase 1: Labeling (uncertainty, leverage, CPT)
3. Phase 1B: CPT validation via micro-training
4. Phase 2: Per-quadrant training (14 runs)
5. Phase 3: Evaluation suite
6. Phase 4: Predictor training (optional)
7. Analysis: Figures and tables

Usage:
    # Dry run (show what would be done)
    python scripts/run_full_pipeline.py --dry-run
    
    # Run specific phases
    python scripts/run_full_pipeline.py --phases 0 1 2
    
    # Full run
    python scripts/run_full_pipeline.py --config configs/pipeline.yaml
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for full pipeline run."""
    
    # General
    output_dir: str = "results"
    seed: int = 42
    dry_run: bool = False
    
    # Phase 0: Trajectory collection
    n_student_trajectories: int = 1000
    n_expert_trajectories: int = 500
    student_model: str = "meta-llama/Llama-3-8B-Instruct"
    expert_model: str = "gpt-4o"
    
    # Phase 1: Labeling
    n_snapshots: int = 2000
    leverage_budget: int = 9  # 7 for p_force, 2 for p_expert
    cpt_budget: int = 10  # 5 conditions x 2 episodes
    
    # Phase 1B: CPT validation
    validation_panel_size: int = 200
    n_per_quadrant: int = 50
    
    # Phase 2: Training
    training_samples_per_quadrant: int = 500
    num_epochs: int = 3
    
    # Phase 3: Evaluation
    eval_episodes: int = 200
    
    # Phases to run
    phases: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("pipeline", data))


class PipelineRunner:
    """
    Orchestrates the full experiment pipeline.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.results = {}
        self.timing = {}
    
    def run(self):
        """Run all configured phases."""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting Teachable Moments Pipeline")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Phases to run: {self.config.phases}")
        logger.info(f"Dry run: {self.config.dry_run}")
        
        try:
            for phase in self.config.phases:
                phase_start = time.time()
                
                if phase == 0:
                    self._run_phase0_trajectory_collection()
                elif phase == 1:
                    self._run_phase1_labeling()
                elif phase == 2:
                    self._run_phase2_training()
                elif phase == 3:
                    self._run_phase3_evaluation()
                elif phase == 4:
                    self._run_phase4_predictor()
                else:
                    logger.warning(f"Unknown phase: {phase}")
                    continue
                
                self.timing[f"phase_{phase}"] = time.time() - phase_start
            
            # Generate analysis outputs
            if 3 in self.config.phases:
                self._run_analysis()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        total_time = time.time() - start_time
        self.timing["total"] = total_time
        
        # Save summary
        self._save_summary()
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info("=" * 60)
    
    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command, respecting dry_run mode."""
        logger.info(f"\n>>> {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Would execute above command")
            return True
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False
    
    def _run_phase0_trajectory_collection(self):
        """Phase 0: Collect student and expert trajectories."""
        logger.info("\n" + "=" * 40)
        logger.info("Phase 0: Trajectory Collection")
        logger.info("=" * 40)
        
        phase0_dir = self.output_dir / "phase0"
        phase0_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect student failures
        self._run_command(
            [
                "python", "scripts/phase0/collect_student_failures.py",
                "--n-tasks", str(self.config.n_student_trajectories),
                "--output-dir", str(phase0_dir / "student_trajectories"),
                "--model", self.config.student_model,
                "--seed", str(self.config.seed),
            ],
            "Collecting student failure trajectories"
        )
        
        # Collect expert trajectories (for EEF comparison)
        self._run_command(
            [
                "python", "scripts/phase0/collect_expert_trajectories.py",
                "--n-tasks", str(self.config.n_expert_trajectories),
                "--output-dir", str(phase0_dir / "expert_trajectories"),
                "--teacher-model", self.config.expert_model,
                "--seed", str(self.config.seed),
            ],
            "Collecting expert trajectories"
        )
        
        # Extract snapshots
        self._run_command(
            [
                "python", "scripts/phase0/analyze_gaps.py",
                "--trajectories", str(phase0_dir / "student_trajectories"),
                "--output", str(phase0_dir / "raw_snapshots.json"),
                "--n-snapshots", str(self.config.n_snapshots),
            ],
            "Extracting snapshots from trajectories"
        )
    
    def _run_phase1_labeling(self):
        """Phase 1: Compute labels (U, L, CPT)."""
        logger.info("\n" + "=" * 40)
        logger.info("Phase 1: Labeling")
        logger.info("=" * 40)
        
        phase0_dir = self.output_dir / "phase0"
        phase1_dir = self.output_dir / "phase1"
        phase1_dir.mkdir(parents=True, exist_ok=True)
        
        snapshots_path = phase0_dir / "raw_snapshots.json"
        
        # Compute uncertainty
        self._run_command(
            [
                "python", "scripts/phase1/compute_uncertainty.py",
                "--snapshots", str(snapshots_path),
                "--output", str(phase1_dir / "uncertainty_labels.json"),
                "--model", self.config.student_model,
            ],
            "Computing uncertainty (entropy, margin)"
        )
        
        # Compute leverage
        self._run_command(
            [
                "python", "scripts/phase1/compute_leverage.py",
                "--snapshots", str(phase1_dir / "uncertainty_labels.json"),
                "--output", str(phase1_dir / "leverage_labels.json"),
                "--budget", str(self.config.leverage_budget),
            ],
            "Computing leverage (L_local, L_upper)"
        )
        
        # Assign quadrants
        self._run_command(
            [
                "python", "scripts/phase1/assign_quadrants.py",
                "--snapshots", str(phase1_dir / "leverage_labels.json"),
                "--output", str(phase1_dir / "quadrant_labels.json"),
            ],
            "Assigning quadrants"
        )
        
        # Generate teacher hints
        self._run_command(
            [
                "python", "scripts/phase1/generate_hints.py",
                "--snapshots", str(phase1_dir / "quadrant_labels.json"),
                "--output", str(phase1_dir / "hints_labels.json"),
                "--teacher-model", self.config.expert_model,
            ],
            "Generating teacher hints"
        )
        
        # Run CPT
        self._run_command(
            [
                "python", "scripts/phase1/run_cpt.py",
                "--snapshots", str(phase1_dir / "hints_labels.json"),
                "--output", str(phase1_dir / "cpt_labels.json"),
                "--budget", str(self.config.cpt_budget),
            ],
            "Running CPT (contextual patch test)"
        )
        
        # Extract features
        self._run_command(
            [
                "python", "scripts/phase1/extract_features_batch.py",
                "--snapshots", str(phase1_dir / "cpt_labels.json"),
                "--output", str(phase1_dir / "labeled_snapshots.parquet"),
            ],
            "Extracting features for predictor"
        )
        
        # CPT validation (Phase 1B)
        logger.info("\n>>> Phase 1B: CPT Validation")
        
        phase1b_dir = self.output_dir / "phase1b"
        phase1b_dir.mkdir(parents=True, exist_ok=True)
        
        self._run_command(
            [
                "python", "scripts/phase1b/run_micro_training_validation.py",
                "--panel", str(phase1_dir / "labeled_snapshots.parquet"),
                "--base-model", self.config.student_model,
                "--output", str(phase1b_dir / "cpt_validation_results.json"),
            ],
            "Validating CPT via micro-training"
        )
    
    def _run_phase2_training(self):
        """Phase 2: Per-quadrant training."""
        logger.info("\n" + "=" * 40)
        logger.info("Phase 2: Per-Quadrant Training")
        logger.info("=" * 40)
        
        phase1_dir = self.output_dir / "phase1"
        phase2_dir = self.output_dir / "phase2"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        
        # Train per-quadrant models (14 runs)
        self._run_command(
            [
                "python", "scripts/phase2/train_per_quadrant.py",
                "--snapshots", str(phase1_dir / "labeled_snapshots.parquet"),
                "--output-dir", str(phase2_dir),
                "--base-model", self.config.student_model,
                "--n-samples", str(self.config.training_samples_per_quadrant),
                "--epochs", str(self.config.num_epochs),
            ],
            "Training 14 per-quadrant models"
        )
        
        # Train baselines
        self._run_command(
            [
                "python", "scripts/phase2/train_baselines.py",
                "--snapshots", str(phase1_dir / "labeled_snapshots.parquet"),
                "--output-dir", str(phase2_dir / "baselines"),
                "--base-model", self.config.student_model,
            ],
            "Training baseline models"
        )
    
    def _run_phase3_evaluation(self):
        """Phase 3: Evaluation suite."""
        logger.info("\n" + "=" * 40)
        logger.info("Phase 3: Evaluation")
        logger.info("=" * 40)
        
        phase2_dir = self.output_dir / "phase2"
        phase3_dir = self.output_dir / "phase3"
        phase3_dir.mkdir(parents=True, exist_ok=True)
        
        # End-to-end evaluation
        self._run_command(
            [
                "python", "scripts/phase3/evaluate_end2end.py",
                "--checkpoints-dir", str(phase2_dir),
                "--output", str(phase3_dir / "end2end_results.json"),
                "--n-episodes", str(self.config.eval_episodes),
            ],
            "Running end-to-end evaluation"
        )
        
        # Transfer matrix
        self._run_command(
            [
                "python", "scripts/phase3/evaluate_transfer.py",
                "--checkpoints-dir", str(phase2_dir),
                "--output", str(phase3_dir / "transfer_matrix.csv"),
                "--n-episodes", str(self.config.eval_episodes // 4),
            ],
            "Computing transfer matrix"
        )
        
        # Retention curves
        self._run_command(
            [
                "python", "scripts/phase3/evaluate_retention.py",
                "--checkpoints-dir", str(phase2_dir),
                "--output", str(phase3_dir / "retention_results.csv"),
            ],
            "Computing retention curves"
        )
        
        # Stuckness metrics
        self._run_command(
            [
                "python", "scripts/phase3/evaluate_stuckness.py",
                "--checkpoints-dir", str(phase2_dir),
                "--output", str(phase3_dir / "stuckness_results.csv"),
            ],
            "Computing stuckness metrics"
        )
    
    def _run_phase4_predictor(self):
        """Phase 4: Train teachability predictor."""
        logger.info("\n" + "=" * 40)
        logger.info("Phase 4: Predictor Training")
        logger.info("=" * 40)
        
        phase1_dir = self.output_dir / "phase1"
        phase4_dir = self.output_dir / "phase4"
        phase4_dir.mkdir(parents=True, exist_ok=True)
        
        self._run_command(
            [
                "python", "scripts/phase4/train_predictor.py",
                "--features", str(phase1_dir / "labeled_snapshots.parquet"),
                "--output-dir", str(phase4_dir),
            ],
            "Training teachability predictor"
        )
    
    def _run_analysis(self):
        """Generate figures and tables."""
        logger.info("\n" + "=" * 40)
        logger.info("Analysis: Figures and Tables")
        logger.info("=" * 40)
        
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate figures
        self._run_command(
            [
                "python", "scripts/analysis/generate_figures.py",
                "--results-dir", str(self.output_dir),
                "--output-dir", str(analysis_dir / "figures"),
            ],
            "Generating figures"
        )
        
        # Generate tables
        self._run_command(
            [
                "python", "scripts/analysis/generate_tables.py",
                "--results-dir", str(self.output_dir),
                "--output-dir", str(analysis_dir / "tables"),
            ],
            "Generating tables"
        )
    
    def _save_summary(self):
        """Save pipeline summary."""
        summary = {
            "config": {
                "output_dir": str(self.output_dir),
                "phases": self.config.phases,
                "dry_run": self.config.dry_run,
            },
            "timing": self.timing,
            "timestamp": datetime.now().isoformat(),
        }
        
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full teachable moments experiment pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Pipeline configuration YAML file",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Phases to run (0=collection, 1=labeling, 2=training, 3=evaluation, 4=predictor)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Load config
    if args.config:
        config = PipelineConfig.from_yaml(str(args.config))
    else:
        config = PipelineConfig()
    
    # Override with command line args
    config.phases = args.phases
    config.output_dir = str(args.output_dir)
    config.dry_run = args.dry_run
    config.seed = args.seed
    
    # Run pipeline
    runner = PipelineRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
