"""
Per-quadrant training orchestration.

Main experiment: 4 quadrants × 3 supervision types + 2 baselines = 14 runs.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
from itertools import product
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


QUADRANTS = ["Q1_highU_highL", "Q2_highU_lowL", "Q3_lowU_lowL", "Q4_lowU_highL"]
SUPERVISION_TYPES = ["demo", "contrast", "hint"]


@dataclass
class TrainingRun:
    """Configuration for a single training run."""
    
    run_id: str
    quadrant: str  # "all" for baselines
    supervision: str
    selection: str  # "quadrant_specific", "uniform_random", "all_quadrants"
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "quadrant": self.quadrant,
            "supervision": self.supervision,
            "selection": self.selection,
        }


@dataclass
class TrainingMatrix:
    """Complete training matrix with all runs."""
    
    runs: list[TrainingRun] = field(default_factory=list)
    
    @classmethod
    def create_default(cls) -> "TrainingMatrix":
        """Create the default 14-run training matrix."""
        runs = []
        
        # Per-quadrant runs (12)
        for quadrant, supervision in product(QUADRANTS, SUPERVISION_TYPES):
            runs.append(TrainingRun(
                run_id=f"{quadrant}_{supervision}",
                quadrant=quadrant,
                supervision=supervision,
                selection="quadrant_specific",
            ))
        
        # Baselines (2)
        runs.append(TrainingRun(
            run_id="B1_uniform",
            quadrant="all",
            supervision="demo",
            selection="uniform_random",
        ))
        runs.append(TrainingRun(
            run_id="B2_all",
            quadrant="all",
            supervision="demo",
            selection="all_quadrants",
        ))
        
        return cls(runs=runs)
    
    def to_dict(self) -> list[dict]:
        return [run.to_dict() for run in self.runs]
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingMatrix":
        with open(path) as f:
            data = json.load(f)
        runs = [TrainingRun(**d) for d in data]
        return cls(runs=runs)


def create_training_matrix() -> list[dict]:
    """Generate 12 per-quadrant runs + 2 baselines."""
    return TrainingMatrix.create_default().to_dict()


def sample_uniform(
    quadrant_partitions: dict[str, list],
    n: int = 500,
    seed: int = 42,
) -> list:
    """
    Sample uniformly across quadrants.
    
    Args:
        quadrant_partitions: Dict mapping quadrant labels to snapshot lists
        n: Total number of samples
        seed: Random seed
        
    Returns:
        Uniformly sampled list of snapshots
    """
    import random
    random.seed(seed)
    
    all_snapshots = []
    for snaps in quadrant_partitions.values():
        all_snapshots.extend(snaps)
    
    if len(all_snapshots) <= n:
        return all_snapshots
    
    return random.sample(all_snapshots, n)


def flatten(quadrant_partitions: dict[str, list]) -> list:
    """Flatten all partitions into single list."""
    all_snapshots = []
    for snaps in quadrant_partitions.values():
        all_snapshots.extend(snaps)
    return all_snapshots


def train_single_run(
    run_config: dict,
    quadrant_partitions: dict[str, list],
    base_model_path: str,
    output_dir: str,
    sft_config: dict = None,
) -> str:
    """
    Train a single model according to run configuration.
    
    Args:
        run_config: Run configuration dict
        quadrant_partitions: Dict mapping quadrants to snapshots
        base_model_path: Path to base model
        output_dir: Output directory
        sft_config: Optional SFT configuration
        
    Returns:
        Path to saved model
    """
    from .sft_trainer import train_sft, SFTConfig
    from ..supervision.format_router import generate_supervision
    
    run_id = run_config["run_id"]
    logger.info(f"Starting training run: {run_id}")
    
    # Allow small, deadline-friendly budgets without changing core code.
    # These keys are optional and intentionally *not* part of SFTConfig.
    sft_config = dict(sft_config or {})
    max_train_samples = sft_config.pop("max_train_samples", None)
    baseline_n_samples = sft_config.pop("baseline_n_samples", None)
    data_seed = int(sft_config.get("seed", 42))

    # Select training data
    if run_config["quadrant"] == "all":
        if run_config["selection"] == "uniform_random":
            n = int(baseline_n_samples or max_train_samples or 500)
            training_snapshots = sample_uniform(quadrant_partitions, n=n, seed=data_seed)
        else:
            training_snapshots = flatten(quadrant_partitions)
    else:
        training_snapshots = quadrant_partitions.get(run_config["quadrant"], [])

    # Optional downsampling for tight-deadline runs.
    if max_train_samples is not None and len(training_snapshots) > int(max_train_samples):
        import random
        rnd = random.Random(data_seed)
        training_snapshots = rnd.sample(training_snapshots, int(max_train_samples))
    
    if not training_snapshots:
        logger.warning(f"No training data for run {run_id}")
        return ""
    
    # Generate supervision in specified format
    supervision_examples = generate_supervision(
        training_snapshots,
        format=run_config["supervision"],
    )
    
    training_data = [ex.to_dict() for ex in supervision_examples if (ex.output_text or "").strip()]
    if not training_data:
        logger.warning(
            "All generated supervision examples were empty for run %s. "
            "(Missing teacher hints?) Skipping.",
            run_id,
        )
        return ""
    
    logger.info(f"Training {run_id}: {len(training_data)} examples")
    
    # Create output path
    run_output_dir = Path(output_dir) / run_id
    
    # Train model
    config = SFTConfig(**sft_config) if sft_config else SFTConfig()
    config.base_model = base_model_path
    
    model_path = train_sft(
        base_model=base_model_path,
        training_data=training_data,
        output_dir=str(run_output_dir),
        config=config,
    )
    
    # Save run metadata
    metadata_path = run_output_dir / "run_config.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "run_config": run_config,
            "n_examples": len(training_data),
            "model_path": model_path,
        }, f, indent=2)
    
    logger.info(f"Completed training run: {run_id}")
    
    return model_path


def run_all_training(
    quadrant_partitions: dict[str, list],
    base_model_path: str,
    output_dir: str,
    n_parallel: int = 1,
    sft_config: dict = None,
    run_filter: Optional[Callable[[dict], bool]] = None,
) -> dict[str, str]:
    """
    Run all 14 training experiments.
    
    Args:
        quadrant_partitions: Dict mapping quadrants to snapshots
        base_model_path: Path to base model
        output_dir: Output directory
        n_parallel: Number of parallel training runs
        sft_config: Optional SFT configuration dict
        run_filter: Optional function to filter which runs to execute
        
    Returns:
        Dict mapping run IDs to model paths
    """
    matrix = TrainingMatrix.create_default()
    runs = matrix.runs
    
    if run_filter:
        runs = [r for r in runs if run_filter(r.to_dict())]
    
    logger.info(f"Running {len(runs)} training experiments")
    
    # Save training matrix
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    matrix.save(str(Path(output_dir) / "training_matrix.json"))
    
    model_paths = {}
    
    if n_parallel <= 1:
        # Sequential execution
        for run in runs:
            path = train_single_run(
                run.to_dict(),
                quadrant_partitions,
                base_model_path,
                output_dir,
                sft_config,
            )
            model_paths[run.run_id] = path
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {
                executor.submit(
                    train_single_run,
                    run.to_dict(),
                    quadrant_partitions,
                    base_model_path,
                    output_dir,
                    sft_config,
                ): run.run_id
                for run in runs
            }
            
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    model_path = future.result()
                    model_paths[run_id] = model_path
                    logger.info(f"Completed: {run_id}")
                except Exception as e:
                    logger.error(f"Failed: {run_id} - {e}")
                    model_paths[run_id] = ""
    
    # Save results summary
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_runs": len(runs),
            "n_completed": sum(1 for p in model_paths.values() if p),
            "model_paths": model_paths,
        }, f, indent=2)
    
    return model_paths


def verify_models(model_paths: dict[str, str]) -> dict:
    """
    Verify all trained models exist and are loadable.
    
    Args:
        model_paths: Dict mapping run IDs to model paths
        
    Returns:
        Verification results
    """
    results = {
        "verified": [],
        "missing": [],
        "failed": [],
    }
    
    for run_id, path in model_paths.items():
        if not path:
            results["missing"].append(run_id)
            continue
        
        path = Path(path)
        if not path.exists():
            results["missing"].append(run_id)
            continue
        
        # Try to verify model files exist
        expected_files = ["adapter_config.json", "adapter_model.safetensors"]
        has_files = all((path / f).exists() or (path / f.replace(".safetensors", ".bin")).exists() 
                       for f in expected_files)
        
        if has_files:
            results["verified"].append(run_id)
        else:
            results["failed"].append(run_id)
    
    return results
