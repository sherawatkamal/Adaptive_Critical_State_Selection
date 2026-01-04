import json
import subprocess
import sys
import pytest
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

@pytest.fixture
def dummy_data(tmp_path):
    """Create dummy data for smoke tests."""
    data_dir = tmp_path / "dummy_data"
    data_dir.mkdir()

    # 1. Student failures for mining
    failures = [
        {
            "trajectory_id": "traj_001",
            "task_id": "task_100",
            "steps": [
                {"observation": "State 1", "action": "click[buy]", "reward": 0.0, "done": False, "env_state_b64": "ZHVtbXk="},
                {"observation": "State 2", "action": "click[buy]", "reward": 0.0, "done": True, "env_state_b64": "ZHVtbXk="}
            ]
        }
    ]
    with open(data_dir / "student_failures.json", "w") as f:
        json.dump(failures, f)

    # 2. Labeled snapshots for training
    # Create enough for splits
    snapshots = []
    for i in range(12):
        snapshots.append({
            "id": f"snap_{i:03d}",
            "snapshot": {
                "id": f"snap_{i:03d}",
                "task_id": f"task_{200+i}",
                "env_state_b64": "ZHVtbXk=",
                "observation": "Product page",
                "valid_actions": ["click[buy]"]
            },
            "quadrant": "Q1_highU_highL",
            "cpt": {"ELP_net": 0.8, "route_net": "demo"},
            "teacher_hint": {
                "suggested_action": "click[buy]",
                "rationale": "It is correct",
                "error_type": "affordance_miss",
                "confidence": "high"
            },
            "instruction_text": "Buy item",
            "uncertainty": {"entropy": 0.5, "margin": 0.1},
            "structural_features": {"trajectory_length": 5, "n_available_actions": 3}
        })
    
    with open(data_dir / "labeled_snapshots.json", "w") as f:
        json.dump(snapshots, f)

    # 3. Panel for selection
    panel = ["snap_001"]
    with open(data_dir / "panel.json", "w") as f:
        json.dump(panel, f)

    return data_dir

def run_script(script_path: Path, args: list[str], cwd: Path = REPO_ROOT):
    """Run a script via subprocess."""
    assert script_path.exists(), f"Script {script_path} not found"
    
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Script failed with code {result.returncode}"
    return result

@pytest.mark.slow
def test_phase0_mining(dummy_data, tmp_path):
    """Test Phase 0: Mine snapshots from student rollouts."""
    script = SCRIPTS_DIR / "phase0/mine_snapshots_from_student_rollouts.py"
    output_file = tmp_path / "mined_snapshots.json"
    
    run_script(script, [
        "--student-rollouts", str(dummy_data / "student_failures.json"),
        "--output", str(output_file),
        "--strategy", "failure_steps",
        "--include-k-before", "1"
    ])
    
    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)
    assert len(data) > 0
    # Check for new schema field
    assert "trajectory_id" in data[0]

@pytest.mark.slow
def test_phase4_predictor(dummy_data, tmp_path):
    """Test Phase 4: Train teachability predictor."""
    script = SCRIPTS_DIR / "phase4/train_predictor.py"
    output_dir = tmp_path / "results_phase4"
    
    run_script(script, [
        "--input", str(dummy_data / "labeled_snapshots.json"),
        "--output-dir", str(output_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--embedder", "mock"
    ])
    
    assert (output_dir / "predictor").exists()
    assert (output_dir / "training_result.json").exists()

@pytest.mark.slow
def test_phase1b_micro(dummy_data, tmp_path):
    """Test Phase 1b: Micro-training (mock env)."""
    script = SCRIPTS_DIR / "phase1b/run_micro_training_v8.py"
    output_dir = tmp_path / "results_phase1b"
    
    run_script(script, [
        "--labeled", str(dummy_data / "labeled_snapshots.json"),
        "--panel", str(dummy_data / "panel.json"),
        "--base-model", "gpt2",
        "--output-dir", str(output_dir),
        "--n-steps", "1",
        "--n-validation-rollouts", "1",
        "--rollout-max-steps", "2",
        "--mock-env",
        "--lora-target-modules", "c_attn"
    ])
    
    assert (output_dir / "micro_training_results.json").exists()
    assert (output_dir / "cpt_correlation.json").exists()

@pytest.mark.slow
def test_phase2_per_quadrant(dummy_data, tmp_path):
    """Test Phase 2: Per-quadrant training."""
    script = SCRIPTS_DIR / "phase2/train_per_quadrant.py"
    output_dir = tmp_path / "results_phase2"
    
    # We set WANDB_MODE=disabled for tests to avoid login prompts
    import os
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    
    cmd = [
        sys.executable, str(script),
        "--input", str(dummy_data / "labeled_snapshots.json"),
        "--base-model", "gpt2",
        "--output-dir", str(output_dir),
        "--n-parallel", "1",
        "--quadrants", "Q1_highU_highL",
        "--epochs", "1",
        "--batch-size", "1",
        "--target-modules", "c_attn"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Script failed with code {result.returncode}"
