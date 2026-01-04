# Teachable Moments from Failures

A research framework for identifying and leveraging "teachable moments" in agent training from failure trajectories.

## Overview

This codebase implements a **teachability model** for agent training that answers three key questions:

1. **What is a teachable moment?** A state where intervention can meaningfully improve agent behavior.
2. **How do we identify it?** Using uncertainty (U) and leverage (L) measurements, organized into quadrants.
3. **What do we do with it?** Apply regime-specific supervision (Demo, Contrast, Hint) based on the state's characteristics.

## Core Concepts

### Teachability Axes

- **Uncertainty U(s)**: How uncertain the agent is at state s (measured via entropy, margin, etc.)
- **Leverage L_local(s)**: How much fixing the current action improves outcomes (Estimator A)
- **Leverage L_upper(s)**: Maximum possible improvement with full expert control (Estimator B)

### Quadrant Framework

States are classified into four quadrants based on U × L_local:

| Quadrant | Uncertainty | Actionability | Interpretation | Recommended Supervision |
|----------|-------------|---------------|----------------|------------------------|
| Q1 | High | High | Uncertain and fixable | Hint/Diagnosis |
| Q2 | High | Low | Uncertain and stuck | Limited intervention |
| Q3 | Low | Low | Confident and stuck | Low ELP expected |
| Q4 | Low | High | Confident but wrong | Contrast |

### Contextual Patch Test (CPT)

A proxy for learning payoff that tests whether in-context teaching signals help the agent recover:
- **Placebo**: Content-neutral control
- **Demo**: Shows correct action with rationale
- **Contrast**: Contrasts wrong vs correct action
- **Hint**: Provides diagnostic insight without action

## Installation

### Prerequisites

- **Python 3.9+** (tested with Python 3.9.6)
- **Java 11+** (required for pyserini/Lucene search in WebShop)
- **Homebrew** (macOS) for installing Java

### Quick Setup (Tested on macOS ARM)

```bash
# Clone the repository
git clone https://github.com/your-org/teachable-moments.git
cd teachable-moments

# Install Java (required for WebShop search engine)
brew install openjdk@11

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip wheel setuptools
pip install torch transformers beautifulsoup4 flask gym numpy pandas pytest PyYAML \
    rank_bm25 requests thefuzz tqdm selenium spacy rich scikit-learn gdown gradio \
    clean-text pyserini sentence-transformers peft

# Set up Java environment variables (add to ~/.zshrc for persistence)
export JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
```

### Linux + CUDA (Known-good tm312 setup)

This repo has a lightweight unit/integration test suite, plus two **opt-in** integration tests for:

- local **Qwen3-8B** fp16 decode smoke (requires CUDA + ~22GB VRAM)
- a tiny **HF download + LoRA SFT** smoke run (may require internet)

The setup that is known to work (as of Jan 2026):

- `torch 2.6.0+cu124` (CUDA 12.4)

Recommended workflow:

```bash
conda create -n tm312 python=3.12 -y
conda activate tm312

pip install -U pip
pip install -r requirements.txt

# If you need CUDA-enabled torch, install the correct wheel for your CUDA stack.
# Verify with:
python scripts/dev/print_env_info.py
```

### Model Setup

Download the required models (e.g., Qwen/Qwen3-8B) using the provided script. 
Note: Qwen/Qwen3-8B is the instruct-tuned version (base model is Qwen/Qwen3-8B-Base).

```bash
# Download Qwen/Qwen3-8B
python scripts/download_models.py --model Qwen/Qwen3-8B
```

### Running Scripts

All scripts should be run with `PYTHONPATH=.`:

```bash
source .venv/bin/activate
export JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
PYTHONPATH=. python scripts/run_full_pipeline.py --dry-run --phases 0 1 2
```

### Verifying Installation

```bash
# Run integration tests
python tests/test_webshop_state_integration.py

# Verify script imports
PYTHONPATH=. python scripts/phase1/compute_leverage.py --help
PYTHONPATH=. python scripts/phase1/run_cpt.py --help
```


## Repository Structure

```
teachable-moments/
├── configs/                    # YAML configuration files
│   ├── data.yaml               # Environment and data configs
│   ├── label.yaml              # Labeling pipeline configs
│   ├── training.yaml           # Training hyperparameters
│   ├── eval.yaml               # Evaluation configs
│   └── simulation.yaml         # Student/teacher rollout configs
├── scripts/                    # Runnable experiment scripts
│   ├── phase0/                 # Data collection via simulation
│   ├── phase1/                 # Teachability characterization
│   ├── phase1b/                # CPT validation via micro-training
│   ├── phase2/                 # Per-quadrant training
│   ├── phase3/                 # Evaluation suite
│   ├── phase4/                 # Predictor training
│   └── analysis/               # Figure and table generation
├── src/                        # Core library code
│   ├── data/                   # Data structures
│   ├── simulation/             # Student/teacher rollout and failure detection
│   ├── label/                  # Labeling algorithms
│   ├── supervision/            # Supervision format generation
│   ├── training/               # Training infrastructure
│   └── eval/                   # Evaluation metrics
├── panels/                     # Fixed evaluation panels
├── results/                    # Output directory
└── tests/                      # Unit tests
```

## Quick Start

### Phase 0: Collect Training Data via Simulation

The first step is discovering where your student model fails. This is where the teachable moments live.

```bash
# Step 0a: Run student model to collect failures
python scripts/phase0/collect_student_failures.py \
    --model-path checkpoints/student_base \
    --n-tasks 500 \
    --output results/phase0/student_failures.json

# Step 0b: Run teacher on failed tasks (for comparison)
python scripts/phase0/collect_teacher_demos.py \
    --model gpt-4o \
    --student-results results/phase0/student_failures.json \
    --failures-only \
    --output results/phase0/teacher_demos.json

# Step 0c: Analyze teachable gaps
python scripts/phase0/analyze_gaps.py \
    --student-results results/phase0/student_failures.json \
    --teacher-results results/phase0/teacher_demos.json \
    --output results/phase0/teachable_gaps.json
```

### Phase 1: Label and Characterize Teachability

```bash
# Compute uncertainty and leverage scores
python scripts/phase1/compute_uncertainty.py
python scripts/phase1/compute_leverage.py

# Assign quadrants based on U x L
python scripts/phase1/assign_quadrants.py

# Run Contextual Patch Test
python scripts/phase1/run_cpt.py

# Generate teacher hints
python scripts/phase1/generate_hints.py

# Analyze label distributions
python scripts/phase1/analyze_labels.py
```

### Phase 1B: Validate CPT via Micro-Training

```bash
python scripts/phase1b/validate_cpt.py \
    --output results/phase1b/validation_report.json
```

### Phase 2: Per-Quadrant Training

```bash
python scripts/phase2/train_per_quadrant.py \
    --config configs/training.yaml \
    --output-dir checkpoints/per_quadrant
```

### Phase 3: Evaluate

```bash
# End-to-end evaluation
python scripts/phase3/evaluate_end2end.py

# Transfer matrix
python scripts/phase3/evaluate_transfer.py

# Full evaluation suite
python scripts/phase3/evaluate_retention.py
python scripts/phase3/evaluate_stuckness.py
```

### Phase 4: Train Teachability Predictor

```bash
python scripts/phase4/train_predictor.py
python scripts/phase4/evaluate_predictor.py
```

## Main Experiment: Per-Quadrant Training

The core experiment trains 14 models:
- 12 quadrant-specific: 4 quadrants × 3 supervision types
- 2 baselines: uniform random, all data combined

Each model is evaluated on:
1. **End-to-end success rate** on held-out tasks
2. **Per-quadrant improvement** (mechanistic validation)
3. **Cross-quadrant transfer** (4×4 matrix)
4. **Stuckness metrics** (loop detection, efficiency)
5. **Retention** (no degradation on successful tasks)

## Configuration

All experiments are configured via YAML files. Key configurations:

### Training (`configs/training/per_quadrant.yaml`)
```yaml
per_quadrant_training:
  base_model: "meta-llama/Llama-3-8B-Instruct"
  lora_rank: 16
  epochs: 3
  batch_size: 8
  learning_rate: 2e-5
```

### Leverage (`configs/label/leverage.yaml`)
```yaml
leverage:
  n_force_rollouts: 7      # Estimator A
  n_expert_rollouts: 2     # Estimator B
```

### CPT (`configs/label/patch_test.yaml`)
```yaml
cpt:
  patch_types: ["base", "placebo", "demo", "contrast", "hint"]
  n_per_condition: 2       # Single-stage allocation
```

## Compute Budget

| Phase | Component | Episodes | GPU-hours | LLM Cost |
|-------|-----------|----------|-----------|----------|
| 1 | Leverage (A+B) | ~18,000 | - | - |
| 1 | CPT (n=2/condition) | ~20,000 | - | - |
| 1 | Teacher hints | - | - | $30 |
| 1B | Micro-training validation | 200 runs | 3 | - |
| 2 | Model training | 14 runs | 28 | - |
| 3 | Evaluation suite | ~40,000 | - | - |
| **Total** | | **~80,000** | **~31** | **~$30** |

## Key Outputs

### Figures
- `results/figures/quadrant_scatter.pdf`: U × L scatter with quadrant boundaries
- `results/figures/cpt_validation_scatter.pdf`: ELP vs micro-training improvement
- `results/figures/per_quadrant_results.pdf`: Success by quadrant and supervision
- `results/figures/transfer_matrix.pdf`: Cross-quadrant transfer heatmap

### Tables
- `results/phase3/per_quadrant_results.csv`: Main results table
- `results/phase3/transfer_matrix.csv`: Transfer matrix data
- `results/phase3/stuckness_results.csv`: Stuckness metrics

## Decision Gates

| Phase | Gate | Pass Condition | Fail Action |
|-------|------|----------------|-------------|
| 1B | CPT validity | ρ > 0.3 | Use L_local as primary signal |
| 2 | Quadrant balance | N > 300 per quadrant | May merge Q2+Q3 |
| 3 | Supervision differentiation | Best ≠ Demo in ≥2 quadrants | Still informative |
| 3 | Retention | Δ > -3% | Adjust learning rate |

## Development

### Running Tests
```bash
# Cheap (default) suite
pytest -q

# Opt-in: local Qwen3-8B decode smoke (requires models/Qwen3-8B)
RUN_QWEN_SMOKE=1 pytest -q tests/integration/test_qwen3_8b_decode_action_smoke.py

# Opt-in: HF download + tiny LoRA SFT smoke
RUN_SLOW_TESTS=1 pytest -q tests/integration/test_train_one_run_slow.py

# Convenience runner (runs all of the above in your current env)
bash scripts/dev/run_test_matrix.sh
```

### OpenAI / teacher hints (avoiding accidental spend)

- Teacher hint generation uses `src.teacher.client.TeacherClient`, which imports `openai` lazily.
- Tests do **not** call OpenAI by default.
- Scripts that can call OpenAI generally provide a switch to avoid API calls:
  - `scripts/phase1/run_labeling.py`: use `--mock-teacher`
  - `scripts/phase1/generate_hints.py`: use `--mock-teacher`
  - `scripts/pilots/run_all_pilots_v8.py`: use `--skip-api` (uses mock teacher)

If you want to run the pilots / real teacher hints, export `OPENAI_API_KEY` (or put it in a local `.env` file; note `.env` is gitignored).

### Code Style
```bash
black src/ scripts/
ruff check src/ scripts/
```

## Citation

If you use this code, please cite:

```bibtex
@article{teachable_moments_2025,
  title={Teachable Moments from Failures: A Framework for Regime-Aware Agent Training},
  author={Your Name},
  year={2025}
}
```

## License

MIT License. See LICENSE for details.
