# Experimental Plan (v8 aligned)

This document is the **authoritative, code-aligned** execution plan for the v8 teachable-moments pipeline.

It supersedes the older `experimental_plan.md` (which references legacy scripts and outdated flags).

It is written so you can hand it to a new student and they can:
1) run a **small pilot** end-to-end (including a real OpenAI API call), and
2) scale to the full experiment suite **without changing scripts**, only by increasing `N`.

---

## The Research Story

Build an empirically grounded **teachability model** for agent training from failures:

1) **What is a teachable moment?**  
   A failure state where the agent is uncertain *and* a small amount of correct supervision can reliably change outcomes.

2) **How do we identify it cheaply and reliably?**  
   Use cheap observables (uncertainty + leverage + CPT patch gains) and train a predictor.

3) **What do we do with it?**  
   Use teachability to **select** failure data and **choose** supervision format, then fine-tune to close the reliability gap while avoiding catastrophic forgetting.

---

## Phase Overview (1A / 1B / 1C framing)

| Phase | Name | Objective |
|-------|------|-----------|
| **1A** | Quadrant training matrix | Categorize states into 4 quadrants (U×L), train models separately, compare supervision types |
| **1B** | CPT validation | Validate that CPT scores correlate with real measured training utility via micro-training |
| **1C** | Teachability predictor | Train predictor from cheap observables; optionally close the loop with predictor-driven selection (H7) |

---

## Prerequisites

```bash
# Create environment
conda create -n teachable python=3.11 -y
conda activate teachable

# Install dependencies
pip install -e ".[full]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import minigrid; print('MiniGrid OK')"
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"  # For GPT-4o teacher
export WANDB_PROJECT="teachable-moments"  # Optional
export CUDA_VISIBLE_DEVICES=0
```

Optional: create a `.env` file at repo root with the same variables.

---

## Model Choices (Practical Defaults)

### Teacher Model (OpenAI)

We need the OpenAI model to produce:
- A **valid** `suggested_action` from the current `valid_actions`
- A short **diagnosis** / hint / critique (for hint/contrast formats)

| Model | Use Case |
|-------|----------|
| `gpt-4o-mini` | Recommended default (fast + low cost) |
| `gpt-4o` | Fallback for hard cases |

> **Important:** In the "SFT + rationale" condition, use a **short diagnosis** (1–2 sentences). Do **not** generate long chain-of-thought.

Set in `configs/experiment.yaml`:
```yaml
teacher:
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 256
```

### Student Model (Local LoRA fine-tuning)

Pick based on GPU:
| GPU | Recommended Model |
|-----|------------------|
| 24GB | 7B–8B instruct model (best quality) |
| 16GB | 3B instruct model (faster, lower quality but works) |
| CPU-only | Pilots only (do not attempt full training) |

---

## Artifacts at a Glance

| Phase | What | Main script | Key outputs |
|---|---|---|---|
| 0 | Student rollouts + failure snapshots | `run_student_rollouts_v8.py` + `mine_failure_snapshots.py` | `rollouts.json`, `failure_snapshots.json` |
| 1 | Label snapshots (U, L, CPT/ELP, quadrant) + hints | `run_labeling.py` | `labeled_snapshots.jsonl`, `.json` |
| 1b | CPT validation (micro-training) | `run_micro_training_v8.py` | `micro_training_results.json`, `cpt_correlation.json` |
| 2 | Train 14-run matrix (4×3 + 2 baselines) | `run_training_matrix_v8.py` | `training_summary.json` |
| 3 | Evaluate (tasks, failure-panel, retention, stuckness) | `run_eval_suite_v8.py` | `results/phase3/v8/*.csv` |
| 4 | Train teachability predictor | `train_predictor_v8.py` | `training_result.json` |
| 5 | Selection experiment (H7) | `run_selection_experiment_v8.py` | `selection_summary.json` |
| A | Paper artifacts | `generate_figures_v8.py`, `generate_tables_v8.py` | `figures/v8/*`, `tables/v8/*` |

---

## Mandatory Pilots (Run Before Scaling)

### Pilot 0: No API (verify pipeline wiring)

```bash
python scripts/pilots/run_all_pilots_v8.py \
  --out-dir results/pilots/v8_no_api \
  --base-model sshleifer/tiny-gpt2 \
  --teacher-model gpt-4o-mini \
  --n-tasks 2 --max-steps 10 \
  --mock-env --skip-api
```

### Pilot 1: Real API (verify teacher calls + real outputs)

Makes **at least one** real OpenAI call during Phase 1 labeling.

```bash
python scripts/pilots/run_all_pilots_v8.py \
  --out-dir results/pilots/v8_real_api \
  --base-model sshleifer/tiny-gpt2 \
  --teacher-model gpt-4o-mini \
  --n-tasks 2 --max-steps 10 \
  --mock-env
```

### Pilot Pass Criteria

- [ ] `results/pilots/.../phase1_labeling/labeled_snapshots.jsonl` exists and is non-empty
- [ ] `results/pilots/.../phase2_training/training_summary.json` exists
- [ ] `results/pilots/.../phase3_eval/*.csv` exist
- [ ] `results/pilots/.../phase4_predictor/training_result.json` exists

---

## Full-Scale Run (Canonical v8)

### Phase 0: Rollouts & Snapshot Mining

We support two experimental setups with different state sampling distributions:

#### Experiment A: Student-Failure-Centric Pipeline

States sampled from student failure trajectories—natural "stuck" points.

```bash
python scripts/phase0/run_student_rollouts_v8.py \
  --model-path <STUDENT_MODEL_OR_CHECKPOINT> \
  --n-tasks 1000 --max-steps 15 \
  --output results/phase0/rollouts.json

python scripts/phase0/mine_failure_snapshots.py \
  --rollouts results/phase0/rollouts.json \
  --output results/phase0/failure_snapshots.json \
  --max-snapshots 2000
```

#### Experiment B: Expert-Sampled State Distribution

States sampled from expert (GPT-4o) rollouts for broader coverage.

```bash
python scripts/phase0/collect_expert_trajectories.py \
  --n-tasks 500 --output-dir results/phase0/expert_trajectories \
  --teacher-model gpt-4o

python scripts/phase0/mine_snapshots_from_expert_trajectories.py \
  --expert-trajectories results/phase0/expert_trajectories/all_trajectories_*.json \
  --output results/phase0/snapshots_expert.json \
  --sample-rate 0.5 --require-env-state
```

| Experiment | Use Case | State Distribution |
|------------|----------|-------------------|
| **A** (Student) | Teachability at student failure points | Biased toward stuck/confused states |
| **B** (Expert) | Teachability across all decision points | More uniform coverage |

For most research, run **both** and compare labeling distributions in Phase 1.

---

### Phase 1: Labeling (Phase 1A)

#### Single Command (recommended)

```bash
python scripts/phase1/run_labeling.py \
  --snapshots results/phase0/failure_snapshots.json \
  --output-dir results/phase1 \
  --student-checkpoint <STUDENT_MODEL_OR_CHECKPOINT> \
  --expert-checkpoint <EXPERT_MODEL_OR_CHECKPOINT> \
  --teacher-model gpt-4o-mini \
  --assign-quadrants
```

**Key fields produced per snapshot:**
- `U` (uncertainty; default: entropy)
- `leverage.L_local` (actionability via single-step forcing)
- `cpt.ELP_net` + `cpt.route_net`
- `quadrant` ∈ {Q1_highU_highL, Q2_highU_lowL, Q3_lowU_lowL, Q4_lowU_highL}

**Pass criteria:** At least ~100 snapshots per quadrant. If not, collect more rollouts.

#### Smoke test (no models/env/teacher)

```bash
python scripts/phase1/run_labeling.py \
  --snapshots results/phase0/snapshots.jsonl \
  --output-dir results/phase1_smoke \
  --mock-env --mock-policy --mock-teacher \
  --skip-leverage --skip-cpt
```

#### Stepwise Labeling (for checkpointing)

```bash
# 1) Generate hints
python scripts/phase1/generate_hints.py \
  --input results/phase0/snapshots.jsonl --output-dir results/phase1

# 2) Compute uncertainty
python scripts/phase1/compute_uncertainty.py \
  --input results/phase1/labeled_snapshots.jsonl --output-dir results/phase1 \
  --student-checkpoint <STUDENT>

# 3) Compute leverage
python scripts/phase1/compute_leverage.py \
  --input results/phase1/labeled_snapshots.jsonl --output-dir results/phase1 \
  --student-checkpoint <STUDENT> --expert-checkpoint <EXPERT>

# 4) Run CPT
python scripts/phase1/run_cpt.py \
  --input results/phase1/labeled_snapshots.jsonl --output-dir results/phase1 \
  --student-checkpoint <STUDENT>
```

#### Manual Threshold Selection Workflow

**1) Plot the distributions:**

```bash
python scripts/analysis/plot_quadrant_thresholds.py \
  --input results/phase1/labeled_snapshots.jsonl \
  --output-dir results/phase1/threshold_plots
```

Outputs: `hist_U.png`, `hist_L.png`, `scatter_U_L.png`, `quantiles.json`

**2) Pick thresholds and assign quadrants:**

```bash
python scripts/phase1/assign_quadrants.py \
  --input results/phase1/labeled_snapshots.jsonl --output-dir results/phase1 \
  --U-threshold <YOUR_CHOSEN_U> --L-threshold <YOUR_CHOSEN_L>
```

---

### Phase 1B: CPT Validation (Micro-Training)

```bash
python scripts/phase1b/run_micro_training_v8.py \
  --labeled results/phase1/labeled_snapshots.json \
  --base-model <STUDENT_MODEL_OR_CHECKPOINT> \
  --output-dir results/phase1b \
  --n-per-quadrant 50 \
  --n-steps 2 \
  --learning-rate 1e-3 \
  --n-validation-rollouts 3 \
  --rollout-max-steps 15
```

**Pass criteria:**
- Positive correlation overall (target: **Spearman ≥ 0.3**, stretch: ≥ 0.5)
- Q1 should show clearly higher utility than Q2 on average
- Exit code 0: proceed to Phase 2; Exit code 1: revise CPT methodology

---

### Phase 2: Training Matrix (Phase 1A continued)

**Supervision type mapping to v8 run_ids:**

| Your concept | v8 run_id suffix | Description |
|--------------|------------------|-------------|
| Recovery trajectory imitation (SFT) | `*_demo` | Standard SFT on expert action |
| SFT + rationale | `*_hint` | Short diagnosis in prompt; target is action |
| Contrastive supervision | `*_contrast` | Contrastive text in prompt; still SFT |

> **Note on DPO:** The v8 training matrix is **SFT-only** for clean comparisons. If you want "contrastive pair DPO", treat it as an **optional extension** after the SFT matrix is complete.

```bash
python scripts/phase2/run_training_matrix_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --config configs/experiment.yaml \
  --base-model <BASE_MODEL> \
  --output-dir results/phase2
```

Trains 14 models: 12 quadrant×supervision + 2 baselines.

---

### Phase 3: Evaluation

```bash
python scripts/phase3/run_eval_suite_v8.py \
  --training-summary results/phase2/training_summary.json \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --output-dir results/phase3/v8 \
  --n-tasks 200 --n-snapshots-per-quadrant 100 --max-steps 15
```

**Key metrics to verify:**
- End-to-end success rate > baseline
- Transfer diagonal > off-diagonal
- No catastrophic forgetting (retention drop < 10%)

**Outputs (CSV):**
- `overall_task_results.csv`
- `per_quadrant_results.csv`
- `transfer_matrix.csv`
- `retention_results.csv`
- `stuckness_results.csv`

**Deliverable (Phase 1A):** Figures/tables showing:
- Q1 (high U, high L) is where supervision yields the biggest gain
- Q2 (high U, low L) is largely "lost causes"
- Best supervision format depends on quadrant
- There exists a reliability–retention frontier (gain without forgetting)

---

### Phase 4: Predictor (Phase 1C)

```bash
python scripts/phase4/train_predictor_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --output-dir results/phase4 \
  --embedder hashing \
  --embedding-dim 256 \
  --epochs 30 \
  --learning-rate 1e-3
```

**Required performance:**
- Uncertainty prediction: R² > 0.6
- Leverage prediction: R² > 0.5
- Quadrant classification: Accuracy > 0.7
- Ranking: Precision@K / NDCG@K beating entropy baseline

---

### Phase 5: Selection Experiment (H7)

Compares selection methods at fixed training budget K:
- **RANDOM**: Uniform sampling
- **ENTROPY**: Top-K by uncertainty U(s)
- **PREDICTOR**: Top-K by predicted ELP
- **ORACLE**: Top-K by true ELP_net (upper bound)

```bash
# Run selection experiment
python scripts/phase5/run_selection_experiment_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --predictor-path results/phase4/predictor/model.pt \
  --base-model <BASE_MODEL> \
  --budget 500 \
  --output-dir results/phase5/selection

# Evaluate each selection method's trained model
for method in random entropy predictor oracle; do
  python scripts/phase3/run_eval_suite_v8.py \
    --training-summary results/phase5/selection/$method/training_summary.json \
    --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
    --output-dir results/phase5/selection/$method/eval
done
```

**Pass/Fail criteria:**
- Predictor-selected > entropy-selected > random-selected on reliability gain
- Retention comparable (no method causes large drop)

---

### Paper Figures and Tables

```bash
python scripts/analysis/generate_figures_v8.py --results-dir results --output-dir figures/v8
python scripts/analysis/generate_tables_v8.py  --results-dir results --output-dir tables/v8
```

---

## Directory Structure After Completion

```
teachable-moments/
├── results/
│   ├── phase0/
│   │   ├── rollouts.json
│   │   └── failure_snapshots.json
│   ├── phase1/
│   │   ├── labeled_snapshots.jsonl
│   │   ├── labeled_snapshots.json
│   │   ├── quadrant_thresholds_auto.json
│   │   └── threshold_plots/
│   ├── phase1b/
│   │   ├── micro_training_results.json
│   │   └── cpt_correlation.json
│   ├── phase2/
│   │   ├── training_summary.json
│   │   └── <run_id>/ (per training run)
│   ├── phase3/v8/
│   │   ├── overall_task_results.csv
│   │   ├── per_quadrant_results.csv
│   │   ├── transfer_matrix.csv
│   │   ├── retention_results.csv
│   │   └── stuckness_results.csv
│   ├── phase4/
│   │   └── predictor/
│   └── phase5/
│       └── selection/
│           ├── selection_summary.json
│           ├── random/
│           ├── entropy/
│           ├── predictor/
│           └── oracle/
├── figures/v8/
└── tables/v8/
```

---

## Common Gotchas Checklist

- **Determinism for eval:** Run eval with temperature=0 for student models
- **Quadrant balance:** If one quadrant is tiny, thresholds are wrong OR you need more rollouts
- **Teacher validity:** Teacher must output an action in `valid_actions` exactly; use structured output / strict parsing
- **CPT noise:** If CPT looks random, increase `n_per_condition` from 2→3 for the validation panel only
- **Micro-training too weak:** If Δ is ~0 everywhere, increase LR (1e-3) and/or steps (2→4) for validation only

---

## Troubleshooting

**Out of memory during training:**
```bash
python scripts/phase2/run_training_matrix_v8.py --batch-size 8 ...
python scripts/phase2/run_training_matrix_v8.py --gradient-accumulation-steps 4 ...
```

**CPT validation fails:**
- Check data quality and distribution
- Adjust uncertainty/leverage thresholds in config
- Increase micro-training samples

**Low transfer performance:**
- Verify quadrant assignments are balanced
- Check for data leakage between train/test
- Consider domain-specific supervision adjustments

---

## Estimated Total Time

| Phase | GPU Hours | Wall Clock |
|-------|-----------|------------|
| Phase 0 | 0.5-1 | 1-2 hours |
| Phase 1 | 1-2 | 2-3 hours |
| Phase 1B | 1-2 | 1-2 hours |
| Phase 2 | 8-12 | 4-8 hours |
| Phase 3 | 2-4 | 2-4 hours |
| Phase 4 | 1-2 | 1-2 hours |
| Phase 5 | 2-4 | 2-4 hours |
| Analysis | 0 | 10 min |
| **Total** | **16-27** | **13-25 hours** |

---

## Legacy Notes

The repo includes older scripts under `scripts/phase*/` that assume different schemas and checkpoint layouts. For v8 runs, prefer the scripts listed above (with `_v8` suffix where applicable).
