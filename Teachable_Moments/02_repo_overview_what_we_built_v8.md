# Repo Overview (v8 aligned)

This repo implements the **v8 teachable moments** pipeline for training WebShop agents from failures.

---

## What is "v8" in this codebase?

There are **two generations of code** in this repo:

| Generation | Data Schema | Training | Status |
|---|---|---|---|
| **v8 (current)** | `src/data/snapshot.py::Snapshot` + `LabeledSnapshot` | `src/training/per_quadrant.py` | Use for paper runs |
| **Legacy (pre-v8)** | `trajectory.Snapshot`, nested `labels.*` dicts | Different checkpoint layouts | Historical reference |

---

## Repo Layout

```
teachable-moments/
├── WebShop-master/                    # WebShop environment (modified for state save/restore)
├── src/
│   ├── data/                          # env wrapper + data schemas (Snapshot, LabeledSnapshot)
│   ├── simulation/                    # student/teacher rollout helpers
│   ├── teacher/                       # teacher API client + hint generator
│   ├── label/                         # uncertainty, leverage, CPT/ELP, quadrants, depth
│   ├── supervision/                   # formatting templates for DEMO/CONTRAST/HINT
│   ├── training/                      # SFT/LoRA trainer + per-quadrant matrix + micro-training
│   ├── features/                      # Tier1 structural + Tier2 embedding features
│   ├── predictor/                     # teachability predictor model + training
│   ├── eval/                          # evaluation utilities
│   └── utils/                         # model factory, logging, helpers
├── scripts/
│   ├── phase0/                        # rollout collection + snapshot mining
│   ├── phase1/                        # labeling (U, L, CPT, quadrants, hints)
│   ├── phase1b/                       # CPT validation via micro-training
│   ├── phase2/                        # training matrix (v8 scripts)
│   ├── phase3/                        # evaluation suite (v8 scripts)
│   ├── phase4/                        # predictor training (v8 scripts)
│   ├── analysis/                      # figure + table generation (v8 scripts)
│   └── pilots/                        # smoke test pilots
├── configs/                           # experiment configs
└── tests/                             # unit tests
```

---

## Canonical v8 Pipeline

| Step | Script | Outputs |
|---|---|---|
| 1. Collect rollouts | `scripts/phase0/run_student_rollouts_v8.py` | `rollouts.json` |
| 2. Mine failure snapshots | `scripts/phase0/mine_failure_snapshots.py` | `failure_snapshots.json` |
| 3. Label snapshots | `scripts/phase1/run_labeling.py` | `labeled_snapshots.jsonl`, `.json` |
| 4. Validate CPT | `scripts/phase1b/run_micro_training_v8.py` | `micro_training_results.json` |
| 5. Train matrix | `scripts/phase2/run_training_matrix_v8.py` | `training_summary.json` |
| 6. Evaluate suite | `scripts/phase3/run_eval_suite_v8.py` | `*.csv` files |
| 7. Train predictor | `scripts/phase4/train_predictor.py` | `predictor/*` |
| 8. Selection experiment (H7) | `scripts/phase5/run_selection_experiment_v8.py` | `selection_summary.json`, per-method CSVs |
| 9. Generate figures/tables | `scripts/analysis/generate_figures_v8.py`, `generate_tables_v8.py` | `figures/v8/*`, `tables/v8/*` |

---

## Core Components

### Simulator Modifications (WebShop State Save/Restore)

**Why:** Leverage and CPT require "time travel"—restoring an intermediate state and branching rollouts.

**File:** `WebShop-master/web_agent_site/envs/web_agent_text_env.py`
- `WebAgentTextEnv.get_state()` / `set_state(state)`
- `SimBrowser.get_state()` / `set_state(state)`
- `_serialize_session_data()` for JSON-safe serialization

**Wrapper:** `src/data/webshop_env.py`
- `WebShopEnvWrapper.reset(task_id)` → `{observation, valid_actions, task_id}`
- `WebShopEnvWrapper.step(action)` → `{observation, valid_actions, reward, done, info}`
- `WebShopEnvWrapper.get_state()` → `bytes`
- `WebShopEnvWrapper.set_state(bytes)` → `dict`

---

### Teachability Labels (src/label/)

| Component | File | Description |
|---|---|---|
| **Uncertainty U(s)** | `uncertainty.py` | Entropy/margin/top-k spread from `policy.get_action_distribution()` |
| **Leverage L(s)** | `leverage.py` | Rollout-based: `L_local = p_force − p_policy`, `L_upper = p_expert − p_policy` |
| **CPT / ELP** | `patch_gain.py` | Placebo/demo/contrast/hint patches, ELP_net, route selection |
| **Quadrants** | `quadrant.py` | Q1_highU_highL, Q2_highU_lowL, Q3_lowU_lowL, Q4_lowU_highL |
| **Depth** | `depth.py` | d_expert / d_force from leverage-at-interval snapshots |

---

### Supervision Formats (src/supervision/)

| File | Purpose |
|---|---|
| `format_router.py` | Routes snapshots to appropriate supervision format |
| `patch_templates.py` | Generates prompt–completion pairs for DEMO/CONTRAST/HINT |

Output: `(input, completion)` where input = Task + Observation + template, completion = teacher's correct action.

---

### Training (src/training/)

| File | Purpose |
|---|---|
| `per_quadrant.py` | Defines 12 quadrant×format runs + 2 baselines, builds training examples |
| `sft_trainer.py` | LoRA SFT training (wraps TRL SFTTrainer or basic loop) |
| `micro_trainer.py` | Tiny LoRA updates for CPT validation |

---

### Predictor (src/predictor/, src/features/)

| File | Purpose |
|---|---|
| `features/tier1_structural.py` | Structural features (step count, action entropy, etc.) |
| `features/tier2_embeddings.py` | Embedding-based features |
| `features/extractor.py` | Unified feature extraction |
| `predictor/model.py` | Multi-task predictor model |
| `predictor/training.py` | Training loop with ELP regression, route classification |
| `predictor/metrics.py` | NDCG@k, precision@k, Spearman for ranking |

---

### Model Interface (src/utils/)

**File:** `model_factory.py`
- `decode_action(...)` — generate action from model
- `get_action_distribution(...)` — score each valid action

---

## Key Artifacts

| Phase | Artifact | Path |
|---|---|---|
| Phase 1 | Labeled snapshots | `results/phase1/labeled_snapshots.jsonl` |
| Phase 2 | Training manifest | `results/phase2/training_summary.json` |
| Phase 3 | Eval CSVs | `results/phase3/v8/*.csv` |
| Phase 4 | Predictor checkpoint | `results/phase4/predictor/*` |
| Phase 5 (H7) | Selection summary | `results/phase5/v8/selection_summary.csv` |
| (Optional) | Recovery trajectories | `results/phase1/recovery_trajectories.jsonl` |

---

## What We Fixed / Aligned for v8

- Added missing v8 CLI wrappers:
  - `run_student_rollouts_v8.py`
  - `run_training_matrix_v8.py`
  - `run_eval_suite_v8.py`
  - `train_predictor.py`
- Added v8 figure/table scripts under `scripts/analysis/`
- Added ranking-centric predictor metrics (precision@K, NDCG@K, Spearman)
- Added HashingEmbedder for fast dependency-free embeddings (`src/features/tier2_embeddings.py`)
- Added H7 selection experiment: `scripts/phase5/run_selection_experiment_v8.py`
- Added H7 paper artifacts: `fig_h7_selection_frontier.py`, `table_h7_selection_summary.py`
- Added optional recovery trajectory workflow:
  - `scripts/phase1/export_lupper_recovery_trajectories_v8.py`
  - `scripts/phase2/train_on_recovery_trajectories_v8.py`

---

## Phase 5 (H7): Selection Experiment

**Goal:** Demonstrate that predictor-based data selection improves reliability without catastrophic forgetting.

### What it does

Compares selection methods at fixed training budget K:
- **random**: uniform sampling
- **entropy**: top-K by uncertainty U(s)
- **predictor**: top-K by predicted ELP (cheap features + trained predictor)
- **oracle**: top-K by true ELP_net (upper bound)

For each method: select K snapshots → train LoRA SFT → evaluate with `run_eval_suite_v8.py`.

### Scripts

| Script | Purpose |
|---|---|
| `scripts/phase5/run_selection_experiment_v8.py` | Run selection + training + evaluation |
| `scripts/analysis/v8/fig_h7_selection_frontier.py` | Generate reliability vs retention frontier plot |
| `scripts/analysis/v8/table_h7_selection_summary.py` | Generate summary table for paper |

### Example

```bash
python scripts/phase5/run_selection_experiment_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --predictor-path results/phase4/predictor \
  --base-model <BASE_MODEL> \
  --budget 500 \
  --output-dir results/phase5/v8
```

---

## Optional: Recovery Trajectory Workflow

**Goal:** Use expert continuations (beyond single-step hints) as an alternative training source.

This is useful when:
- Quadrant separation is weak
- Local hints alone aren't strong enough
- You want to compare trajectory-level vs step-level supervision

### Scripts

| Script | Purpose |
|---|---|
| `scripts/phase1/export_lupper_recovery_trajectories_v8.py` | Export expert continuation rollouts from high-L_upper snapshots |
| `scripts/phase2/train_on_recovery_trajectories_v8.py` | Train SFT on step-by-step recovery demos |

### Example

```bash
# Export recovery trajectories (teacher-guided expert rollouts)
python scripts/phase1/export_lupper_recovery_trajectories_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --output results/phase1/recovery_trajectories.jsonl \
  --expert-type teacher --teacher-model gpt-4.1-mini \
  --max-trajectories 200

# Train on recovery steps
python scripts/phase2/train_on_recovery_trajectories_v8.py \
  --recovery-trajectories results/phase1/recovery_trajectories.jsonl \
  --base-model <BASE_MODEL> \
  --output-dir results/phase2/recovery_sft_v8 \
  --max-examples 2000
```

---

## Legacy Components (Keep but Don't Use for v8)

Scripts are legacy if they:
- Import `src/data/trajectory.py` or expect `labels.uncertainty / labels.leverage` nested dicts
- Expect `results/phase2/models/*` + `results/phase2/baselines/*` layout

These may be useful historical references but are not wired into the v8 paper path.

---

## Quick File Reference

| Component | File |
|---|---|
| State restore | `WebShop-master/web_agent_site/envs/web_agent_text_env.py` |
| Env wrapper | `src/data/webshop_env.py` |
| Snapshot schema | `src/data/snapshot.py` |
| Uncertainty | `src/label/uncertainty.py` |
| Leverage | `src/label/leverage.py` |
| CPT/ELP | `src/label/patch_gain.py` |
| Quadrants | `src/label/quadrant.py` |
| Training matrix | `src/training/per_quadrant.py` |
| SFT trainer | `src/training/sft_trainer.py` |
| Micro-training | `src/training/micro_trainer.py` |
| Predictor | `src/predictor/training.py` + `src/features/extractor.py` |
