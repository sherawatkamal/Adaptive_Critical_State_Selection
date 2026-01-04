# ICML Execution Protocol: Teachable Moments (v8)

**Status:** ACTIVE  
**Infrastructure:** v8 Codebase (Locked)  
**Objective:** ICML Submission Deadline  
**Owners:** @Kamal (Lead/Modeling), @Chris (Validation/Infra)

---

## 1. Infrastructure Standardization

To guarantee **deterministic resettability** and **CPT logging granularity**, we are standardizing all experiments on the `v8` codebase effective immediately.

- **Focus:** Your primary responsibility is **Execution, Tuning, and Analysis**, not infrastructure engineering.

---

## 2. Pre-Flight Checklist

Before launching production runs, verify the following state.

### A. Context Loading (Required)

| Priority | Document | Purpose |
|----------|----------|---------|
| ⭐ 1 | `experimental_plan.md` | **The Execution Bible**: Precise commands, flags, and pass/fail criteria for every phase. Read this to know *exactly* what to run. |
| ⭐ 2 | `teachable_moments_research_blueprint_v8.md` | **The "Why"**: Scientific logic for Quadrants, CPT, and Supervision. Read this to understand the research hypotheses. |
| 3 | `02_repo_overview_what_we_built_v8.md` | **Code Map**: Explains where code lives (Training, Labeling, Predictor) and how components connect. Read this to navigate the codebase. |
| 4 | `results_by_hypothesis.md` | **The Goal**: Detailed outline of every figure and table we need for the paper. Read this to know what "done" looks like. |
| 5 | `01_experiments_audit_execution_plan.md` | **The Context**: Historical audit of what was fixed/added for v8. Read this if you're confused about legacy vs new methods. |
| 6 | `README.md` | **The Onramp**: High-level conceptual overview and installation guide. Read this first if you are new to the project. |

### B. Environment & Sanity Check

```bash
# 1. Setup (Clean env recommended)
conda create -n teachable python=3.11 -y && conda activate teachable
pip install -e ".[full]"

# 2. Config
cp .env.example .env  # Add OPENAI_API_KEY

# 3. Download Student Model (~16GB)
python scripts/download_models.py --model Qwen/Qwen3-8B

# 4. Smoke Test (Verifies E2E pipeline without API cost)
python scripts/pilots/run_all_pilots_v8.py \
  --out-dir results/pilots/smoke_test \
  --base-model sshleifer/tiny-gpt2 \
  --n-tasks 2 --mock-env --skip-api
```

**Gatekeeper:** If `results/pilots/smoke_test/` contains valid JSON outputs for all phases, proceed.

---

## 3. Execution Phase 1A: Quadrant-Based Training

**Owner:** @Kamal  
**Objective:** Establish the baseline performance matrix across teachability quadrants.

### Protocol

**Step 1: Generate Data (Student Rollouts)**
- Target: ~3,000 tasks (failures)
- Script: `scripts/phase0/run_student_rollouts_v8.py`

```bash
python scripts/phase0/run_student_rollouts_v8.py \
  --model-path models/Qwen3-8B \
  --n-tasks 3000 --max-steps 15 \
  --output results/phase0/rollouts.json
```

**Step 2: Mine & Label**
- Scripts: `mine_failure_snapshots.py` → `run_labeling.py`
- ⚠️ Constraint: This step consumes OpenAI API budget.

```bash
python scripts/phase0/mine_failure_snapshots.py \
  --rollouts results/phase0/rollouts.json \
  --output results/phase0/failure_snapshots.json \
  --max-snapshots 3000

python scripts/phase1/run_labeling.py \
  --snapshots results/phase0/failure_snapshots.json \
  --output-dir results/phase1 \
  --student-checkpoint models/Qwen3-8B \
  --teacher-model gpt-4o-mini \
  --assign-quadrants
```

**Step 3: Train Matrix**
- Script: `scripts/phase2/run_training_matrix_v8.py`
- Output: A 14-run matrix (Quadrants × Supervision Types)

```bash
python scripts/phase2/run_training_matrix_v8.py \
  --labeled-snapshots results/phase1/labeled_snapshots.jsonl \
  --base-model models/Qwen3-8B \
  --output-dir results/phase2
```

### Deliverable
`results/phase2/training_summary.json`

### Success Criteria
| Metric | Threshold |
|--------|-----------|
| Data Volume | >100 snapshots per quadrant |
| Signal | Q1 (High U, High L) shows distinctly higher improvement than Q3 |
| Transfer | Diagonal dominates in transfer matrix |

---

## 4. Execution Phase 1B: CPT Validation

**Owner:** @Chris  
**Objective:** Validate that CPT (in-context) scores correlate with training gains (LoRA).

### Protocol

> **Parallel Execution:** Do not wait for Phase 1A. Generate a local "Pilot Set" (N=50) to validate the pipeline immediately.

**Step 1: Pilot Generation**

```bash
python scripts/phase0/run_student_rollouts_v8.py \
  --model-path models/Qwen3-8B \
  --n-tasks 50 --max-steps 15 \
  --output results/phase0/rollouts_pilot.json

python scripts/phase0/mine_failure_snapshots.py \
  --rollouts results/phase0/rollouts_pilot.json \
  --output results/phase0/failure_snapshots_pilot.json

python scripts/phase1/run_labeling.py \
  --snapshots results/phase0/failure_snapshots_pilot.json \
  --output-dir results/phase1_pilot \
  --student-checkpoint models/Qwen3-8B \
  --teacher-model gpt-4o-mini \
  --assign-quadrants
```

**Step 2: Micro-Training Validation**

```bash
python scripts/phase1b/run_micro_training_v8.py \
  --labeled results/phase1_pilot/labeled_snapshots.json \
  --base-model models/Qwen3-8B \
  --output-dir results/phase1b \
  --n-per-quadrant 50 \
  --n-steps 2 \
  --learning-rate 1e-3 \
  --n-validation-rollouts 3
```

### Deliverable
`results/phase1b/cpt_correlation.json`

### Success Criteria
| Metric | Target | Stretch |
|--------|--------|---------|
| Spearman ρ | ≥ 0.3 | ≥ 0.5 |
| Differentiation | Q1 utility > Q2 utility on average | — |

---

## 5. Troubleshooting & "Gotchas"

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| Quadrant Imbalance | Dataset skew | If one quadrant < 50 samples, update thresholds or increase N rollouts |
| Teacher Invalid Action | Hallucination | Check `valid_actions` filter in teacher wrapper |
| OOM (Training) | VRAM constraint | Use `--batch-size 8 --gradient-accumulation-steps 4` |
| CPT Noise | Variance | Increase `n_per_condition` from 2 → 3 for validation runs only |
| Low correlation | Noisy labels | Verify labeling completed successfully; check for NaN values |

---

## 6. Available Infrastructure (Ready-to-Use)

> **All scripts are implemented and tested.** You are executing a pre-built pipeline, not building one.

### Pipeline Scripts by Phase

| Phase | Script | Purpose |
|-------|--------|---------|
| **0** | `scripts/phase0/run_student_rollouts_v8.py` | Generate student failure trajectories |
| **0** | `scripts/phase0/mine_failure_snapshots.py` | Extract failure snapshots from rollouts |
| **0** | `scripts/phase0/mine_retention_tasks.py` | Mine global retention panel |
| **0** | `scripts/phase0/collect_expert_trajectories.py` | (Alt) Expert-failure pipeline |
| **0** | `scripts/phase0/mine_snapshots_from_expert_trajectories.py` | (Alt) Mine from expert trajectories |
| **1** | `scripts/phase1/run_labeling.py` | Compute U, L, CPT, assign quadrants |
| **1** | `scripts/phase1/compute_uncertainty.py` | Uncertainty metrics only |
| **1** | `scripts/phase1/compute_leverage.py` | Leverage metrics only |
| **1** | `scripts/phase1/run_cpt.py` | CPT scoring only |
| **1** | `scripts/phase1/assign_quadrants.py` | Manual threshold assignment |
| **1** | `scripts/phase1/create_validation_panel.py` | Create stratified evaluation panel |
| **1** | `scripts/phase1/export_lupper_recovery_trajectories_v8.py` | Export high-leverage recovery trajectories |
| **1b** | `scripts/phase1b/run_micro_training_v8.py` | CPT validation via micro-training |
| **2** | `scripts/phase2/run_training_matrix_v8.py` | Train 14-model matrix |
| **2** | `scripts/phase2/train_on_recovery_trajectories_v8.py` | Train on recovery trajectories |
| **3** | `scripts/phase3/run_eval_suite_v8.py` | Full evaluation suite |
| **4** | `scripts/phase4/train_predictor_v8.py` | Teachability predictor |
| **4** | `scripts/phase4/evaluate_predictor.py` | Comprehensive predictor evaluation |
| **5** | `scripts/phase5/run_selection_experiment_v8.py` | H7 selection comparison |
| **Pilot** | `scripts/pilots/run_all_pilots_v8.py` | End-to-end smoke test |

### Baseline Scripts (Reference)

| Script | Purpose |
|--------|---------|
| `scripts/baselines/run_random_selection.py` | Baseline: Random data selection |
| `scripts/baselines/run_entropy_selection.py` | Baseline: Uncertainty-based selection |
| `scripts/baselines/run_eef_comparison.py` | Baseline: EEF (Expected Error Reduction) |

### Analysis Scripts → Paper Artifacts

These scripts **auto-generate figures and tables** for the paper. See `results_by_hypothesis.md` for expected outputs.

#### Hypothesis H1: Teachability Heterogeneity
| Script | Output |
|--------|--------|
| `fig_h1_teachability_landscape.py` | **Fig H1.1** — U × L scatter colored by ELP |
| `table_h1_quadrant_summary.py` | **Tab H1.1** — Quadrant characteristics |
| `phase1_figure_quadrant_distribution.py` | Quadrant balance visualization |
| `phase1_table_quadrant_characteristics.py` | Detailed quadrant stats |

#### Hypothesis H2: CPT Validation
| Script | Output |
|--------|--------|
| `phase1b_figure_cpt_validation.py` | **Fig H2.1** — CPT vs micro-training scatter |
| `phase1b_figure_cpt_correlation.py` | **Fig H2.2** — CPT validity by quadrant |

#### Hypothesis H3: Supervision × Quadrant
| Script | Output |
|--------|--------|
| `fig_h3_in_quadrant_heatmap.py` | **Fig H3.1** — 4×3 heatmap |
| `table_h3_best_supervision.py` | **Tab H3.1** — Best format per quadrant |
| `phase2_figure_supervision_comparison.py` | Supervision comparison bars |
| `phase2_table_supervision_effectiveness.py` | Effectiveness metrics |

#### Hypothesis H4: Cross-Quadrant Transfer
| Script | Output |
|--------|--------|
| `fig_h4_transfer_matrix.py` | **Fig H4.1** — Transfer matrix heatmap |
| `phase3_figure_transfer_matrix.py` | Alternative transfer visualization |
| `phase3_table_transfer_matrix.py` | **Tab H4.1** — Numeric transfer matrix |

#### Hypothesis H5: Reliability & Retention
| Script | Output |
|--------|--------|
| `fig_h5_frontier_reliability_vs_retention.py` | **Fig H5.3** — Frontier plot |
| `table_h5_main_results.py` | **Tab H5.1** — Main results table |
| `phase3_figure_retention_curves.py` | Retention over checkpoints |

#### Hypothesis H6: Teachability Predictor
| Script | Output |
|--------|--------|
| `phase4_figure_predictor_performance.py` | **Fig H6.1** — NDCG@K curves |
| `phase4_table_predictor_performance.py` | **Tab H6.1** — Predictor metrics |
| `table_h6_predictor_metrics.py` | Detailed predictor stats |

#### Hypothesis H7: Selection Experiment
| Script | Output |
|--------|--------|
| `fig_h7_selection_frontier.py` | **Fig H7.1** — Selection method comparison |
| `table_h7_selection_summary.py` | **Tab H7.1** — Selection summary |

#### Summary Figures (Paper Main)
| Script | Output |
|--------|--------|
| `summary_figure_main_results.py` | Combined main results figure |
| `summary_table_main_results.py` | Paper main results table |
| `plot_quadrant_thresholds.py` | Threshold selection helper |

### Advanced / Experimental Capabilities

| Script | Purpose |
|--------|---------|
| `scripts/phase3/evaluate_mechanistic.py` | **Mechanistic Eval**: Deep dive on state/goal tracking |
| `scripts/phase3/evaluate_drift.py` | **Drift Eval**: Analyze feature distribution shift |
| `scripts/phase3/evaluate_end2end.py` | **End-to-End**: Full pipeline verification |

---

## 7. Artifact Generation (One Command)

Once experiments complete:

```bash
# Generate ALL paper figures
python scripts/analysis/generate_figures_v8.py --results-dir results --output-dir figures/v8

# Generate ALL paper tables  
python scripts/analysis/generate_tables_v8.py --results-dir results --output-dir tables/v8
```

**Outputs:** `figures/v8/` and `tables/v8/` — ready for LaTeX inclusion.

---

## 8. Minimum Viable Paper (If Deadline Critical)

Per `results_by_hypothesis.md`, if compute/time is tight, prioritize:

| Hypothesis | Must-Have Artifacts |
|------------|---------------------|
| H1 | Fig H1.1 + Tab H1.1 |
| H2 | Fig H2.1 + Tab H2.1 |
| H3/H4/H5 | Fig H3.1 + Fig H4.1 + Fig H5.3 + Tab H5.1 |
| H6/H7 | Fig H6.1 + Fig H7.1 |
