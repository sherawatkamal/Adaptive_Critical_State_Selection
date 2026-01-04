# Teachable Moments: Experiments Inventory + Code Audit + Phase-by-Phase Execution Plan (v8)

This document does two things:

1) **From the Research Blueprint v8**: enumerates **all experiments, methods (ours + baselines), analyses, and required figures/tables**.  
2) **From the codebase** (`teachable-moments-source/`): audits what is already implemented vs. missing/incorrect, and gives a **practical, step-by-step execution plan** to get every experiment to run and produce publishable results.

> **Core deliverables (v8)** are explicitly:  
> (i) teachability model + instruments (U×L + probes),  
> (ii) **per-quadrant training** (14 runs), and  
> (iii) **CPT validation via micro-training**.【turn4file11†teachable_moments_research_blueprint_v8.md†L20-L24】

---

## 1) What the Research Plan Requires (Inventory)

### 1.1 The core “teachability model” objects to compute (labels/instruments)

These are the **state-level labels** that define the teachability model and are used for analysis + selecting training data.

#### A) Uncertainty axis U(s)
- **Definition:** scalar uncertainty computed from the student’s action distribution (entropy / margin / top‑k spread).  
- **Usage:** quadrant assignment, selection baselines (entropy), and as a predictor target/features.

#### B) Actionability axis L_local(s) and upper bound L_upper(s)
- **Definition:** leverage decomposes into:
  - **L_local(s)** = improvement from forcing a single expert action at s, then letting student continue.  
  - **L_upper(s)** = improvement if a strong expert takes full control from s.  
- **Usage:** quadrant assignment uses **L_local**, depth analysis uses both, predictor targets include both.

#### C) Recovery depth d(s) along trajectories
- Derived **offline** using leverage computed at fixed intervals along a failure trajectory (no additional rollouts).【turn4file6†teachable_moments_research_blueprint_v8.md†L34-L49】【turn4file5†teachable_moments_research_blueprint_v8.md†L9-L18】  
- Two measures:
  - d_expert: “how far back until expert can recover”  
  - d_force: “how far back until a single forced action makes student recover”【turn4file4†teachable_moments_research_blueprint_v8.md†L55-L66】

#### D) CPT / ELP teachability score and route label
- CPT protocol (single stage) computes: p_base, p_placebo, p_demo, p_contrast, p_hint with **n=2 episodes per condition** (10 total).【turn4file7†teachable_moments_research_blueprint_v8.md†L1-L20】  
- Labels:
  - **ELP_net(s)** = max over {demo, contrast, hint} of (p_m − p_placebo)  
  - **ELP_raw(s)** = max over {demo, contrast, hint} of (p_m − p_base)  
  - **route(s)** = argmax_m (p_m − p_placebo)【turn4file7†teachable_moments_research_blueprint_v8.md†L12-L16】

> **Interpretation:** ELP_net is the “expected learning payoff” proxy (net of generic prompt effects). Route is the best *intervention type* for that state.

---

### 1.2 The quadrant framework (primary organizing principle)

Quadrants are computed from **U(s)** and **L_local(s)** with median thresholds.【turn4file5†teachable_moments_research_blueprint_v8.md†L20-L53】

| Quadrant | U | L_local | Interpretation |
|---|---:|---:|---|
| **Q1** | High | High | Uncertain and fixable |
| **Q2** | High | Low | Uncertain and stuck |
| **Q3** | Low | Low | Confident and stuck |
| **Q4** | Low | High | Confident but wrong |

**Hypothesized best supervision by quadrant:** Hint best in Q1, Contrast best in Q4, Q3 low ELP, Q2 may be hard overall.【turn4file10†teachable_moments_research_blueprint_v8.md†L1-L10】

---

### 1.3 Supervision “methods” to train with (ours)

Per the blueprint, you train on **the same underlying snapshot set**, but vary the *format* of supervision:

1) **DEMO**: a direct teacher demonstration / correct action.  
2) **CONTRAST**: contrasts a wrong action vs correct action (unlearning confident mistake).  
3) **HINT**: diagnosis/hint without giving reasoning chain; output is still the correct action.

(These align with CPT patch types and the “route” concept.)

---

## 2) Experiments and Baselines to Run

### 2.1 E1 Main Experiment: Per‑Quadrant Training (14 runs)

**Matrix:** 4 quadrants × 3 supervision types = 12 runs, plus 2 baselines.【turn4file3†teachable_moments_research_blueprint_v8.md†L31-L45】

| Quadrant | Demo | Contrast | Hint |
|---|---|---|---|
| Q1 (high U, high L) | Q1_demo | Q1_contrast | Q1_hint |
| Q2 (high U, low L) | Q2_demo | Q2_contrast | Q2_hint |
| Q3 (low U, low L) | Q3_demo | Q3_contrast | Q3_hint |
| Q4 (low U, high L) | Q4_demo | Q4_contrast | Q4_hint |

**Baselines (2 runs):**
- **B1_uniform**: uniform random snapshots across all quadrants, demo format  
- **B2_all**: all quadrants combined, demo format【turn4file3†teachable_moments_research_blueprint_v8.md†L42-L45】

**Training configuration** (target values): LoRA rank 16–32, epochs 3, batch size 8, lr 2e‑5.【turn4file3†teachable_moments_research_blueprint_v8.md†L46-L56】

#### E1 evaluation protocol (per model)
Blueprint requires: overall success, per‑quadrant success, stuckness, retention.【turn4file3†teachable_moments_research_blueprint_v8.md†L62-L68】

1) **Overall success rate**: 500 held‑out tasks, 3 rollouts each  
2) **Per‑quadrant success**: held‑out snapshots from each quadrant, 2 rollouts each  
3) **Stuckness**: repeat_action_rate, loop_detected, wasted_steps  
4) **Retention**: 100 tasks base succeeds on, 2 rollouts each

#### Core hypotheses tested by E1
H1–H4 (heterogeneity, format×quadrant, limited transfer, beats uniform).【turn4file14†teachable_moments_research_blueprint_v8.md†L18-L35】

---

### 2.2 E2 Main Validation: CPT validation via micro‑training (Phase 1B)

Goal: CPT should predict actual “learnability” from a single example.

Protocol (required):
- sample **N=200 snapshots** stratified by quadrant (50 each)【turn4file4†teachable_moments_research_blueprint_v8.md†L9-L18】  
- compute CPT labels (ELP_net + route)  
- do **1–2 LoRA steps** on the single route‑matched example  
- evaluate before/after on:
  - failure trajectory from snapshot state  
  - 10 similar tasks  
- record Δ_micro = success_after − success_before【turn4file4†teachable_moments_research_blueprint_v8.md†L15-L19】

Pass criteria:
- ρ(ELP_net, Δ_micro) > 0.3 overall【turn4file4†teachable_moments_research_blueprint_v8.md†L20-L23】

---

### 2.3 E3 Analysis: Leverage & depth (post‑hoc, but important)

Uses existing labels to explain *why* and *when* patching works.
- Stratify improvements by leverage profile (single‑action bottleneck vs. needs trajectory vs. dead‑end) and by depth bins.【turn4file9†teachable_moments_research_blueprint_v8.md†L38-L47】  
- Key outcome: show “teachable moments” are those with **small d_force** and/or large L_local, while some failures are beyond local repair.

---

### 2.4 E4 Teachability predictor (scaling)
Goal: predict teachability cheaply at scale (Tier1+2 features only).
- Targets: predict ELP_hat, route, L_local, L_upper, depth bins, quadrant.  
- Metrics: NDCG@k, Precision@k, Spearman for ELP; Macro‑F1 + confusion matrix for route, etc.【turn4file1†teachable_moments_research_blueprint_v8.md†L61-L72】【turn4file12†teachable_moments_research_blueprint_v8.md†L23-L33】

---

### 2.5 Follow‑ups retained (post‑sprint/time‑permitting)

From v8 follow‑ups section (8B):
- **E5**: selection comparison (Uniform vs Entropy vs ELP_hat), fixed K=500 snapshots, route‑matched SFT【turn4file9†teachable_moments_research_blueprint_v8.md†L21-L29】  
- **E8**: teachability drift panel across checkpoints (200 snapshots, rerun CPT per checkpoint)【turn4file9†teachable_moments_research_blueprint_v8.md†L49-L60】  
- **E10**: DPO vs SFT for contrast‑route subset if contrast matters【turn4file9†teachable_moments_research_blueprint_v8.md†L69-L78】

---

## 3) Required Analyses + Figures/Tables (Paper Backbone)

### 3.1 Main paper figures (must generate)

From the blueprint’s “Figures and Tables Specifications” section.【turn4file15†teachable_moments_research_blueprint_v8.md†L12-L25】

**Figure 1: Teachability Landscape (Hero figure)**  
Source: `labeled_snapshots.parquet`
- Panel A: scatter of U vs L_local (color=ELP_net; show median thresholds)  
- Panel B: violin of ELP_net by quadrant (Q1/Q4 higher expected)【turn4file2†teachable_moments_research_blueprint_v8.md†L1-L7】  
- Panel C: route distribution by quadrant (stacked bars)【turn4file2†teachable_moments_research_blueprint_v8.md†L8-L14】

**Figure 2: CPT validation via micro‑training**  
Source: `cpt_validation_results.parquet`
- Panel A: scatter ELP_net vs Δ_micro with regression and correlation annotation【turn4file2†teachable_moments_research_blueprint_v8.md†L28-L35】  
- Panel B: correlation bars by quadrant with ρ=0.3 line【turn4file2†teachable_moments_research_blueprint_v8.md†L36-L41】

**Figure 3: Per‑quadrant training results (Main result)**  
Source: `per_quadrant_results.csv`
- Panel A: 4×3 heatmap quadrant×supervision success rates【turn4file2†teachable_moments_research_blueprint_v8.md†L55-L62】  
- Panel B: best supervision per quadrant vs demo baseline【turn4file2†teachable_moments_research_blueprint_v8.md†L63-L70】  
- Panel C: best per‑quadrant vs baselines overall success【turn4file2†teachable_moments_research_blueprint_v8.md†L71-L76】

**Figure 4: Cross‑quadrant transfer matrix**  
Source: `transfer_matrix.csv` (expected diagonal highest).【turn4file0†teachable_moments_research_blueprint_v8.md†L35-L38】

**Figure 5: Retention frontier**  
Plot failure improvement vs retention degradation (method points).【turn4file0†teachable_moments_research_blueprint_v8.md†L1-L4】

**Figure 6: Stuckness metrics**  
Repeat action rate, loop detect, wasted steps (before/after, by model).

### 3.2 Required tables

**Table 1**: Transfer matrix values (or per‑quadrant success)  
**Table 2**: best supervision per quadrant + improvement + hypothesis match (template appears in blueprint).【turn4file14†teachable_moments_research_blueprint_v8.md†L5-L13】  
**Table 3**: predictor performance (NDCG@k, Macro‑F1, etc).

---

## 4) Codebase Audit: What’s Implemented vs Missing

Repo root examined: `/mnt/data/teachable-moments-source/`

### 4.1 Strong/implemented components (good foundation)

#### Simulator state save/restore
- ✅ `WebShop-master/web_agent_site/envs/web_agent_text_env.py` implements `get_state()` / `set_state()` for `WebAgentTextEnv` and `SimBrowser`.
- ✅ Wrapper `src/data/webshop_env.py` exposes `get_state()` and `set_state()` returning/accepting **bytes** (pickled dict) and provides `observation + valid_actions`.

This is the key capability required for leverage + CPT (counterfactual rollouts).

#### Teachability instruments (core algorithms)
- ✅ **Leverage estimation**: `src/label/leverage.py` (`estimate_leverage`, `run_rollouts`, `run_forced_rollouts`)  
- ✅ **Uncertainty**: `src/label/uncertainty.py` (entropy / margin / etc from action distribution)  
- ✅ **CPT / ELP**: `src/label/patch_gain.py` (`run_cpt`)  
- ✅ **Depth**: `src/label/depth.py`  
- ✅ **Quadrant assignment**: `src/label/quadrant.py` (median thresholds, correct Q2/Q4 mapping)

#### Supervision format generation + training
- ✅ Supervision router/templates: `src/supervision/format_router.py`, `src/supervision/patch_templates.py`  
- ✅ Per‑quadrant training matrix (12 + 2 baselines): `src/training/per_quadrant.py`  
- ✅ SFT + LoRA trainer: `src/training/sft_trainer.py`

#### Predictor (scaling)
- ✅ Tier1+2 features: `src/features/tier1_structural.py`, `src/features/tier2_embeddings.py`, `src/features/extractor.py`  
- ✅ Predictor model + training: `src/predictor/model.py`, `src/predictor/training.py`, `src/predictor/metrics.py`

### 4.2 Critical gaps that block end‑to‑end experiments

These must be addressed for “real results” (not mock outputs):

#### Gap A: Snapshot creation pipeline is missing / inconsistent
- `src/data/snapshot.py` defines the *right* schema (Snapshot/LabeledSnapshot) with base64 env state, but:
  - student/teacher rollout code (`src/simulation/student_rollout.py`, `src/simulation/teacher_rollout.py`) **does not record env_state per step** and cannot produce restorable snapshots.
  - many Phase 1 scripts in `scripts/phase1/` assume an **old snapshot schema** (`trajectory_id`, `step`, `state`) and will not run.

**Action:** implement a canonical snapshot extraction script that emits `Snapshot.to_dict()` format and stores env_state as base64 (`env_state_b64`).

#### Gap B: Teacher hint generation is not wired correctly
- `src/teacher/structured_hint.py` expects `teacher_client.generate_text()`, but `src/teacher/client.py` only implements `generate()`.
- There is also an older `src/teacher/hint_generator.py` that expects different snapshot fields (`task`, `state`).

**Action:** unify on **structured_hint** and add a `generate_text()` alias or update calls to `generate()`.

#### Gap C: Micro‑training validation is implemented but currently evaluates the wrong model
- `src/training/micro_trainer.py` trains a `peft_model`, but evaluation rollouts call `model_factory.decode_action()` **without passing the tuned model**, so it reuses the base model. (So Δ_micro ≈ 0 regardless.)

**Action:** pass the tuned model/tokenizer into decoding, or update `ModelFactory` to point to the tuned model during evaluation.

#### Gap D: Evaluation harness is missing for WebShopEnvWrapper + LoRA adapters
- Most `src/eval/*.py` uses an old env interface (reset→str; step→(obs,reward,done,info)). `WebShopEnvWrapper` returns dicts.
- `ModelFactory` cannot load a LoRA adapter saved by training (it loads only `AutoModelForCausalLM` base).

**Action:** implement:
1) a thin **evaluation runner** that uses the existing wrapper interface (dict obs + valid_actions), and  
2) LoRA adapter loading (PeftModel.from_pretrained) in `ModelFactory` or an `InferenceFactory`.

#### Gap E: Analysis / plotting scripts are placeholders or incorrect
- `scripts/analysis/generate_figures.py` fabricates random outputs if files missing and has quadrant label mismatch (Q2/Q4 swapped).  
- `scripts/analysis/generate_tables.py` references unrelated benchmarks.

**Action:** replace placeholders with real figure builders reading the output artifacts you will actually produce.

### 4.3 Non‑blocking but important quality issues
- B1_uniform size currently hardcoded to 2000 in `src/training/per_quadrant.py`; should match per‑quadrant run budget (~400–600) to be a fair baseline.【turn4file3†teachable_moments_research_blueprint_v8.md†L48-L50】  
- SFT training currently uses full‑sequence loss; consider masking prompt tokens (completion-only loss) for cleaner action learning.

---

## 5) Concrete Execution Plan (What to do in Each Phase)

This is the practical checklist to produce all results & figures.

### Phase 0: Data collection & snapshot mining

#### P0.0 Model Download
**Goal:** Ensure local availability of teacher/student models.
- **Model:** Qwen/Qwen3-8B (Instruct version)
- **Script:** `scripts/download_models.py`

```bash
python scripts/download_models.py --model Qwen/Qwen3-8B
```

#### P0.1 Verify state save/restore correctness (must be 100% reliable)
**Goal:** prove that `env.get_state()` → `env.set_state()` restores the exact same observable page and valid actions.

**How:**
- Create a unit/integration test that:
  1) reset task → take 2 steps  
  2) call `get_state()`  
  3) take 1 more step  
  4) `set_state()` to earlier state  
  5) compare observation string + valid_actions list with the saved ones

**Output:** `results/phase0/state_restore_check.json` (pass/fail + sample diffs).

#### P0.2 Collect failures (choose Setup A or B)
Blueprint recommends prioritizing **Setup A** (expert failures) for core experiments.【turn4file3†teachable_moments_research_blueprint_v8.md†L22-L25】

**Setup A (recommended to unblock quickly):**
- Use `scripts/phase0/collect_expert_trajectories.py`, but FIX serialization (bytes→base64).
- Partition into successes/failures.

**Setup B (secondary):**
- Use `scripts/phase0/collect_student_failures.py`, but UPDATE `StudentRollout` to record per-step env state (see Gap A).

**Output artifacts:**
- `results/phase0/expert_trajectories.json` (or jsonl)  
- `results/phase0/student_failures.json`

#### P0.3 Mine candidate snapshots from failed trajectories
**Goal:** produce ~2000 snapshots. Each snapshot should be **2–5 steps before failure** (blueprint)【turn4file7†teachable_moments_research_blueprint_v8.md†L3-L4】 and include:
- `task_id`, `step_idx`
- `observation` (string)
- `valid_actions` (list[str])
- `last_action` (student/expert action taken at that step, for contrast)
- `env_state_b64` (base64 of wrapper state bytes)
- plus optional: recent history window (last N obs/actions) for Tier1 features

**Output:** `results/phase0/snapshots.jsonl` and/or `snapshots.parquet`.

**QC checks:**
- 0% missing env_state_b64  
- valid_actions non-empty  
- restore works for random 50 snapshots  
- action is in valid_actions (or mapped)

---

### Phase 1A: Labeling (U, leverage, CPT, quadrant)

This phase produces the **single most important dataset**:
`labeled_snapshots.parquet` (the paper’s backbone).【turn4file15†teachable_moments_research_blueprint_v8.md†L24-L25】

#### P1.1 Teacher hint generation (structured)
**Goal:** attach to each snapshot:
- suggested_action (must be valid)
- rationale (short)
- diagnosis (short)
- error_type ∈ {affordance_miss, attribute_confusion, planning_error, exploration_failure}【turn4file10†teachable_moments_research_blueprint_v8.md†L55-L60】

**Implementation location:** `src/teacher/structured_hint.py` (+ fix client method).

**Output:** `results/phase1/teacher_hints.jsonl` and merged into `labeled_snapshots.parquet`.

**QC:** suggested_action must be in valid_actions (or fuzzy-matched).

#### P1.2 Compute uncertainty U(s)
**Implementation location:** `src/label/uncertainty.py` (use entropy as primary).

**Output columns:**
- entropy, margin, top_k_mass, effective_actions, action_space_size

#### P1.3 Compute leverage (L_local, L_upper + components)
**Implementation location:** `src/label/leverage.py`

**Outputs per snapshot:**
- p_policy, p_force, p_expert
- L_local = p_force − p_policy
- L_upper = p_expert − p_policy
- leverage_gap = L_upper − L_local

**QC sanity:**
- p_expert ≥ p_force ≥ p_policy for most snapshots (not always, but should trend).

#### P1.4 Run CPT (ELP + route)
**Implementation location:** `src/label/patch_gain.py`  
CPT protocol: 10 episodes per snapshot (2×5).【turn4file7†teachable_moments_research_blueprint_v8.md†L17-L20】

**Outputs per snapshot:**
- p_base, p_placebo, p_demo, p_contrast, p_hint
- ELP_raw, ELP_net
- route_net

**QC:** placebo should not outperform demo/contrast/hint systematically.

#### P1.5 Quadrant assignment
**Implementation location:** `src/label/quadrant.py`  
Use median thresholds for U and L_local.【turn4file5†teachable_moments_research_blueprint_v8.md†L33-L36】

**Output:** `quadrant` label appended.

#### P1.6 Depth analysis (optional now, required for E3)
If leverage computed at fixed intervals along trajectories, compute:
- d_expert, d_force; plus bins (0,1,2,3+).【turn4file4†teachable_moments_research_blueprint_v8.md†L55-L66】

**Output:** columns in `labeled_snapshots.parquet`.

---

### Phase 1B: CPT validation (micro‑training)

**Goal:** show ELP_net predicts Δ_micro.  
Protocol required by blueprint.【turn4file4†teachable_moments_research_blueprint_v8.md†L9-L23】

#### P1B.1 Sample the validation panel
- stratified 50 per quadrant (N=200).【turn4file4†teachable_moments_research_blueprint_v8.md†L9-L12】

#### P1B.2 Run micro‑training per snapshot
- build one SFT example in route‑matched format  
- do 1–2 LoRA steps  
- eval before/after on (snapshot rollout + similar tasks)

**Implementation location:** `src/training/micro_trainer.py` (needs fix to evaluate tuned model).

**Output:** `results/phase1b/cpt_validation_results.parquet`.

#### P1B.3 Correlation analysis + Figure 2
- compute Pearson/Spearman overall and per quadrant  
- require overall ρ > 0.3 (pass)【turn4file4†teachable_moments_research_blueprint_v8.md†L20-L23】

---

### Phase 2: Per‑quadrant training (14 runs)

**Goal:** run the full E1 matrix + baselines.【turn4file3†teachable_moments_research_blueprint_v8.md†L31-L45】

#### P2.1 Build training datasets
For each run:
- choose snapshot subset
- format supervision: demo/contrast/hint
- output teacher action only

**Implementation:** `src/training/per_quadrant.py` + `src/supervision/format_router.py`

**Fix needed:** make B1_uniform sample size equal to a typical quadrant size (~400–600).【turn4file3†teachable_moments_research_blueprint_v8.md†L48-L50】

#### P2.2 Train models (LoRA SFT)
**Implementation:** `src/training/sft_trainer.py`

**Outputs:**
- `results/phase2/models/{run_id}/` with adapter weights + config + logs

**QC:**
- ensure each run trains on correct number of examples  
- log training_config.json per run

---

### Phase 3: Evaluation (all metrics + CSVs for figures)

This phase is currently the biggest missing implementation in code.

#### P3.1 Implement evaluation harness (must)
It must support:
- end‑to‑end from reset(task_id)  
- snapshot evaluation from env_state_b64 restore  
- multiple rollouts (stochastic if desired)  
- LoRA adapter loading for each trained model

**Output files (minimum):**
- `results/phase3/overall_results.csv` (model × success_mean ± se)  
- `results/phase3/per_quadrant_results.csv` (model × quadrant success)  
- `results/phase3/transfer_matrix.csv` (4×4)  
- `results/phase3/retention.csv`  
- `results/phase3/stuckness.csv`

#### P3.2 Compute transfer matrix
Diagonal should exceed off‑diagonal; expected patterns described in blueprint.【turn4file0†teachable_moments_research_blueprint_v8.md†L35-L38】

#### P3.3 Generate Figures 3–6 + Tables
Use outputs above to generate the paper figures.

---

### Phase 4: Teachability predictor (E4) + selection follow‑up (E5)

#### P4.1 Build training dataframe
From `labeled_snapshots.parquet`, extract Tier1 features and optional Tier2 embeddings.

#### P4.2 Train multi‑task predictor
**Implementation:** `src/predictor/training.py` + `src/features/extractor.py`

**Report:** NDCG@k, Precision@k, Spearman; Macro‑F1 for route.【turn4file1†teachable_moments_research_blueprint_v8.md†L61-L72】

#### P4.3 Follow‑up E5 (if time)
Compare selection methods with K=500 snapshots.【turn4file9†teachable_moments_research_blueprint_v8.md†L25-L29】

---

## 6) “What results are we aiming for?” (Strong claims mapping)

This is the checklist for what you need to show to claim a strong paper:

### Claim A: Teachability heterogeneity exists and is structured
**Evidence:**
- Figure 1A scatter shows meaningful structure in U×L, with quadrant thresholds.  
- Figure 1B ELP distributions differ by quadrant (ANOVA significant).【turn4file15†teachable_moments_research_blueprint_v8.md†L1-L6】

### Claim B: The optimal supervision format depends on the teachability regime
**Evidence:**
- Figure 3 heatmap shows best format differs across quadrants (e.g., Q4→Contrast, Q1→Hint).【turn4file10†teachable_moments_research_blueprint_v8.md†L1-L9】  
- Table 2 summarizes best format per quadrant and hypothesis match.【turn4file14†teachable_moments_research_blueprint_v8.md†L5-L13】

### Claim C: CPT is a valid proxy for learning payoff
**Evidence:**
- Figure 2 shows correlation ELP_net vs Δ_micro > 0.3 overall.【turn4file4†teachable_moments_research_blueprint_v8.md†L20-L23】  
- Placebo sanity checks pass.

### Claim D: Training on teachable moments improves reliability without catastrophic forgetting
**Evidence:**
- Figure 5 retention frontier: high failure improvement with low retention drop.  
- Retention eval (100 tasks) shows minimal degradation.

### Claim E: We can identify teachable moments cheaply at scale
**Evidence:**
- Predictor beats entropy-only baseline on NDCG@k / Precision@k and route Macro-F1.【turn4file12†teachable_moments_research_blueprint_v8.md†L23-L33】  
- Optional: E5 selection improves compute–performance frontier.

---

## 7) Immediate “To‑Do” Patch List (Highest leverage engineering)

If you do only these items, you unblock nearly everything:

1) **Canonical snapshot schema + extractor** (write once, use everywhere).  
2) **Teacher hint generation wiring** (`generate_text` or call `generate`).  
3) **ModelFactory LoRA loading** for evaluation + micro-training.  
4) **Evaluation harness** that runs end‑to‑end and from snapshot states, producing CSVs.  
5) Fix baseline sizing (B1_uniform) and remove placeholder plotting.

---

## 8) File outputs checklist (so analyses are reproducible)

- `results/phase0/snapshots.parquet`  
- `results/phase1/labeled_snapshots.parquet`  
- `results/phase1b/cpt_validation_results.parquet`  
- `results/phase2/models/{run_id}/...` (14 runs)  
- `results/phase3/per_quadrant_results.csv`  
- `results/phase3/transfer_matrix.csv`  
- `results/phase3/retention.csv`  
- `results/phase3/stuckness.csv`  
- `results/figures/figure1.pdf` … `figure6.pdf`  
- `results/tables/table1.csv` …

