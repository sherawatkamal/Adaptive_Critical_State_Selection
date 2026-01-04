# Results Outline (Template): Teachable Moments from Failures

> **Central spine**: an empirically grounded *teachability model* for agent training from failures.
>
> 1) **What is a teachable moment?**  
> 2) **How do we identify it cheaply and reliably?**  
> 3) **What do we do with it** (data + training algorithms) **to close the reliability last‑mile without forgetting?**

This file is intentionally **results-only** (what we will show + how to interpret), organized by hypothesis.

---

## Figure and Table Index

### H1 — Teachability heterogeneity is real and structured
- **Fig H1.1** Teachability landscape: *U × L_local* scatter colored by *ELP_net* (+ quadrant cuts)
- **Fig H1.2** ELP distribution by quadrant (violin/box + effect sizes)
- **Fig H1.3** Route/format prevalence by quadrant (stacked bars)
- **Fig H1.4** Leverage decomposition: *(L_upper vs L_local)* and leverage-profile breakdown
- **Fig H1.5 (optional)** Qualitative case studies: 2–4 snapshots showing {high‑ELP vs low‑ELP} with patches and rollouts
- **Tab H1.1** Quadrant characteristics summary (counts, medians, route mix, dead-end rate)
- **Tab H1.2** Correlation table: {U, L_local, L_upper, depth, ELP_net} (+ Spearman)

### H2 — CPT is a valid proxy for training utility
- **Fig H2.1** CPT validation: *ELP_net* vs micro-training improvement (Δsuccess / Δreward)
- **Fig H2.2** CPT validity by quadrant (correlation bars + confidence intervals)
- **Tab H2.1** Correlation metrics (Pearson/Spearman, overall + per quadrant)
- **Tab H2.2** Placebo sanity checks (mean placebo gain ≈ 0; patch helpful only when contentful)

### H3 — Different supervision works best in different quadrants
- **Fig H3.1** 4×3 heatmap: (quadrant × supervision) in-quadrant success
- **Fig H3.2** Best format per quadrant (Δ over DEMO baseline + CIs)
- **Tab H3.1** Best supervision per quadrant + effect sizes + significance
- **Tab H3.2** Training budget parity table (N examples, tokens, steps, wall-clock)

### H4 — Cross-quadrant transfer is limited
- **Fig H4.1** Transfer matrix: train quadrant → eval quadrant (Δsuccess)
- **Fig H4.2** Diagonal vs off-diagonal summary (generalization gap)
- **Tab H4.1** Transfer matrix numeric table (Δsuccess)
- **Tab H4.2** Diagonal–offdiagonal gap stats (mean, CI)

### H5 — Teachability-guided training closes reliability gaps without catastrophic forgetting
- **Fig H5.1** Reliability improvement vs baseline (targeted eval on failure-panel states)
- **Fig H5.2** Retention curve on base-success tasks (forgetting check)
- **Fig H5.3** Improvement–retention frontier (each method as a point)
- **Fig H5.4** Stuckness metrics (repeat-rate / loop-rate) before vs after
- **Tab H5.1** Main results table (success, Δsuccess, retention drop, stuckness)

### H6 — Teachability predictor identifies high-ELP states cheaply and reliably
- **Fig H6.1** Ranking quality: Precision@K / NDCG@K for predicting top teachable states
- **Fig H6.2** Predicted vs true ELP (calibration / reliability curve)
- **Fig H6.3** Feature ablation (structural-only vs embedding-only vs both)
- **Fig H6.4** Sample efficiency: performance vs # labeled snapshots
- **Fig H6.5 (optional)** Robustness: predictor ranking across checkpoints (teachability drift panel)
- **Tab H6.1** Predictor metrics table (NDCG@{10,50}, Spearman, quadrant accuracy)
- **Tab H6.2** Cost table (labeling rollouts vs predictor inference time)

### H7 — Predictor-selected data improves training efficiency (compute-feasible selection story)
- **Fig H7.1** Selection comparison at fixed budget K: Random vs Entropy vs Predictor vs Oracle
- **Fig H7.2** Improvement–retention frontier for selection methods
- **Tab H7.1** Selection experiment summary (Δsuccess, retention, stuckness)

---

## H1 — Teachability heterogeneity is real and structured

### Claim
Failure states are not equally useful for training; teachability concentrates in specific *U × L_local* regimes and is measurable.

### Evidence
**Fig H1.1 — Teachability landscape (hero)**
- Plot: U(s) (entropy) vs L_local(s), color = ELP_net
- Expectation: high ELP_net clusters in **Q1 (high U, high L)** and **Q4 (low U, high L)**
- Interpretation: *fixable* failures exist, and they are structurally distinct.

**Fig H1.2 — ELP distribution by quadrant**
- Violin/box; report pairwise effect sizes (Cliff’s delta) + bootstrap CI
- Expectation: median/mean ELP_net(Q1,Q4) > ELP_net(Q2,Q3)

**Fig H1.3 — Route distribution by quadrant**
- Stacked bars: {DEMO, HINT, CONTRAST}
- Expectation: Q1 skew → HINT; Q4 skew → CONTRAST

**Fig H1.4 — Leverage decomposition**
- Scatter L_upper vs L_local; annotate leverage profiles: bottleneck vs trajectory-level vs dead-end
- Expectation: dead-ends (low L_upper) have low ELP_net

### Tables
**Tab H1.1 — Quadrant characteristics**
- N snapshots; median U; median L_local; mean ELP_net; % route types; % dead-end (L_upper<thr)

**Tab H1.2 — Correlation matrix**
- Spearman + Pearson between U, L_local, L_upper, depth, ELP_net

### Pass/Fail criteria (actionable)
- ANOVA or Kruskal-Wallis shows quadrant ELP differs (p < 0.01)
- At least one quadrant shows **substantially higher** ELP_net (effect size > small)

---

## H2 — CPT is a valid proxy for training utility

### Claim
CPT’s placebo-controlled gain (*ELP_net*) is predictive of actual learning progress from micro-training.

### Evidence
**Fig H2.1 — CPT validation scatter**
- x: ELP_net; y: Δmicro (success-from-snapshot or reward)
- Expectation: positive correlation overall

**Fig H2.2 — CPT validity by quadrant**
- Report correlation per quadrant; expect strongest in quadrants with meaningful leverage (Q1/Q4)

### Tables
**Tab H2.1 — Correlation metrics**
- Overall Pearson/Spearman r with CI; per quadrant breakdown

**Tab H2.2 — Placebo sanity checks**
- Mean placebo gain ~0; content patches ≫ placebo in high-ELP bins

### Pass/Fail criteria
- Spearman r ≥ 0.4 overall (or ≥0.5 in Q1/Q4) with p < 0.05

---

## H3 — Different supervision works best in different quadrants

### Claim
The optimal training supervision format depends on the teachability regime.

### Evidence
**Fig H3.1 — 4×3 heatmap (quadrant × supervision)**
- Metric: in-quadrant success on held-out failure snapshots
- Expectation:
  - Q1 best: HINT (or comparable)
  - Q4 best: CONTRAST
  - Q2/Q3: smaller gains / harder

**Fig H3.2 — Best format per quadrant**
- Bar plot of best − DEMO per quadrant with bootstrap CI

### Tables
**Tab H3.1 — Best supervision table**
- Argmax format per quadrant; Δ; CI; p-value vs DEMO

**Tab H3.2 — Budget parity**
- Ensure all runs comparable (same N snapshots or same tokens)

### Pass/Fail criteria
- Best format differs from DEMO in ≥2 quadrants with non-trivial effect size

---

## H4 — Cross-quadrant transfer is limited

### Claim
Training on one quadrant primarily improves that quadrant; transfer to others is weaker.

### Evidence
**Fig H4.1 — Transfer matrix**
- Rows: training quadrant; cols: eval quadrant; values: Δsuccess vs base
- Expectation: diagonal dominance

**Fig H4.2 — Diagonal vs off-diagonal**
- Show mean diagonal Δ vs mean off-diagonal Δ

### Tables
**Tab H4.1** Transfer matrix (numeric)

**Tab H4.2** Diagonal–off-diagonal gap

### Pass/Fail criteria
- Mean(diagonal) − Mean(offdiag) ≥ 0.10 (or clearly positive with CI excluding 0)

---

## H5 — Teachability-guided training closes reliability gaps without catastrophic forgetting

### Claim
Targeted training on teachable failure states improves reliability where it matters and does not catastrophically degrade prior skills.

### Evidence
**Fig H5.1 — Reliability improvements on failure-panel states**
- Compare: Base vs Standard-SFT vs **Recovery-Training (L_upper)** vs Baselines

**Fig H5.2 — Retention curve**
- Evaluate on base-success tasks; retention drop should be small

**Fig H5.3 — Frontier**
- x = retention drop (lower better), y = reliability gain (higher better)

**Fig H5.4 — Stuckness reduction**
- Repeat-rate/loop-rate changes

### Table
**Tab H5.1 — Main results**
- success_on_failures, Δsuccess_on_failures, success_on_base_tasks, retention_drop, stuckness

### Pass/Fail criteria
- Reliability gain on target failure set is statistically > 0
- Retention drop < 5–10% absolute (configurable)

---

## H6 — Teachability predictor identifies high-ELP states cheaply and reliably

### Claim
A lightweight predictor (Tier1+Tier2) can rank states by teachability (ELP) well enough to replace expensive CPT for selection.

### Evidence
**Fig H6.1 — Ranking curves**
- Precision@K / NDCG@K for selecting top-K teachable states
- Compare to baselines: random, entropy-only, L_local-only, U×L heuristic

**Fig H6.2 — Calibration / reliability**
- Bin by predicted ELP; plot mean true ELP per bin

**Fig H6.3 — Feature ablation**
- Structural-only vs embedding-only vs both

**Fig H6.4 — Sample efficiency**
- Train on {100, 250, 500, 1000} labeled snapshots; plot NDCG@K

### Tables
**Tab H6.1 — Predictor metrics**
- NDCG@{10,50}, Spearman, quadrant accuracy

**Tab H6.2 — Cost**
- Human/API rollouts + teacher calls vs predictor inference

### Pass/Fail criteria
- Predictor beats entropy baseline on NDCG@K by a clear margin

---

## H7 — Predictor-selected data improves training efficiency

### Claim
Training on predictor-selected teachable states yields better reliability gains at fixed budget than random or entropy selection, without increasing forgetting.

### Evidence
**Fig H7.1 — Selection comparison (fixed K)**
- Methods: Random, Entropy, Predictor-ELP, Oracle-ELP
- Train *one* LoRA SFT model per method with the same budget K

**Fig H7.2 — Frontier (selection)**
- Plot reliability gain vs retention drop for these methods

### Table
**Tab H7.1 — Selection summary**
- Δsuccess_on_failure_panel, retention_drop, stuckness

### Pass/Fail criteria
- Predictor-selected > entropy-selected > random-selected on reliability gain
- Retention comparable (no method causes large drop)

---

## Minimal set to ship a tight-deadline paper

If schedule/compute is tight, prioritize these:
- **H1**: Fig H1.1 + Tab H1.1
- **H2**: Fig H2.1 + Tab H2.1
- **H3/H4/H5**: Fig H3.1 + Fig H4.1 + Fig H5.3 + Tab H5.1
- **H6/H7** (predictor story): Fig H6.1 + Fig H7.1

---

## Appendix — Mechanistic Understanding

### Claim
Failures in teachable states are characterized by specific breakdowns in state tracking or goal adherence, which are measurable.

### Evidence
**Fig A.1 — Mechanistic metrics by Quadrant**
- Radar chart or Bar plot: Action Accuracy, State Understanding, Goal Tracking
- Expectation: High-ELP states show specific deficits (e.g. low State Understanding) that are recoverable.

