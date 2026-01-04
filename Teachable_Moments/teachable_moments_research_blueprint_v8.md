# Teachable Moments from Failures: Research Blueprint v8 (Per-Quadrant Training Focus)

> **North star**: Build a *practical, empirically grounded* **teachability model** for agent training from failures:
>
> **(1) What is a teachable moment?**  
> **(2) How do we identify it cheaply and reliably?**  
> **(3) What do we do with it (data + training algorithms) to close the reliability last-mile without forgetting?**
>
> The framework applies to both expert-generated and student-generated failures, requiring only a warm-started policy with meaningful action distributions.
>
> v8 update: **Per-quadrant training** as main experiment (14 runs); **aligned estimator naming** (A=single-step, B=full); **micro-training CPT validation**; **simplified CPT** (single-stage n=2); **drop Estimator C**; Tier 1+2 features only for predictor; **task division** for Kamal (PhD) and Chris (undergrad).

---

## 0) Scope control & risk management (so we finish and still hit ICML caliber)

### 0.1 Core story (must land)
We aim for a paper that is strong even if some exploratory parts underperform.

**Core deliverables**
1. **Teachability model + instruments**: measurable axes (U × L) + a small suite of probes that separate *genuinely teachable* from *wasteful* states.
2. **Per-quadrant training experiment**: 14 training runs (4 quadrants × 3 supervision types + 2 baselines) testing which supervision works where.
3. **CPT validation via micro-training**: empirical correlation between in-context patching and actual training improvement.

**What makes this ICML-grade**
- We are not just proposing a pipeline; we are contributing a **measurable model**, **diagnostic instruments**, and **regime-aware experiments** that explain *why* some interventions help.
- **Per-quadrant training** directly tests the core hypothesis that different teachability regimes benefit from different supervision.

### 0.2 High-upside additions (only after core is stable)
- Teachability predictor for scaling (Tier 1+2 features).
- Teachability drift analysis across checkpoints.
- Per-regime supervision format analysis beyond the main experiment.
- Setup B (student-failure) + ScienceWorld transfer.
- Bounded DPO vs SFT ablation for CONTRAST-route subset.

### 0.3 "Feasibility knobs" (explicit budget controls)
- **Leverage estimation** uses two estimators (A: single-step, B: full control); **Estimator C dropped** for this sprint.
- **CPT** uses single-stage allocation (n=2 per condition) instead of adaptive two-stage.
- **Per-quadrant training** is the main experiment: 4 quadrants × 3 supervision types + 2 baselines = 14 runs.
- **Micro-training validation** on 200 snapshots for CPT correlation check.
- **Predictor** uses Tier 1+2 features only (structural + embeddings).
- **Drift panel** reduced to 200 snapshots with n=1 rollout.
- Routing is an **analysis label**, not a training-time objective switch.

---

## 1) Thesis and key contribution statements (ICML spine)

### 1.1 Thesis
Failure trajectories are a massive, under-used training resource, but **not all failure moments are equal**. The central challenge is **heterogeneity**:

- Sometimes the agent knows it's uncertain (true ambiguity).
- Sometimes it is confident but wrong (overconfidence / misgrounding).
- Sometimes it is effectively stuck (low action leverage at the current point).
- Sometimes it could recover, but only by "undoing" a few steps or restarting.

A **teachability model** makes this heterogeneity explicit and measurable, enabling **surgical improvements** (reliability / corner cases) while preserving existing skills (plasticity vs forgetting), and revealing how teachability shifts as the student improves (student dependence).

### 1.2 Core contributions (3 bullets; v8)
1. **Teachability model + instruments (U × L_local axes).**  
   We formalize teachability around:
   - **Uncertainty U(s)**: how uncertain the student is at the decision point.
   - **Actionability L_local(s)**: whether fixing the current action can materially change outcomes (Estimator A: single-step).
   - **Recovery ceiling L_upper(s)**: maximum possible improvement with full expert control (Estimator B: full control).

2. **Contextual Patch Test (CPT): a validated proxy for learning payoff.**  
   We introduce a cheap "in-context intervention" probe with placebo controls that estimates how much a minimal teaching signal helps the student recover from a failure:
   - **Placebo-controlled gains**: isolate content-specific effects from prompt-framing artifacts.
   - **ELP(s)**: expected learning payoff proxy at state *s* (net of placebo).
   - **Micro-training validation**: empirical correlation check between CPT gains and actual training improvements.

3. **Per-quadrant training experiment + mechanistic validation.**  
   Using the U × L_local quadrant structure, we:
   - train separate models on data from each quadrant with each supervision type,
   - evaluate which supervision is most effective in which regime,
   - compute cross-quadrant transfer matrix,
   - track stuckness/efficiency metrics and retention.

> **Important stance:** We use routing as an analysis label, not a training-time objective switch. All training is SFT; supervision format varies by experimental condition. This simplifies engineering while preserving the core insight that different teachability regimes benefit from different supervision styles.

---

## 2) Positioning: sharp deltas vs existing work (incl. ACSS)

We want one crisp intro sentence:

> Prior work uses failures as data or diagnoses errors, but lacks a **state-level, model-conditional teachability model** that predicts *learning payoff* and guides *which supervision to use* while measuring retention and last-mile reliability.

### EEF (Exploring Expert Failures)
EEF uses simulation from intermediate states of failed expert trajectories to identify beneficial segments for student training.  
**Our delta:** EEF is an algorithm; we provide a **teachability model** (what/why) + **CPT proxy** (how to measure at scale) + **per-quadrant experiments** (what supervision works when), with a reliability/forgetting framing.

### ATLaS / "critical steps" methods
Critical-step selection methods focus on identifying important steps in expert trajectories for efficiency/generalization.  
**Our delta:** "critical" is not "teachable." We focus on **failure states** and model-conditional learning payoff.

### STeP / reflection-based trajectory synthesis
Reflection methods improve trajectory quality by adding corrections/reflections.  
**Our delta:** we study **which moments are teachable and which supervision works**, and use the model to route or select data. Reflection becomes one *candidate* supervision in one regime.

### AgentDebug / AgentErrorTaxonomy / AgentErrorBench
These emphasize diagnosing failure types and generating debugging feedback.  
**Our delta:** we are not building a taxonomy per se; we are predicting **learning payoff** and training utility from failures. Taxonomy/benchmarks can be used as optional external labels for analysis.

### How we differ (high-level)
- **Learning from expert failures pipelines**: leverage failures to mine beneficial segments; typically don't model *why* a state is teachable or how teachability shifts with student capability.
- **Critical-step selection (expert trajectories)**: selects a sparse subset of "important" steps for efficiency; not focused on failure-specific learning payoff or correction difficulty.
- **Debugging/taxonomy work**: classifies errors and provides fixes; not directly optimizing **training utility** nor connecting categories to measurable learning payoff and retention.


---

## 3) Teachability model: axes, probes, quadrants, and "aha moments"

### 3.1 Axis 1: Student ambiguity / uncertainty U(s) (observable)
Let **U(s)** quantify how uncertain the student is at state *s*. We draw on established uncertainty quantification methods:

**Uncertainty estimators (ordered by computational cost):**
- **Entropy**: H(π(·|s)) = -Σ_a π(a|s) log π(a|s)
- **Margin**: π(a_1|s) - π(a_2|s) where a_1, a_2 are top-2 actions
- **Top-k spread**: π(a_1|s) - π(a_k|s)
- **Effective actions**: exp(H), the perplexity of the action distribution
- **Length-normalized sequence log-probability** over the option tokens (assuming each option comes with a description)

We do not commit to a single estimator a priori. During Phase 1, we compute these estimators for all snapshots, then evaluate which correlates best with ELP. The primary estimator is **entropy** for quadrant assignment.

**Implementation note**: For discrete action spaces (WebShop), these estimators are sufficient and near-zero cost.

### 3.2 Axis 2: Actionability / Counterfactual Leverage L(s) (measured)

Let **L(s)** quantify whether doing something different at *s* can meaningfully change success probability. We define **two** leverage estimators with aligned naming:

**Estimator A: Single-step expert leverage (current action focus)**
- The expert suggests the best action at state s, then the student policy continues for remaining steps.
- p_force(s) = P(success | start at s, force expert's action at step 0, then student policy)
- **L_local(s) = p_force(s) - p_policy(s)**
- Interpretation: How much does fixing the **current action** help?
- Budget: 7 rollouts per snapshot

**Estimator B: Expert upper bound (full takeover)**
- The expert teacher takes full control from state s and acts for all subsequent steps until episode termination.
- p_expert(s) = P(success | start at s, expert policy for all remaining steps)
- **L_upper(s) = p_expert(s) - p_policy(s)**
- Interpretation: What's the **best possible outcome** from here?
- Side benefit: successful expert continuations become training data that directly addresses the failure mode.
- Budget: 2 rollouts per snapshot (expert is deterministic or low-variance).

**Key relationships:**
- **High L_local** → Current action is the bottleneck; **high actionability**
- **High L_upper, low L_local** → Problem is downstream, not current action; **low local actionability**
- **L_upper ≈ L_local** → Single action fix captures most of the gap
- **Low L_upper** → State is a dead-end; even expert can't recover

**Actionability for quadrant assignment:**
- Use **L_local (Estimator A)** as the actionability axis
- Threshold: median or 75th percentile of L_local distribution

**Baseline measurement:**
- p_policy(s) = P(success | start at s, student policy continues) 
- Computed as part of Estimator A rollouts (7 rollouts shared)

**Total rollout budget: ~9 episodes per snapshot** (7 for p_policy/p_force + 2 for p_expert)

**Mapping to implementation variables:**
```python
# In code, use these names:
L_local = p_force - p_policy      # Estimator A (single-step actionability)
L_upper = p_expert - p_policy     # Estimator B (full control upper bound)

# Actionability threshold for quadrant assignment
high_actionability = (L_local > L_threshold)  # Estimator A > threshold
```
### 3.2.1 Leverage Profile Classification (for Analysis)

For interpreting per-quadrant results, we classify snapshots by leverage profile:

```python
def get_leverage_profile(L_local: float, L_upper: float) -> str:
    """
    Classify snapshot by leverage profile for analysis.
    
    Profiles:
    - A_bottleneck: Single action is the bottleneck (high L_local, small gap)
    - B_trajectory: Needs trajectory-level help (low L_local, large gap)  
    - C_deadend: Even expert can't recover (low L_upper)
    """
    L_gap = L_upper - L_local
    
    if L_upper < 0.3:
        return "C_deadend"
    elif L_local > 0.3 and L_gap < 0.2:
        return "A_bottleneck"
    elif L_local < 0.2 and L_gap > 0.2:
        return "B_trajectory"
    else:
        return "mixed"
```

**Expected patterns by profile:**
- **A_bottleneck**: Should show highest improvement with Demo supervision (single action fix is sufficient)
- **B_trajectory**: May need Hint supervision or show limited improvement (problem is downstream)
- **C_deadend**: Should show minimal improvement regardless of supervision (validates dead-end classification)

This classification is used for **post-hoc analysis** to understand why certain quadrants respond differently to supervision types.

### 3.3 Contextual Patch Test (CPT) for learning payoff (v8: single-stage, placebo-controlled)

We need a proxy that connects states to "what teaching helps" without running full fine-tunes.

**CPT idea:** insert a minimal "teaching patch" into the agent context, then re-run from a few steps before failure. Critically, we also validate that patches do not degrade performance on successful trajectories.

#### 3.3.1 Patch types and prompt templates

**PLACEBO (control):**
Content-neutral, matches format and approximate length of real patches.
```
[System note: Review the current situation carefully before selecting your next action.]
```
Requirements:
- No task-specific hints
- No reasoning interventions ("think step by step" is a real intervention, not placebo)
- Similar token count to real patches (~20-40 tokens)

**DEMO (demonstration):**
Shows the correct action with brief rationale, framed as a contextual example.
```
[Example from a similar situation]
In a situation like this, where the observation shows: "{brief_observation_summary}"
The correct action was: {teacher_action}
Reason: {rationale}
[End of example]

Now continue with your task:
```

**CONTRAST (preference):**
Explicitly contrasts the wrong action (what the agent chose or would choose) with the correct action.
```
[Feedback on action choice]
In a situation like this, where the observation shows: "{brief_observation_summary}"
- Avoid: {bad_action} — This fails because: {why_bad}
- Instead: {teacher_action} — This works because: {why_good}
[End of feedback]

Now continue with your task:
```

**HINT/DIAGNOSIS:**
Provides diagnostic insight without directly giving the action. Framed as a contextual observation, not a general rule.
```
[Observation from a similar situation]
When facing: "{brief_observation_summary}"
A key insight was: {diagnosis}
(For example: "The 'Add to Cart' button is available but requires scrolling down" 
or "The search results don't match the criteria; backtracking to search is needed")
[End of observation]

Now continue with your task:
```

**Important framing principle:** All patches are presented as examples from similar situations, not as general instructions. This prevents confusion when the patch content doesn't apply to other tasks and makes the intervention more naturalistic.

#### 3.3.2 CPT protocol (v8: single-stage, n=2 per condition)

Given a snapshot s (typically chosen 2-5 steps before failure):
1. Run baseline rollout from s for n=2 episodes → `p_base(s)`
2. Run placebo rollout from s for n=2 episodes → `p_placebo(s)`
3. For each real patch type m ∈ {DEMO, CONTRAST, HINT}:
   - create patched prefix using templates above
   - run n=2 episodes → `p_m(s)`
   - compute raw gain: `Δ_raw[m] = p_m(s) - p_base(s)`
   - compute net gain: `Δ_net[m] = p_m(s) - p_placebo(s)`

Then label:
- `ELP_net(s) = max_m Δ_net[m]` (net of placebo)
- `ELP_raw(s) = max_m Δ_raw[m]` (for comparison)
- `route(s) = argmax_m Δ_net[m]`

**Total episodes per snapshot: 10** (2 per condition × 5 conditions)

**Logging requirement:**
Store per-snapshot: `p_base`, `p_placebo`, `p_demo`, `p_contrast`, `p_hint`, `ELP_net`, `ELP_raw`, `route_net`, `total_episodes`, `seed`.

**Why CPT matters**
- It's aligned with what we do downstream (supervision format varies by route).
- Placebo control isolates content-specific effects.
- It captures "confident mistakes" where uncertainty is low but a contrast/hint fixes behavior.
- Single-stage allocation is simpler and sufficient for 14-run experiment.

### 3.4 CPT Validation Protocol: Micro-Training (Phase 1B)

CPT claims that in-context patching predicts training benefit. We validate this with micro-training experiments.

#### 3.4.1 Micro-Training Validation

**Goal**: Correlate CPT scores with actual fine-tuning improvement.

**Protocol**:
1. Sample N=200 snapshots, stratified by quadrant (50 per quadrant)
2. For each snapshot s:
   a. Compute CPT scores: ELP_net, route, per-patch gains
   b. Create single-example training data in route-matched format
   c. Fine-tune base model with 1-2 LoRA gradient steps on this example
   d. Evaluate micro-trained model on:
      - The failure trajectory (starting from snapshot state)
      - 10 similar tasks (near-failure validation set)
   e. Record improvement: Δ_micro = success_after - success_before

**Correlation analysis**:
- Scatter plot: ELP_net vs Δ_micro
- Expected: ρ(ELP_net, Δ_micro) > 0.3 for validation to pass
- Stratify by quadrant to check if correlation holds within each

**Advantages over full training**:
- 1-2 LoRA steps per snapshot vs full fine-tuning
- Isolates single-example effect
- Computationally tractable for N=200

**Budget**: 
- 200 snapshots × 1 micro-training = 200 LoRA updates
- 200 × 11 evaluation episodes = 2,200 episodes
- Total: ~3 GPU-hours + ~2,200 episodes

#### 3.4.2 Validation Criteria

| Check | Metric | Pass | Fail Action |
|-------|--------|------|-------------|
| CPT predicts micro-training | ρ(ELP_net, Δ_micro) | > 0.3 | Report L_local selection as primary |
| Quadrant-specific correlation | ρ within each quadrant | > 0.2 | Note which quadrants CPT fails for |
| Placebo sanity | median(Δ_net) > 0 | Yes | Revise placebo template |
| Route prediction | Route accuracy | > 40% | Use uniform format |

#### 3.4.3 Parallel Execution

Validation runs **in parallel** with main experiment (Phase 1A), not as a blocking gate.
Results inform interpretation of main results.

### 3.5 Recovery depth d(s) as an offline severity instrument

Depth helps interpret when local corrections are futile because the agent is already beyond recovery. With the two leverage estimators from §3.2, we obtain depth "for free" by computing leverage at fixed intervals along the trajectory.

**Two recovery depth measures:**

Given a failure trajectory (s_0 ... s_T) where we compute leverage at states {s_0, s_k, s_{2k}, ...} for interval k:

1. **d_expert(s_t)**: steps back until expert can fully recover
   - Smallest d where p_expert(s_{t-d}) ≥ τ_recover (e.g., 0.5)
   - Interpretation: how far back before the task becomes "easy" for a strong policy
   - Often small (expert can recover from most states)

2. **d_force(s_t)**: steps back until single expert action enables student recovery
   - Smallest d where p_force(s_{t-d}) ≥ τ_recover
   - Interpretation: how far back before one correct action unlocks student success
   - The key teaching-relevant depth measure

**Recovery depth relationships:**

Typically: d_expert ≤ d_force

The gap reveals teachability characteristics:
- **d_force - d_expert small**: single correct action is nearly as good as full expert takeover → high local teachability
- **d_force - d_expert large**: student struggles even after correct action → needs more than one-shot correction

**Compute efficiency:**

Since we compute leverage (p_expert, p_force, p_policy) at snapshots sampled at fixed intervals along trajectories:
```
Trajectory: s_0 → s_5 → s_10 → s_15 → s_20 (failure)
Snapshots:  [    snap_1  snap_2  snap_3  snap_4 ]
```

For each snapshot, we already have (p_expert, p_force). Depth is simply:
- Walk backward through snapshots until p_* exceeds threshold
- Depth in steps = snapshot_interval × number_of_snapshots_back

**No additional rollouts required** beyond what's computed for leverage.

**Feasibility stance**
- Depth is computed **offline** on stored snapshots at fixed intervals (e.g., every 3-5 steps).
- We do **not** assume a deployed agent can arbitrarily reset.
- Depth is used for analysis, stratification, and motivating early abort/restart training.

### 3.6 Four Quadrants: Primary Experimental Framework

The U × L quadrant structure is the **primary organizing principle** for our experiments, not just an analysis tool.

#### 3.6.1 Quadrant Definitions

| Quadrant | Uncertainty (U) | Actionability (L_local) | Interpretation |
|----------|-----------------|-------------------------|----------------|
| Q1 | High | High | Uncertain and fixable |
| Q2 | High | Low | Uncertain and stuck |
| Q3 | Low | Low | Confident and stuck |
| Q4 | Low | High | Confident but wrong |

**Threshold computation:**
- U_threshold = median(U) across all snapshots
- L_threshold = median(L_local) across all snapshots
- Alternative: Use 33rd/67th percentiles for three-way split

**Quadrant assignment:**
```python
def assign_quadrant(U: float, L_local: float, 
                    U_threshold: float, L_threshold: float) -> str:
    high_U = U > U_threshold
    high_L = L_local > L_threshold
    
    if high_U and high_L:
        return "Q1_highU_highL"
    elif high_U and not high_L:
        return "Q2_highU_lowL"
    elif not high_U and not high_L:
        return "Q3_lowU_lowL"
    else:  # low U, high L
        return "Q4_lowU_highL"
```

#### 3.6.2 Hypothesized Supervision Effectiveness by Quadrant

| Quadrant | Hypothesis | Rationale |
|----------|------------|-----------|
| Q1 (high U, high L) | Hint/diagnosis most effective | Agent needs direction, not full demo |
| Q2 (high U, low L) | All supervision types may struggle | State may be beyond local repair |
| Q3 (low U, low L) | Low ELP expected | Confident and stuck = hard to teach |
| Q4 (low U, high L) | Contrast most effective | Agent needs to unlearn confident mistake |

These hypotheses are tested empirically in the per-quadrant training experiment.

#### 3.6.3 Per-Quadrant Training Experiment (Main Result)

**Design**: Train separate models on data from each quadrant, using each supervision type.

**Matrix**: 4 quadrants × 3 supervision types = 12 runs

| Run ID | Quadrant | Supervision | Training Data |
|--------|----------|-------------|---------------|
| Q1_demo | Q1 | Demo | All Q1 snapshots, demo format |
| Q1_contrast | Q1 | Contrast | All Q1 snapshots, contrast format |
| Q1_hint | Q1 | Hint | All Q1 snapshots, hint format |
| Q2_demo | Q2 | Demo | All Q2 snapshots, demo format |
| Q2_contrast | Q2 | Contrast | All Q2 snapshots, contrast format |
| Q2_hint | Q2 | Hint | All Q2 snapshots, hint format |
| Q3_demo | Q3 | Demo | All Q3 snapshots, demo format |
| Q3_contrast | Q3 | Contrast | All Q3 snapshots, contrast format |
| Q3_hint | Q3 | Hint | All Q3 snapshots, hint format |
| Q4_demo | Q4 | Demo | All Q4 snapshots, demo format |
| Q4_contrast | Q4 | Contrast | All Q4 snapshots, contrast format |
| Q4_hint | Q4 | Hint | All Q4 snapshots, hint format |

**Baselines (2 runs)**:
- B1_uniform: Uniform random selection across all quadrants, demo format
- B2_all: All quadrants combined, demo format

**Total: 14 training runs**

**Evaluation**: Each model evaluated on:
1. Success rate on held-out tasks
2. Per-quadrant improvement (does training on Q_i help Q_i states?)
3. Cross-quadrant transfer (does training on Q_i help Q_j states?)
4. Stuckness/efficiency metrics
5. Retention on successful tasks

**Expected findings**:
- Q4 (low U, high L) should show highest improvement with contrast format
- Q3 (low U, low L) should show lowest improvement across all formats
- Cross-quadrant transfer may be limited (training on Q1 may not help Q3)

### 3.6.4 Error Type Analysis (Optional Post-Hoc)

Teacher hints include an `error_type` classification. While not used for quadrant assignment, this provides an orthogonal analysis dimension:

**Error types:**
- `affordance_miss`: Failed to notice an available action
- `attribute_confusion`: Confused product attributes  
- `planning_error`: Wrong sequence of actions
- `exploration_failure`: Didn't search/explore enough

**Analysis questions (post-hoc):**
1. Do certain error types cluster in specific quadrants?
2. Do certain error types have higher ELP than others?
3. Does supervision effectiveness vary by error type?

This analysis is **optional** but could reveal finer-grained teachability structure beyond U×L.

### 3.7 "Aha moments": Empirical discoveries to watch for and double down on

Throughout this research, we are planning top-down based on theory. But the most valuable insights often emerge unexpectedly from data. An "aha moment" in this context is an empirical observation that:
- Wasn't anticipated in the research plan
- Reveals something fundamental about teachability
- Warrants doubling down with additional analysis or experiments

**Why this matters:** Students executing this plan may encounter surprising patterns but not recognize their significance, or may notice something interesting but not know how to pursue it. This section provides examples of high-value discoveries to watch for and concrete actions to take.

#### 3.7.1 Example aha moments and how to double down

**Aha #1: "The U×L quadrant boundaries aren't where we expected"**

*What you might see:* When plotting ELP against U and L, the high-ELP region doesn't align with quadrant boundaries. Maybe teachability peaks at medium U, not high U. Or there's a diagonal boundary, not axis-aligned.

*Why it matters:* This suggests the true teachability structure is different from our assumed factorization. Could lead to a better teachability model.

*How to double down:*
- Try alternative axis definitions (e.g., U × depth instead of U × L)
- Report the empirical structure as a finding, propose refined quadrant definitions
- This could become a central contribution: "We discover that teachability is better characterized by X than by U×L"

---

**Aha #2: "One patch type dominates unexpectedly"**

*What you might see:* HINT patches consistently outperform DEMO and CONTRAST across all quadrants, even in Quadrant 4 where we expected CONTRAST to win. Or CONTRAST never helps, even for confident mistakes.

*Why it matters:* Challenges assumptions about intervention design. Might reveal that agents need diagnosis more than examples, or that contrastive signals don't transfer to weights well.

*How to double down:*
- Analyze the winning patch type: what makes HINT work? Is it the diagnostic framing? The specificity?
- Create hybrid patches (DEMO + HINT) to test if effects combine
- Check if this holds after training (not just in-context): does HINT-format supervision also win during fine-tuning?
- Investigate failure cases: when does the dominant patch type fail?

---

**Aha #3: "CPT gains don't predict training gains in one quadrant"**

*What you might see:* CPT-predicted ELP strongly correlates with training improvement in Quadrants 1, 2, 4, but the correlation breaks down in Quadrant 3 (confident stuck). Or vice versa.

*Why it matters:* Reveals where in-context learning diverges from weight updates. Could identify regimes where different selection strategies are needed.

*How to double down:*
- Characterize the failure cases: what's different about snapshots where CPT fails?
- Test alternative proxies for that quadrant (e.g., depth-based selection)
- This becomes a nuanced finding: "CPT is a valid proxy except in regime X, where Y is needed"
- Investigate mechanism: why does in-context help but training doesn't (or vice versa)?

---

**Aha #4: "Teachability drift follows a surprising pattern"**

*What you might see:* As the student improves across checkpoints, teachability doesn't uniformly decrease. Instead, it shifts: Quadrant 4 (confident mistakes) empties out, but Quadrant 2 (uncertain & stuck) grows. Or aha moments cluster at different trajectory positions.

*Why it matters:* Reveals the learning dynamics. Suggests curriculum or iterative training strategies.

*How to double down:*
- Plot teachability landscape at each checkpoint as a 2D heatmap
- Track individual snapshot trajectories across checkpoints (does a snapshot move quadrants?)
- Propose a curriculum: "Train on Quadrant 4 first (confident mistakes), then Quadrant 2 (stuck states) as they emerge"
- Measure: does curriculum ordering improve final performance?

---

**Aha #5: "Certain error types are highly teachable, others aren't"**

*What you might see:* Snapshots where teacher labels error_type = "affordance_miss" have much higher ELP than "planning_error". Or attribute_confusion is teachable in-context but not via training.

*Why it matters:* Provides a finer-grained teachability model based on error taxonomy, not just U×L.

*How to double down:*
- Compute ELP statistics by error_type (mean, variance, fraction with ELP > 0)
- Add error_type as a predictor feature; does it improve ELP prediction?
- Design error-type-specific interventions: maybe affordance_miss needs visual grounding, planning_error needs decomposition
- Propose: "Error type is a stronger predictor of teachability than uncertainty"

---

**Aha #6: "Expert trajectories from leverage computation are surprisingly effective training data"**

*What you might see:* When training on expert continuation trajectories (collected during leverage Estimator B), the model improves more than expected—even more than on curated teachable moments.

*Why it matters:* Suggests that trajectory-level data from failure states is undervalued. Could simplify the pipeline.

*How to double down:*
- Compare: training on expert trajectories vs. training on CPT-selected snapshots
- Analyze: what's special about these trajectories? (They start from failure states, showing recovery)
- Scale up: collect more expert trajectories, measure diminishing returns
- Propose: "Recovery trajectories are the key data type; teachable moment selection is secondary"

---

#### 3.7.2 How to recognize aha moments

Watch for these signals in your experiments:
- **Unexpected rankings:** A method you expected to lose wins, or vice versa
- **Broken correlations:** Two things you expected to correlate don't, or unexpected correlations appear
- **Bimodal distributions:** A metric that should be smooth has two modes—there may be two distinct phenomena
- **Quadrant anomalies:** One quadrant behaves very differently from others
- **Checkpoint surprises:** A metric changes non-monotonically across training

#### 3.7.3 Process for doubling down

When you notice something surprising:

1. **Document immediately:** Write down what you observed, with plots/numbers. Don't wait.
2. **Sanity check:** Rule out bugs, small sample sizes, or data leakage. Reproduce on held-out data.
3. **Quantify the effect:** How big is it? Is it statistically significant? Does it hold across seeds?
4. **Generate hypotheses:** Why might this happen? What would confirm/refute each hypothesis?
5. **Design a targeted experiment:** What's the minimum additional work to validate this finding?
6. **Assess paper impact:** If this holds up, how does it change the story? Is it a side finding or central?
7. **Communicate:** Flag to advisor. Don't bury surprising findings in appendices.

#### 3.7.4 Aha moments we hope to find (aspirational)

The strongest version of this paper would discover:
- A teachability structure that's cleaner and more predictive than U×L
- A supervision format that's robustly effective across regimes (simplifying guidance)
- A failure mode that's highly teachable but currently undertargeted in the literature
- Evidence that teachability-guided selection provides 2-3× compute efficiency gains

Stay alert for evidence toward any of these. They're not guaranteed, but they're what would make this paper memorable.

---

## 4) How to identify teachable moments at scale: predictor targets and features

### 4.1 Outcome labels (v8: simplified, multi-view)

We train a predictor g_φ(s) to output multiple targets that capture different aspects of teachability:

**Primary targets (for selection and routing)**
- **ELP_hat(s)** (regression or ranking): "how much payoff if we teach here?" (net of placebo)
- **route_hat(s)** ∈ {DEMO, CONTRAST, HINT}: which supervision format helps most

**Leverage targets (for understanding recoverability)**
- **L_upper_hat(s)**: predicted gap to expert upper bound
- **L_local_hat(s)**: predicted gap from fixing current action

**Depth targets (for understanding severity)**
- **d_expert_hat(s)**: predicted steps back until expert can recover (ordinal: 0, 1, 2, 3+)
- **d_force_hat(s)**: predicted steps back until single-action fix works (ordinal)

**Derived targets (for stratification)**
- **depth_gap_hat(s)**: d_force - d_expert (measures student's downstream capability)

**Why predict depth?**
Depth prediction enables:
1. **Efficient labeling:** Skip expensive CPT for states where d_expert > 0 (likely dead-ends with low ELP)
2. **Intervention routing:** High d_force but low d_expert → needs trajectory-level help, not single-action demo
3. **Curriculum design:** Train on low-depth (easy) failures first, high-depth (hard) failures later
4. **Early warning:** If predicted d_force is high at an early step, consider restart training

**Auxiliary (for analysis)**
- **U_hat(s)**: predicted uncertainty
- **quadrant_hat(s)**: derived from U_hat and L_local_hat

Quadrant label is derived post-hoc as a reporting tool, not a direct prediction target.

**Multi-task predictor architecture:**
```
Input: features(s)
    ↓
Shared MLP layers
    ↓
    ├── ELP head (regression)
    ├── Route head (3-way classification)
    ├── L_upper head (regression)
    ├── L_local head (regression)
    ├── d_expert head (ordinal classification: 0,1,2,3+)
    └── d_force head (ordinal classification)
```

The multi-task setup encourages the model to learn representations that capture the full teachability picture, not just ELP.

### 4.2 Feature families for teachability prediction (v8: Tier 1+2 only for sprint)

The predictor needs features that can capture nuanced outcomes like recovery depth, actionability, and optimal intervention type. For the ICML sprint, we use **Tier 1+2 features only** (structural + embeddings).

#### Tier 1: Structural features (near-zero cost, always compute)

These capture surface-level properties of the state and trajectory:

**State-level:**
- Uncertainty measures: entropy, margin, top-k spread, effective actions
- Action space: size, presence of escape actions (Back/Search/Restart), action type distribution
- Observation length, presence of key tokens (e.g., "Add to Cart", "error", "not found")

**Trajectory-level:**
- Step index, normalized position (step / typical_episode_length)
- Actions taken so far (one-hot or count features)
- Repeated action count, loop indicators
- Time since last "progress" (e.g., last successful click)

**Task-level:**
- Task complexity proxy (instruction length, number of constraints)
- Task category if available

*Cost:* Essentially free; computed from logged data.
*Expected signal:* Weak for nuanced targets (depth, route), moderate for coarse targets (quadrant).

#### Tier 2: Embedding-based features (cheap, one forward pass)

Use pretrained encoders to capture semantic content:

**Observation embedding:**
- Encode current observation with sentence-transformer (e.g., all-MiniLM-L6-v2) or BERT
- Dimensionality reduction if needed (PCA to 64-128 dims)

**Instruction/goal embedding:**
- Encode task instruction separately
- Compute similarity: cos(obs_embedding, instruction_embedding) as "goal alignment" feature

**Trajectory summary embedding:**
- Concatenate recent N observations, encode as single sequence
- Or: mean-pool embeddings of recent observations

**Contrastive features:**
- Similarity between current observation and "typical success states" (precomputed centroid)
- Similarity to "typical failure states"
- Distance from trajectory start (embedding space)

*Cost:* ~10ms per snapshot (batch efficiently).
*Expected signal:* Moderate; captures semantic similarity but not causal reasoning about recoverability.

#### Deferred: Tier 3 and Tier 4 (LLM-derived and ICL-based)

For the ICML sprint, we defer Tier 3 (LLM-derived features) and Tier 4 (ICL-based prediction) to post-sprint work. These are more expensive and may not be needed if Tier 1+2 achieve sufficient NDCG.

### 4.3 Predictor evaluation

The predictor serves multiple purposes: selecting high-ELP snapshots, routing supervision formats, and understanding teachability structure. Evaluation must cover all uses.

#### 4.3.1 Primary metrics (selection and routing)

**ELP prediction (main selection signal):**
- **NDCG@k** (k = 50, 100, 200): ranking quality for top teachable states
- **Precision@k**: fraction of predicted top-k that are truly high-ELP
- **Spearman correlation**: overall ranking agreement with true ELP
- **Calibration plot**: predicted ELP vs. actual ELP (should be diagonal)

**Route prediction (supervision format):**
- **Macro-F1**: balanced accuracy across DEMO/CONTRAST/HINT
- **Confusion matrix**: which routes are confused with which?
- **Route-conditional ELP**: for states predicted as route R, what's their actual ELP under supervision R?

#### 4.3.2 Leverage and depth prediction metrics

**Leverage components:**
- **L_upper regression:** MSE, Pearson correlation
- **L_local regression:** MSE, Pearson correlation  
- **Leverage gap prediction:** corr(predicted L_upper - L_local, actual)
- **Actionability classification:** if we threshold L_local > 0.2 as "actionable," what's precision/recall?

**Recovery depth:**
- **Ordinal accuracy:** exact match on depth bin (0, 1, 2, 3+)
- **Ordinal MAE:** mean absolute error across bins
- **Depth classification:** precision/recall for "immediately recoverable" (d=0) vs. "needs backtracking" (d>0)
- **Depth gap prediction:** corr(predicted d_force - d_expert, actual)

#### 4.3.3 Comparison baselines

Compare learned predictor against:
- **Entropy-only:** rank by uncertainty, predict DEMO route always
- **Random:** random ELP scores, random routes
- **Heuristic rules:** hand-crafted rules based on step_idx, action space, etc.
- **Oracle features:** predictor with access to p_base (1-rollout probe) as a feature

The predictor should significantly beat entropy-only; approaching oracle indicates good feature design.

#### 4.3.4 Feature importance analysis

For interpretability and future feature engineering:
- **Permutation importance:** shuffle each feature, measure prediction degradation
- **SHAP values:** for individual predictions, which features drive the score?
- **Tier contribution:** how much does each feature tier (1/2) contribute?

Report top-5 most predictive features for each target.

---

## 5) What to do with teachability: supervision choices and meta algorithm (v8: per-quadrant focus)

### 5.1 Unified SFT with format variation

**Core principle:** All training is SFT. Supervision format varies by experimental condition, not by runtime routing.

This simplification:
- Reduces engineering complexity (one training stack)
- Avoids confounds from mixing SFT/DPO/other objectives
- Preserves the insight that different regimes benefit from different supervision styles
- Works identically for both student-failure (Setup A) and expert-failure (Setup B) pipelines

**Supervision data sources by setup:**

| Setup | Source of failed action | Source of correct action | Notes |
|-------|------------------------|-------------------------|-------|
| A (student failures) | Student's action in failed trajectory | Teacher demonstration | No training on student successes to avoid mode collapse |
| B (expert failures) | Expert's action in failed trajectory | Teacher suggestion or student simulation success | Expert partial progress + student completion |

**Supervision formats by type:**

*DEMO (standard SFT):*
```
input: observation + history
output: teacher_action
```

*CONTRAST (preference-as-text, still SFT):*
```
input: "The action {failed_action} was suboptimal. The better choice is {teacher_action} because: {rationale}" + context
output: teacher_action
```
Or:
```
input: "Which action is better: {failed_action} or {teacher_action}? Explain." + context
output: "Choose {teacher_action}. Reason: {rationale}"
```

Note: `{failed_action}` is the student's action (Setup A) or expert's action (Setup B) at the failure point.

*HINT (diagnosis-then-action, still SFT):*
```
input: hint_text + observation + history
output: teacher_action
```

**What we can claim:**
- "Different supervision formats are effective in different teachability regimes"
- "Per-quadrant training reveals which supervision works where"
- "Teachability heterogeneity is measurable"
- "Framework applies to both expert-sourced and student-sourced failures"

**What we defer:**
- "DPO is optimal in region X, SFT in region Y" (optional bounded ablation in Phase 4)

### 5.2 Teacher model specification

The teacher provides correct actions and hints for failed states. This role is identical across setups.

**Teacher configuration:**
- Model: GPT-4o (or Claude 3.5 Sonnet)
- Temperature: 0.3

**Hint prompt template:**
```
You are observing an agent that failed at step {step_idx}.
Observation: {observation}
Agent chose: {agent_action}
Valid actions: {valid_actions}

Provide:
1. suggested_action: the best action to take
2. rationale: 1-2 sentences explaining why
3. error_type: one of [affordance_miss, attribute_confusion, planning_error, exploration_failure]
4. confidence: high/medium/low
```

**Cost control:** 
- Teacher hints are generated once per snapshot and cached.
- For 3,000 snapshots at ~$0.01 per call, budget is ~$30.

---

## 6) Evaluation: end-to-end, mechanistic, behavioral, drift, and validation

Evaluation operates at multiple levels: task success (did training help?), mechanistic (where did behavior change?), behavioral (how did decision-making improve?), and methodological (did our instruments work?).

### 6.1 End-to-end performance
- WebShop task success rate and score (primary)
- ScienceWorld success rate (secondary validation, if Setup B completed)

Report with confidence intervals across seeds. Compare to baselines under matched compute budgets.

### 6.2 Mechanistic evaluation by teachability regime

**Checkpoint-style tests from held-out snapshots:**

For snapshots stratified by quadrant and by predicted teachability (ELP_hat bins):
- **Success-from-state:** Does the trained model succeed more often when starting from this state?
- **First-action correctness:** Does the model take the expert-recommended action?
- **Action distribution shift:** How much did P(action|state) change toward expert's action?

```python
def mechanistic_eval(
    model_before: Policy,
    model_after: Policy, 
    test_snapshots: list[Snapshot],
    expert_actions: dict[str, str],
) -> pd.DataFrame:
    results = []
    for snap in test_snapshots:
        # Success from state
        success_before = evaluate_from_state(model_before, snap, n_rollouts=3)
        success_after = evaluate_from_state(model_after, snap, n_rollouts=3)
        
        # First action correctness
        action_before = model_before.get_action(snap)
        action_after = model_after.get_action(snap)
        expert_action = expert_actions[snap.id]
        
        # Action probability shift
        prob_expert_before = model_before.get_action_prob(snap, expert_action)
        prob_expert_after = model_after.get_action_prob(snap, expert_action)
        
        results.append({
            "snapshot_id": snap.id,
            "quadrant": snap.labels["quadrant"],
            "ELP_bin": bin_elp(snap.labels["ELP_net"]),
            "route": snap.labels["route_net"],
            "success_before": success_before,
            "success_after": success_after,
            "success_delta": success_after - success_before,
            "correct_action_before": action_before == expert_action,
            "correct_action_after": action_after == expert_action,
            "prob_expert_before": prob_expert_before,
            "prob_expert_after": prob_expert_after,
            "prob_shift": prob_expert_after - prob_expert_before,
        })
    
    return pd.DataFrame(results)
```

**Report by stratum:**
- Mean success_delta by quadrant
- Mean success_delta by ELP_bin (validate: high-ELP snapshots should improve more)
- Mean success_delta by route (validate: does matched supervision help more?)

### 6.3 Stuckness and efficiency metrics

Simple metrics that capture behavioral quality without LLM-as-judge:

```python
def compute_stuckness_metrics(trajectory: Trajectory) -> dict:
    actions = [step.action for step in trajectory.steps]
    
    # Repeated actions
    repeated = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i-1])
    repeat_rate = repeated / max(len(actions) - 1, 1)
    
    # Loop detection (same action sequence repeated)
    loops = detect_loops(actions, min_length=2)
    
    # Wasted steps (actions that don't change state meaningfully)
    wasted = count_wasted_steps(trajectory)
    
    # Escape latency (steps until using Back/Search after being stuck)
    escape_latency = compute_escape_latency(trajectory)
    
    return {
        "repeat_action_rate": repeat_rate,
        "loop_detected": len(loops) > 0,
        "n_loops": len(loops),
        "wasted_steps": wasted,
        "escape_latency": escape_latency,
        "efficiency": 1.0 - (wasted / len(actions)) if actions else 0,
    }
```

### 6.4 Behavioral evaluation via LLM-as-judge (optional)

For more nuanced behavioral assessment, use GPT-4o as a judge:

**Trajectory quality judgment:**
```
You are evaluating an AI shopping agent's trajectory on a web shopping task.

Task goal: {goal}
Trajectory:
{formatted_trajectory}

Outcome: {success/failure}

Rate this trajectory on the following dimensions (1-5 scale):

1. **Efficiency:** Did the agent take a direct path or wander unnecessarily?
2. **Exploration quality:** When the agent explored, was it purposeful?
3. **Error recovery:** When the agent made mistakes, did it recover well?
4. **Goal alignment:** Did actions consistently move toward the goal?
5. **Stuck behavior:** Did the agent get stuck or loop?

Provide scores and a 1-sentence justification for each.
Respond in JSON format.
```

**Sampling strategy:**
- 200 trajectories total per model variant
- Stratified by: task difficulty, trajectory length, success/failure, quadrant
- Cost: ~$4-8 per model variant

### 6.5 Teachability drift panel (optional for sprint)

Track how teachability changes as the student improves, using a reduced panel.

**Panel design:**
- 200 snapshots (reduced from 300)
- Stratified by: step position, quadrant, initial ELP
- n=1 rollout per condition (reduced from 3)

**Measurement:**
For each checkpoint (ckpt_0, ckpt_mid, ckpt_final):
1. Run CPT on panel with that checkpoint's policy
2. Record: p_base, p_placebo, Δ_net for each patch type, ELP_net, route

**Drift analysis:**
- ELP distribution shift across checkpoints
- Which quadrants empty out first?
- Which snapshots "graduate" (become easy)?

### 6.6 Retention evaluation

Ensure training on failure data doesn't degrade existing capabilities.

**Retention test set:**
- 100 successful trajectories from training distribution
- Prioritize "near-miss" successes (high uncertainty during trajectory)

**Retention metrics:**
- Success rate preservation: Δ in success rate on retention set (should be ≥ -2%)
- Reward preservation: Δ in mean reward on retention set

**Retention frontier:**
Plot: (failure improvement, retention degradation) for each method
- Ideal: high failure improvement, zero retention degradation
- Acceptable: high failure improvement, small retention degradation

### 6.7 Cross-quadrant transfer matrix

For per-quadrant training experiment, compute transfer matrix:

```python
def compute_transfer_matrix(
    results_df: pd.DataFrame,
    quadrants: list[str],
) -> pd.DataFrame:
    """Compute cross-quadrant transfer matrix."""
    
    transfer = pd.DataFrame(
        index=[f"trained_{q}" for q in quadrants],
        columns=[f"eval_{q}" for q in quadrants],
    )
    
    for trained_q in quadrants:
        for eval_q in quadrants:
            # Find runs trained on trained_q (any supervision type)
            trained_runs = results_df[
                results_df["run_id"].str.startswith(trained_q)
            ]
            # Get mean performance on eval_q
            transfer.loc[f"trained_{trained_q}", f"eval_{eval_q}"] = \
                trained_runs[f"success_{eval_q}"].mean()
    
    return transfer
```

**Expected patterns:**
- Diagonal should be highest (training on Q_i helps Q_i most)
- Off-diagonal reveals transfer/interference patterns
- Q3 (low U, low L) may not transfer well to other quadrants

---

## 7) Experimental setups (prioritized)

### 7.1 Warm-start assumption (applies to both setups)

ACSS and CPT require that the student policy has meaningful structure (non-uniform entropy, ability to sometimes succeed from intermediate states). This does **not** require SFT on expert successes. Any warm-started model suffices:

- Instruction-tuned base model (e.g., Llama-3-8B-Instruct)
- Model fine-tuned on other tasks
- Model from a previous training iteration

The key requirement is that the policy produces informative action distributions. If starting from a random or untrained model, an initial behavior cloning phase on any reasonable demonstration data is needed to bootstrap meaningful entropy signals.

### 7.2 Setup A: student-failure pipeline (Primary)

- Warm-started student generates failures directly.
- Teacher provides demonstrations/preferences/hints.
- **No SFT on student successes:** We avoid training on the student's own outputs to prevent mode collapse and ensure new information flows into the model.
- Beneficial segments come from teacher demonstrations or student simulation (EEF-style) at high-ELP states.
- **When to use:** Cleaner "student-conditional teachability" since failures reflect the current student's actual weaknesses; avoids assumption that student and expert have complementary strengths.

**Scripts:** `run_student_rollouts_v8.py` → `mine_failure_snapshots.py`

### 7.3 Setup B: expert-failure pipeline (Secondary)

- Expert (e.g., GPT-4) generates trajectories; partition into successes and failures.
- Mine teachable moments from failed expert trajectories.
- Identify beneficial segments via teacher demonstrations or student simulation (EEF-style) or CPT patches.
- **When to use:** Expert failures often contain partial progress that a weaker student can complete; good for knowledge distillation scenarios.

**Scripts:** `collect_expert_trajectories.py` → `mine_snapshots_from_expert_trajectories.py`

### 7.4 Setup choice for this paper

We prioritize **Setup A (student-failure-centric)** for the core experiments. This is run first because:
1. Failures reflect the current student's actual weaknesses
2. Avoids distributional mismatch between training data and deployment
3. Aligns with the experimental plan's "Experiment A" specification

Setup B (expert-failure) is a secondary validation that demonstrates the framework generalizes to knowledge distillation scenarios.


---

## 8) Experiment Matrix: Per-Quadrant Training (v8: Main Experiment)

### 8.1 Main Experiment (14 runs)

**Quadrant-specific training (12 runs)**:

| Quadrant | Demo | Contrast | Hint |
|----------|------|----------|------|
| Q1 (high U, high L) | Q1_demo | Q1_contrast | Q1_hint |
| Q2 (high U, low L) | Q2_demo | Q2_contrast | Q2_hint |
| Q3 (low U, low L) | Q3_demo | Q3_contrast | Q3_hint |
| Q4 (low U, high L) | Q4_demo | Q4_contrast | Q4_hint |

**Baselines (2 runs)**:
- B1_uniform: Uniform random selection, demo format
- B2_all: All quadrants combined, demo format

### 8.2 Training Configuration

**Per-run settings**:
- Training examples: All snapshots in target quadrant (expect ~400-600 per quadrant)
- Format: Supervision type determines input formatting; output is always teacher action
- Base model: [specify your base model]
- LoRA rank: 16-32
- Epochs: 3
- Batch size: 8
- Learning rate: 2e-5

**Parallelization**:
- With 4 GPUs: 4 runs in parallel → 4 batches → ~6 hours total training time
- With 2 GPUs: 2 runs in parallel → 7 batches → ~12 hours total training time
- With 1 GPU: Sequential → ~28 hours total training time

### 8.3 Evaluation Protocol

**Per-model evaluation**:
1. **Overall success rate**: 500 held-out tasks, 3 rollouts each
2. **Per-quadrant success**: Evaluate on held-out snapshots from each quadrant, 2 rollouts
3. **Stuckness metrics**: repeat_action_rate, loop_detected, wasted_steps
4. **Retention**: 100 tasks the base model succeeds on, 2 rollouts

**Cross-quadrant transfer analysis**:
- For model trained on Q_i: measure improvement on Q_j states (j ≠ i)
- Creates 4×4 transfer matrix

### 8.4 Expected Results Structure

**Table 1: Per-Quadrant Success Rate**

| Model | Q1 | Q2 | Q3 | Q4 | Overall |
|-------|----|----|----|----|---------|
| Base | x.xx | x.xx | x.xx | x.xx | x.xx |
| B1_uniform | x.xx | x.xx | x.xx | x.xx | x.xx |
| B2_all | x.xx | x.xx | x.xx | x.xx | x.xx |
| Q1_demo | **x.xx** | x.xx | x.xx | x.xx | x.xx |
| Q1_contrast | **x.xx** | x.xx | x.xx | x.xx | x.xx |
| Q1_hint | **x.xx** | x.xx | x.xx | x.xx | x.xx |
| Q2_demo | x.xx | **x.xx** | x.xx | x.xx | x.xx |
| Q2_contrast | x.xx | **x.xx** | x.xx | x.xx | x.xx |
| Q2_hint | x.xx | **x.xx** | x.xx | x.xx | x.xx |
| Q3_demo | x.xx | x.xx | **x.xx** | x.xx | x.xx |
| Q3_contrast | x.xx | x.xx | **x.xx** | x.xx | x.xx |
| Q3_hint | x.xx | x.xx | **x.xx** | x.xx | x.xx |
| Q4_demo | x.xx | x.xx | x.xx | **x.xx** | x.xx |
| Q4_contrast | x.xx | x.xx | x.xx | **x.xx** | x.xx |
| Q4_hint | x.xx | x.xx | x.xx | **x.xx** | x.xx |

Bold = in-quadrant performance (should be highest for that quadrant)

**Table 2: Best Supervision Type by Quadrant**

| Quadrant | Best Supervision | Improvement over Demo | Hypothesis Match? |
|----------|------------------|----------------------|-------------------|
| Q1 | Hint (expected) | +X% | ✓/✗ |
| Q2 | Demo (expected) | +X% | ✓/✗ |
| Q3 | N/A (low ELP) | +X% | ✓/✗ |
| Q4 | Contrast (expected) | +X% | ✓/✗ |

### 8.5 Hypothesis Testing

**Core hypotheses for per-quadrant experiment:**

**H1: Teachability heterogeneity is real and structured**
- Failures cluster into meaningfully different regimes along U×L axes
- These regimes have different optimal interventions (or one dominates with clear margins)

**H2: Different supervision types work best in different quadrants**
- Q1 (high U, high L): Hint/diagnosis most effective
- Q2 (high U, low L): All supervision types may struggle  
- Q3 (low U, low L): Low improvement expected across all formats
- Q4 (low U, high L): Contrast most effective

**H3: Cross-quadrant transfer is limited**
- Training on Q_i primarily helps Q_i states
- Transfer to Q_j is weaker than in-quadrant improvement

**H4: Per-quadrant training outperforms uniform baselines**
- Best per-quadrant model outperforms B1_uniform on its quadrant
- Ensemble of per-quadrant models could outperform B2_all overall

### 8.6 Compute Budget for Main Experiment

| Component | Runs | GPU-hours | Episodes |
|-----------|------|-----------|----------|
| Training | 14 | 28 | - |
| End-to-end eval | 14 × 500 × 3 | - | 21,000 |
| Per-quadrant eval | 14 × 200 × 4 × 2 | - | 22,400 |
| Retention eval | 14 × 100 × 2 | - | 2,800 |
| Stuckness metrics | 14 × 100 × 2 | - | 2,800 |
| **Training subtotal** | | **28** | **49,000** |

For 2-3 week sprint with 4 GPUs, this is feasible (7 hours training time).
For 1 GPU, consider reducing to 2 supervision types per quadrant (8 runs).

---

## 8B) Follow-up Experiments (Post-Sprint or Time Permitting)

These experiments from v7 are retained for follow-up work after the main per-quadrant experiment.

### E5: ELP-Guided Selection Comparison (if predictor is built)

**Hypothesis:** ELP-guided selection improves compute-performance frontier

**Design:**
- Three selection methods: Uniform, Entropy, ELP_hat
- Fixed selection budget: K = 500 snapshots
- Unified SFT with route-matched formats

**This extends the main experiment** by testing whether a predictor can identify high-ELP states without quadrant labels.

### E6: Retention Evaluation

**Hypothesis:** Surgical training preserves retention

Already included in main experiment evaluation.

### E7: Leverage and Depth Analysis

**Hypothesis:** Leverage decomposition and depth provide actionable signals

**Design:**
- Stratify by leverage profile (single action bottleneck vs. needs trajectory help vs. dead-end)
- Stratify by depth (d=0, d=1, d=2+)
- Analyze which profiles respond to which supervision

**Uses existing labeled data from main experiment.**

### E8: Teachability Drift Analysis

**Hypothesis:** Teachability drifts predictably across checkpoints

**Design:**
- Fixed drift panel: 200 snapshots
- Checkpoints: ckpt_0, ckpt_mid, ckpt_final
- Re-run CPT on panel for each checkpoint

**Reduced budget for sprint:**
- 200 × 5 conditions × 1 rollout × 3 checkpoints = 3,000 episodes

### E9: Supervision Format Conditional Analysis

**Hypothesis:** Supervision format matters conditionally

**This is the core of the main per-quadrant experiment.** Additional analysis:
- 3×3 comparison: route × format within each quadrant
- Effect size of matching

### E10: DPO vs SFT Ablation (Optional)

**Hypothesis:** DPO may outperform SFT for CONTRAST-route data

**Design:**
- Take CONTRAST-labeled subset from Q4 (low U, high L)
- Train two models: SFT vs DPO
- Compare on Q4 test snapshots

**Bounded scope:** Only run if main experiment shows CONTRAST format matters.


### Hypothesis Mapping for Per-Quadrant Experiment

| Hypothesis | Test | Success Criteria |
|------------|------|------------------|
| H1: Heterogeneity is real | Quadrant ELP differs significantly | ANOVA F > 5, p < 0.01 |
| H2: Supervision varies by quadrant | Best ≠ Demo in ≥2 quadrants | Per-quadrant best differs |
| H3: Transfer is limited | Diagonal > off-diagonal by >10% | Mean diagonal - mean off-diagonal > 0.1 |
| H4: Per-quadrant beats uniform | Best ensemble > B1, B2 | Significance at p < 0.05 |



---

## 9) Figures and Tables Specifications (Paper Backbone)

This section specifies every figure and table needed for the paper, with exact data sources, visualization details, and generating experiments.

---

### 9.1 Main Paper Figures

#### Figure 1: Teachability Landscape (Hero Figure)

**Purpose:** Establish that teachability heterogeneity exists and is structured.

**Source:** Phase 1 labeling data (`labeled_snapshots.parquet`)

**Panel A: Teachability Plane Scatter**
- X-axis: Uncertainty U(s) (entropy, normalized 0-1)
- Y-axis: Actionability L_local(s) (normalized 0-1)
- Points: ~2,000 failure snapshots
- Color: ELP_net (continuous, viridis colormap)
- Quadrant boundaries: Dashed lines at median U and median L
- Quadrant labels: 
  - Q1: "Uncertain & Fixable" (top-right)
  - Q2: "Uncertain & Stuck" (top-left)
  - Q3: "Confident & Stuck" (bottom-left)
  - Q4: "Confident but Wrong" (bottom-right)
- Annotations: 2-3 example snapshots with callouts

**Panel B: ELP Distribution by Quadrant**
- Type: Violin plot with jittered points
- X-axis: Quadrant (Q1, Q2, Q3, Q4)
- Y-axis: ELP_net
- Statistical annotation: Pairwise significance markers
- Key insight: Q1 and Q4 have higher ELP than Q2 and Q3

**Panel C: Route Distribution by Quadrant**
- Type: Stacked bar chart (normalized to 100%)
- X-axis: Quadrant
- Y-axis: Percentage
- Colors: Demo (blue), Contrast (orange), Hint (green)
- Key insight: Q4 shows more Contrast; Q1 shows more Hint

**Specifications:**
- Size: Full width (~5" × 3")
- Font: 8pt axis labels, 9pt panel labels
- Format: PDF vector

---

#### Figure 2: CPT Validation via Micro-Training

**Purpose:** Demonstrate that CPT predicts actual training gains.

**Source:** Phase 1B micro-training validation (`cpt_validation_results.parquet`)

**Panel A: ELP vs. Micro-Training Improvement**
- X-axis: ELP_net (from CPT)
- Y-axis: Δ_micro (success improvement from micro-training)
- Points: 200 validation snapshots
- Color: By quadrant
- Regression line: With 95% CI shaded
- Annotation: Pearson ρ and p-value in corner

**Panel B: Correlation by Quadrant**
- Type: Bar chart
- X-axis: Quadrant (Q1, Q2, Q3, Q4)
- Y-axis: Pearson ρ
- Reference line: ρ = 0.3 (pass threshold)
- Key insight: Which quadrants have validated CPT

**Specifications:**
- Size: Half width (~3.5" × 2.5")
- This validates the CPT methodology

---

#### Figure 3: Per-Quadrant Training Results (Main Result)

**Purpose:** Show which supervision type works best in each quadrant.

**Source:** Phase 3 evaluation (`per_quadrant_results.csv`)

**Panel A: Success Rate Heatmap**
- Type: 4×3 heatmap
- Rows: Quadrants (Q1, Q2, Q3, Q4)
- Columns: Supervision types (Demo, Contrast, Hint)
- Cell values: Success rate on that quadrant
- Color: RdYlGn colormap (red=low, green=high)
- Annotation: Best supervision per quadrant highlighted

**Panel B: Best Supervision by Quadrant**
- Type: Grouped bar chart
- X-axis: Quadrant
- Y-axis: Success rate (%)
- Bars: Best supervision type for that quadrant
- Comparison: vs. Demo baseline
- Error bars: Std across evaluation

**Panel C: Improvement over Baselines**
- Type: Bar chart
- X-axis: Model (B1_uniform, B2_all, Best per-quadrant)
- Y-axis: Overall success rate
- Key insight: Per-quadrant training outperforms uniform

**Specifications:**
- Size: Full width (~5" × 3")
- Panel A is the headline visual; make it prominent

---

#### Figure 4: Cross-Quadrant Transfer Matrix

**Purpose:** Show transfer/interference patterns across quadrants.

**Source:** Phase 3 transfer analysis (`transfer_matrix.csv`)

**Main Panel: 4×4 Transfer Heatmap**
- Rows: Training quadrant (Q1, Q2, Q3, Q4)
- Columns: Evaluation quadrant (Q1, Q2, Q3, Q4)
- Cell values: Mean success rate
- Color: RdYlGn colormap
- Diagonal highlight: Expected to be highest
- Annotation: Off-diagonal patterns of interest

**Inset: Diagonal vs. Off-Diagonal Summary**
- Type: Small bar chart
- Bars: Mean diagonal, Mean off-diagonal
- Key insight: Quantify transfer limitation

**Specifications:**
- Size: Half width (~3.5" × 3")
- Critical for "limited transfer" claim

---

#### Figure 5: Retention Frontier

**Purpose:** Show surgical training improves failures without regressing on successes.

**Source:** Phase 3 retention evaluation (`retention_results.csv`)

**Main Panel: Pareto Frontier Plot**
- X-axis: Failure set improvement (Δ success rate)
- Y-axis: Retention set change (Δ success rate)
- Points: Each model with error bars
- Quadrant shading:
  - Top-right (green): Ideal
  - Top-left (yellow): Acceptable
  - Bottom-right (orange): Suspicious
  - Bottom-left (red): Bad
- Reference lines: x=0, y=0

**Specifications:**
- Size: Half width (~3" × 3")
- Critical for "surgical training" claim

---

#### Figure 6: Stuckness Metrics Improvement

**Purpose:** Show behavioral improvement beyond task success.

**Source:** Phase 3 stuckness evaluation (`stuckness_results.csv`)

**Panel A: Stuckness Metrics by Model Type**
- Type: Grouped bar chart
- X-axis: Metric (repeat_rate, loop_rate, wasted_steps)
- Y-axis: Value (lower is better)
- Groups: Base, B1_uniform, Best per-quadrant
- Key insight: Per-quadrant training reduces stuck behavior

**Panel B: Efficiency by Quadrant**
- Type: Line plot or bar chart
- X-axis: Quadrant
- Y-axis: Efficiency score
- Lines: Before vs. after training
- Key insight: Which quadrants show efficiency gains

**Specifications:**
- Size: Half width (~3.5" × 2.5")

---

### 9.2 Main Paper Tables

#### Table 1: Per-Quadrant Success Rates

| Model | Q1 | Q2 | Q3 | Q4 | Overall |
|-------|----|----|----|----|---------|
| Base | x.xx | x.xx | x.xx | x.xx | x.xx |
| B1_uniform | x.xx | x.xx | x.xx | x.xx | x.xx |
| B2_all | x.xx | x.xx | x.xx | x.xx | x.xx |
| Q1_demo | **x.xx** | x.xx | x.xx | x.xx | x.xx |
| Q1_contrast | **x.xx** | x.xx | x.xx | x.xx | x.xx |
| Q1_hint | **x.xx** | x.xx | x.xx | x.xx | x.xx |
| ... (continue for all 14 models) |

Bold = in-quadrant performance

---

#### Table 2: Best Supervision by Quadrant

| Quadrant | Best Supervision | Δ vs. Demo | Hypothesis Match? |
|----------|------------------|------------|-------------------|
| Q1 (high U, high L) | Hint (expected) | +X.X% | ✓/✗ |
| Q2 (high U, low L) | TBD | +X.X% | ✓/✗ |
| Q3 (low U, low L) | TBD | +X.X% | ✓/✗ |
| Q4 (low U, high L) | Contrast (expected) | +X.X% | ✓/✗ |

---

#### Table 3: Quadrant Characterization Summary

| Quadrant | N | Mean ELP | Dominant Route | Mean L_local | Interpretation |
|----------|---|----------|----------------|--------------|----------------|
| Q1 | XXX | 0.XX | Demo/Hint | 0.XX | Uncertain & fixable |
| Q2 | XXX | 0.XX | TBD | 0.XX | Uncertain & stuck |
| Q3 | XXX | 0.XX | TBD | 0.XX | Confident & stuck |
| Q4 | XXX | 0.XX | Contrast | 0.XX | Confident but wrong |

---

#### Table 4: CPT Validation Results

| Metric | Value | Pass? |
|--------|-------|-------|
| Overall ρ(ELP, Δ_micro) | X.XX | ✓/✗ |
| Q1 ρ | X.XX | ✓/✗ |
| Q2 ρ | X.XX | ✓/✗ |
| Q3 ρ | X.XX | ✓/✗ |
| Q4 ρ | X.XX | ✓/✗ |

---

### 9.3 Appendix Figures

#### Figure A1: CPT Placebo Control Validation

**Purpose:** Show placebo controls isolate content effects.

**Panel A: Base vs. Placebo vs. Real Patches**
- Type: Box plot
- X-axis: Condition (Base, Placebo, Demo, Contrast, Hint)
- Y-axis: Success probability
- Key insight: Placebo > Base (framing effect); Real > Placebo (content matters)

**Panel B: Raw vs. Net Gains Scatter**
- X-axis: Δ_raw (vs. base)
- Y-axis: Δ_net (vs. placebo)
- Diagonal: y=x reference
- Key insight: Points below diagonal show framing confound

---

#### Figure A2: Example Teachable Moments (Qualitative)

**Purpose:** Qualitative illustration of high-ELP states.

**Format:** 3-4 example panels, each containing:
- Observation text (truncated)
- Agent's chosen action
- Teacher's suggested action
- Patch type that helped
- Before/after success probability

**Selection criteria:**
- One from each quadrant with ELP > 0.3
- Diversity in error types
- Clear visual difference in patch effect

---

#### Figure A3: Leverage Profile Analysis

**Purpose:** Validate leverage profile classification.

**Panel A: Training Improvement by Leverage Profile**
- Type: Bar chart
- X-axis: Leverage profile (A_bottleneck, B_trajectory, C_deadend, mixed)
- Y-axis: Training improvement (%)
- Key insight: Profile A improves most

**Panel B: Leverage Profile by Quadrant**
- Type: Stacked bar chart
- X-axis: Quadrant
- Y-axis: Percentage
- Segments: By leverage profile
- Key insight: Which profiles dominate which quadrants

---

#### Figure A4: Teachability Drift (Optional)

**Purpose:** Show how teachability evolves across checkpoints.

**Panel A: ELP Distribution Shift**
- Type: Overlaid violin plots
- X-axis: Checkpoint (ckpt_0, ckpt_mid, ckpt_final)
- Y-axis: ELP_net
- Key insight: Distribution shifts as student improves

**Panel B: Quadrant Flow (Sankey)**
- Left: Initial quadrant assignment
- Right: Final checkpoint behavior
- Key insight: Which quadrants empty first

---

### 9.4 Data Dependencies for Figures

| Figure | Required Data Files | Generated By |
|--------|---------------------|--------------|
| Fig 1 | `labeled_snapshots.parquet` | Phase 1 (scripts 03-09) |
| Fig 2 | `cpt_validation_results.parquet` | Phase 1B (script 12) |
| Fig 3 | `per_quadrant_results.csv` | Phase 3 (script 18) |
| Fig 4 | `transfer_matrix.csv` | Phase 3 (script 19) |
| Fig 5 | `retention_results.csv` | Phase 3 (script 21) |
| Fig 6 | `stuckness_results.csv` | Phase 3 (script 20) |
| Table 1 | `per_quadrant_results.csv` | Phase 3 (script 18) |
| Table 2 | `per_quadrant_results.csv` | Phase 3 (script 18) |
| Table 3 | `labeled_snapshots.parquet` | Phase 1 (scripts 03-09) |
| Table 4 | `cpt_validation_results.parquet` | Phase 1B (script 12) |
```
---

## 10) Expected outcomes (what we can credibly claim)

**Minimum viable ICML story:**
- Teachability heterogeneity is real and measurable via U × L_local axes
- Per-quadrant training reveals which supervision works where
- CPT (with placebo control) provides a validated ELP proxy via micro-training
- Different quadrants benefit from different supervision (or one dominates with explanation)
- Cross-quadrant transfer matrix reveals learning dynamics

**High-upside "paper pops":**
- Clear per-quadrant format differences matching hypotheses (Q4 → Contrast, Q1 → Hint)
- Transfer matrix shows interesting patterns (e.g., Q4 training hurts Q1)
- CPT validation shows strong correlation (ρ > 0.5)
- Stuckness metrics show large improvement

---

## 11) Task Division for 2-3 Week Sprint

### Assignment Overview

| Phase | Kamal (PhD Lead) | Chris (Undergrad) |
|-------|------------------|-------------------|
| 1A | Per-quadrant training (main experiment) | Dataset construction + labeling |
| 1B | Training infrastructure + runs | CPT validation via micro-training |
| 1C | Evaluation + analysis | Teachability predictor (stretch) |

### Week 1: Foundation + Dataset

| Day | Kamal | Chris |
|-----|-------|-------|
| 1 | Training infrastructure setup | **WebShop state reset verification** |
| 2 | Supervision format templates | Trajectory collection (1000 tasks) |
| 3 | Review labeling pipeline | Snapshot extraction + uncertainty labeling |
| 4 | CPT implementation review | Estimator B (p_expert) on snapshots |
| 5 | Training config finalization | Estimator A (p_force) on snapshots |
| 6 | Dry run: 1 training run | CPT on 500 snapshots |
| 7 | Debug training pipeline | Continue CPT + compute thresholds |

**Week 1 deliverables**:
- Kamal: Working training pipeline, tested on 1 run
- Chris: ~2000 snapshots with U, L_local, L_upper labels; 500+ with CPT; quadrant thresholds computed

### Week 2: Labeling + Training + Validation

| Day | Kamal | Chris |
|-----|-------|-------|
| 8 | Partition data by quadrant | Complete CPT on all snapshots |
| 9 | Generate supervision data (12 configs) | Stratified sample for validation (N=200) |
| 10 | Begin training runs (batch 1: Q1) | Micro-training validation setup |
| 11 | Training runs (batch 2: Q2) | Run micro-training validation |
| 12 | Training runs (batch 3: Q3) | Analyze CPT correlations |
| 13 | Training runs (batch 4: Q4 + baselines) | Retention validation set |
| 14 | Verify all 14 models trained | Generate validation figures |

**Week 2 deliverables**:
- Kamal: 14 trained models
- Chris: CPT validation results (correlation plots), retention validation set

### Week 3: Evaluation + Paper

| Day | Kamal | Chris |
|-----|-------|-------|
| 15-16 | End-to-end evaluation (all models) | Per-quadrant evaluation |
| 17 | Compute transfer matrix | Stuckness metrics |
| 18 | Generate main figures | Generate tables |
| 19 | Draft methods + experiments | Draft appendix |
| 20 | Revise based on results | Supplementary figures |
| 21 | Final paper polish | Final tables + code cleanup |

**Week 3 deliverables**:
- Main paper figures (quadrant results, supervision comparison, transfer matrix)
- Tables (per-quadrant success, best supervision by quadrant)
- Supplementary materials (CPT validation, stuckness metrics)

---

## 12) Updated Compute Budget Summary

| Phase | Component | Runs/Episodes | GPU-hours | LLM cost |
|-------|-----------|---------------|-----------|----------|
| 1 | Leverage (A+B) | ~18,000 eps | - | - |
| 1 | CPT (n=2/condition) | ~30,000 eps | - | - |
| 1 | Teacher hints | ~3000 calls | - | $30 |
| 1B | Micro-training validation | 200 runs | 3 | - |
| 1B | Validation evaluation | ~2,200 eps | - | - |
| 2 | Model training | 14 runs | 28 | - |
| 2 | End-to-end eval | ~21,000 eps | - | - |
| 2 | Per-quadrant eval | ~22,400 eps | - | - |
| 2 | Retention eval | ~2,800 eps | - | - |
| 2 | Stuckness metrics | ~2,800 eps | - | - |
| 2 | Drift panel (optional) | ~3,000 eps | - | - |
| **Total** | | **~102,200 eps** | **~31** | **~$30** |

### GPU Requirements

| Scenario | Training Time | Feasible? |
|----------|---------------|-----------|
| 4 GPUs (parallel) | ~7 hours | ✓ Yes |
| 2 GPUs (parallel) | ~14 hours | ✓ Yes |
| 1 GPU (sequential) | ~28 hours | Tight but doable |

---

## Appendix: Risk mitigation summary

| Risk | Mitigation |
|------|------------|
| CPT doesn't predict training gains | Micro-training validation; fall back to L_local selection |
| Quadrant imbalance | Adjust thresholds; may need to merge Q2+Q3 |
| One supervision type dominates | Still informative; simplifies guidance |
| Cross-quadrant transfer confounds | Evaluate on held-out snapshots strictly by quadrant |
| Teacher hints are low quality | Explicit teacher model spec (GPT-4o); cache and validate |
| Training doesn't converge | Use proven hyperparameters; early stopping |
| Insufficient episodes for CPT | Single-stage n=2 is tractable; may increase if needed |

---

## Appendix: Key v7 → v8 Changes

| Component | v7 | v8 |
|-----------|----|----|
| Main experiment | ELP-guided selection (3 runs) | Per-quadrant training (14 runs) |
| Estimator naming | L_upper=A, L_local=B | L_local=A (single-step), L_upper=B (full) |
| Estimator C | Top-k student exploration | **Dropped** for sprint |
| CPT allocation | Two-stage adaptive | Single-stage (n=2/condition) |
| CPT validation | Correlation check | Micro-training (1-2 LoRA steps) |
| Feature tiers | Tier 1-4 | Tier 1+2 only for sprint |
| Drift panel | 300 snapshots, n=3 | 200 snapshots, n=1 |
| Task division | Generic | Kamal (training) / Chris (validation) |
