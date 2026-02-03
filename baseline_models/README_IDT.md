# Intervention-Defined Teachability (IDT)

IDT changes the unit from "predict mistake step" to **find minimal patches (tiny interventions) that reliably convert failures into successes** under a small attempt budget, then train on those patches.

## What is IDT?

- **Input**: Failed agent trajectories (e.g. from WebShop).
- **Output**: A dataset of `(context_at_t, patch δ, teachable label, recovery probs R1/R3/R5)`.
- **Goal**: Search for a minimal patch at some step `t` that makes recovery likely; use this for teachability prediction and training.

## Architecture

- **`idt/types.py`**: `Trajectory`, `StepContext`, `PatchSearchResult`; load/save JSONL.
- **`idt/env_adapter.py`**: `WebShopEnvAdapter`, `ToyEnvAdapter`: `reset(task)`, `step(action)`, `replay(actions_prefix)`.
- **`idt/agent_adapter.py`**: `AgentAdapter`: `act()`, `propose_actions(top_n)` (with fallback).
- **`idt/patches.py`**: `ReplaceActionPatch`, `InsertActionPatch`, `EditQueryPatch`.
- **`idt/propose.py`**: `HeuristicPatchProposer`: replace/insert/edit_query candidates.
- **`idt/verify.py`**: `estimate_recovery_probability(env, agent, traj, step_t, patch, attempt_budget, ...)`.
- **`idt/search.py`**: `search_minimal_patch(...)`: minimal patch search with K=1,3,5 schedule.
- **`idt/dataset.py`**: Build/save/load IDT JSONL dataset.
- **`idt/train_teachability.py`**: Baseline teachability predictor (logreg/MLP + features).
- **`idt/eval.py`**: Aggregate metrics + teachability landscape plots (matplotlib).
- **`idt/toy_env/`**: Deterministic toy env for fast, deterministic unit tests.

## How to run patch search on failures

Run from `baseline_models/` with `PYTHONPATH=.` (or from repo root with `PYTHONPATH=baseline_models`).

```bash
# Toy env (no WebShop)
python -m idt.scripts.run_patch_search --toy --num 1

# Real failures file (uses toy env adapter by default; plug WebShop env/agent for full pipeline)
python -m idt.scripts.run_patch_search --failures_path simulation/failures_3k.json --num 1
```

## How to build the dataset

```bash
# Toy mode: N=20 toy failure trajectories → idt_dataset.jsonl
python -m idt.scripts.build_idt_dataset --toy --num_trajectories 20 --out_path idt_dataset.jsonl --K 5 --threshold 0.6

# From failures file (loads trajectories; env/agent default to toy)
python -m idt.scripts.build_idt_dataset --failures_path path/to/failures.json --num_trajectories 200 --out_path idt_dataset.jsonl --K 5 --threshold 0.6
```

## How to train the teachability predictor

```bash
python -m idt.scripts.train_teachability_cli --dataset_path idt_dataset.jsonl --model_out teachability_model.joblib
```

## How to run evaluation plots

```bash
python -m idt.scripts.eval_landscape --dataset_path idt_dataset.jsonl --out_dir idt_eval_out
```

Outputs: `idt_eval_out/metrics.json`, `idt_eval_out/landscape_by_k.png`, `idt_eval_out/landscape_by_patch_type.png`.

## How to run tests

```bash
# From baseline_models/ with PYTHONPATH=.
cd baseline_models
export PYTHONPATH=.
pytest idt/tests/test_toy_env.py idt/tests/test_patches.py idt/tests/test_replay.py idt/tests/test_verifier.py idt/tests/test_search_min_patch.py -v
```

---

## RUNBOOK (exact commands)

All commands assume you are in **baseline_models/** with `PYTHONPATH=.` (or in repo root with `PYTHONPATH=baseline_models`).

### 1) Run unit tests

```bash
cd baseline_models
export PYTHONPATH=.
pytest idt/tests/test_toy_env.py idt/tests/test_patches.py idt/tests/test_replay.py idt/tests/test_verifier.py idt/tests/test_search_min_patch.py -v
```

### 2) Build dataset on N=20 failures (toy)

```bash
cd baseline_models
export PYTHONPATH=.
python -m idt.scripts.build_idt_dataset --toy --num_trajectories 20 --out_path idt_dataset.jsonl --K 5 --threshold 0.6
```

### 3) Train teachability predictor

```bash
cd baseline_models
export PYTHONPATH=.
python -m idt.scripts.train_teachability --dataset_path idt_dataset.jsonl --model_out teachability_model.joblib
```

### 4) Generate evaluation plots

```bash
cd baseline_models
export PYTHONPATH=.
python -m idt.scripts.eval_landscape --dataset_path idt_dataset.jsonl --out_dir idt_eval_out
```

---

## ARC OOD (out-of-distribution cluster)

On ARC OOD, use **toy mode** so you don't depend on WebShop env/Java/Lucene. IDT lives under `baseline_models/`; no changes to `web_agent_site` are required.

### Setup (once)

```bash
# Clone repo
cd /path/to/WebShop

# Create env (conda or venv) and install (from repo root)
conda create -n idt python=3.10 -y
conda activate idt
pip install -r requirements.txt

# Optional: if you have failure data on ARC, copy it (e.g. failures.json)
# Otherwise use --toy for all steps.
```

### Run (from baseline_models/)

```bash
cd /path/to/WebShop/baseline_models
export PYTHONPATH=.

# 1) Unit tests
pytest idt/tests/test_toy_env.py idt/tests/test_patches.py idt/tests/test_replay.py idt/tests/test_verifier.py idt/tests/test_search_min_patch.py -v

# 2) Build IDT dataset (toy, N=20)
python -m idt.scripts.build_idt_dataset --toy --num_trajectories 20 --out_path idt_dataset.jsonl --K 5 --threshold 0.6

# 3) Train teachability predictor
python -m idt.scripts.train_teachability --dataset_path idt_dataset.jsonl --model_out teachability_model.joblib

# 4) Evaluation plots
python -m idt.scripts.eval_landscape --dataset_path idt_dataset.jsonl --out_dir idt_eval_out
```

### One-shot pipeline script

```bash
cd /path/to/WebShop/baseline_models
export PYTHONPATH=.
bash run_all_idt.sh
```

### With real failure data on ARC

If you have a `failures.json` (or JSONL) on ARC and want to run on it **without** the full WebShop env (toy env/agent will be used; recovery stats are not WebShop-meaningful):

```bash
cd /path/to/WebShop/baseline_models
export PYTHONPATH=.
python -m idt.scripts.build_idt_dataset \
  --failures_path /path/to/failures.json \
  --num_trajectories 100 \
  --out_path idt_dataset.jsonl \
  --toy
```

To use the **real** WebShop env and agent on ARC you need the baseline_models WebEnv, agent checkpoint, and a working search backend (Java/Lucene or a BM25 fallback); that setup is environment-specific.

---

## Trajectory format

Failed trajectories can be:

- **JSON array**: `[{"task_id": 0, "goal": "...", "steps": [{"step": 0, "observation": "...", "action_taken": "search[...]", "valid_actions": [...]}, ...]}, ...]`
- **JSONL**: One such object per line.

`idt.types.load_trajectories(path)` supports both. If you have another format, add a converter and then `trajectory_from_failure_dict(d)` to get `Trajectory`.

## Running on real WebShop failures

To run the pipeline on **real failures** with the **real WebShop env and agent**:

1. **From repo root**, run:
   ```bash
   PYTHONPATH=. python -m idt.scripts.build_idt_dataset \
     --failures_path baseline_models/trajectories_500_failures/failures.json \
     --num_trajectories 200 \
     --out_path idt_dataset.jsonl
   ```
   The default `--model_path` is `baseline_models/ckpts/web_click/epoch_9/model.pth` (repo root).
2. **You need**:
   - An agent checkpoint at `--model_path` (default: `baseline_models/ckpts/web_click/epoch_9/model.pth`).
   - WebShop dependencies: `baseline_models` env (WebEnv) and search engine. If Java/Lucene (Pyserini) fails (e.g. `Module jdk.incubator.vector not found`), fix your Java setup or run with `--toy` to use the toy env (results then use toy actions, not WebShop).
3. **Fallback**: If WebShop env/agent setup fails (missing model, Java/Lucene error, etc.), the script **falls back to the toy env and agent** and continues; the run then uses toy semantics (not meaningful for WebShop). Use `--toy` explicitly to force toy mode and skip WebShop setup.

## Integration with WebShop

- **Env**: For real WebShop we use `WebShopWebEnvAdapter` wrapping `baseline_models.env.WebEnv` (so `info['valid']` is available). It exposes `reset(task_id)`, `step(action)`, `replay(task_id, actions_prefix)`; success = done and reward ≥ 10 (WebEnv scales reward by 10).
- **Agent**: `WebShopAgentAdapter` wraps the EEF `Agent` from `baseline_models.eef_detailed_with_diagnosis_parallel` and calls `get_action(obs, info, method)`. Use `--model_path` to point to the BERT checkpoint.

## Dependencies

- Python 3.9+
- For teachability training: `scikit-learn`
- For plots: `matplotlib`
- For full WebShop pipeline: existing WebShop env and agent (see repo).

See `requirements.txt` or project root `pyproject.toml` for versions.
