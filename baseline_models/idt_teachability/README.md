# IDT Patch Teachability (WebShop/EEF)

This folder adds an **action-patching** layer on top of your existing EEF simulator.

## What it does (in one sentence)

Given a failed trajectory and a candidate mistake step, we **restart from that step**, optionally **force a different first action**, then let the same student policy run — and measure recovery.

## Files

- `patch_simulator.py`  
  Replays a prefix then runs rollouts, with optional forced first action.

- `patchers.py`  
  Strategies to propose alternative patch actions:
  - `agent_topk`: propose top-k actions under the student's policy
  - `random`: random actions
  - `diagnosis_text`: parse `click[...]` or `search[...]` from the diagnosis model response (if present)

- `run_idt_patch.py`  
  CLI pipeline to run patch-teachability evaluation and export:
  - `patch_results_*.json`
  - `patch_stats_*.json`
  - `patch_training_samples_*.json` (patch supervision)

- `tests/`  
  Lightweight unit tests with a mock environment (no WebShop required).

## Running

### Run from repo root

This repo uses a `baseline_models/` layout. The recommended invocation is to run
the script via its path so it can find `baseline_models/ckpts` and
`baseline_models/simulation/...`.

```bash
python baseline_models/idt_teachability/run_idt_patch.py \
  --failure_data baseline_models/failures.json \
  --strategy diagnosis --M 3 \
  --diagnosis_model_path baseline_models/simulation/Qwen2.5/qwen25_instruct_v1 \
  --diagnosis_base_model Qwen/Qwen2.5-3B-Instruct \
  --model_path baseline_models/ckpts/web_click/epoch_9/model.pth \
  --patcher agent_topk --patch_k 5 \
  --baseline_attempts 3 --patch_attempts 1 \
  --max_steps 50 \
  --output_dir baseline_models/idt_teachability/outputs
```

### Run unit tests

```bash
pytest -q baseline_models/idt_teachability/tests
```

## Notes

- Reward scaling: consistent with your EEF simulator, the code treats `reward==10.0` as success and reports `reward*10` (so success → 100).
- If you want to allow arbitrary `search[query]` strings (not only those in `info['valid']`), add `--allow_any_search`.
