#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Repo: $ROOT_DIR"
echo

echo "[1/3] Running cheap suite"
pytest -q

echo

echo "[2/3] Opt-in: local Qwen3-8B decode smoke"
if [[ "${RUN_QWEN_SMOKE:-}" == "1" ]]; then
  pytest -q tests/integration/test_qwen3_8b_decode_action_smoke.py
else
  echo "SKIP (set RUN_QWEN_SMOKE=1 to enable)"
fi

echo

echo "[3/3] Opt-in: slow HF download + LoRA SFT smoke"
if [[ "${RUN_SLOW_TESTS:-}" == "1" ]]; then
  pytest -q tests/integration/test_train_one_run_slow.py
else
  echo "SKIP (set RUN_SLOW_TESTS=1 to enable)"
fi
