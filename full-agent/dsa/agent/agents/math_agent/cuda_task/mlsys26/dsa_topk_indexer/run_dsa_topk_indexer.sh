#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY=sk-... before running.}"
: "${LLM_BASE_URL:?LLM_BASE_URL not set. export LLM_BASE_URL=https://api.chatanywhere.tech/v1 before running.}"
export LLM_API_KEY LLM_BASE_URL

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$SCRIPT_DIR" || exit 1

RENDERED_CONFIG="$(mktemp -t task_config.XXXXXX).yaml"
trap 'rm -f "$RENDERED_CONFIG"' EXIT
envsubst '${LLM_API_KEY} ${LLM_BASE_URL}' < task_config.yaml > "$RENDERED_CONFIG"

echo "[LoongFlow] Starting contest task"

"$PYTHON_BIN" ../../../math_evolve_agent.py \
    --config "$RENDERED_CONFIG" \
    --task-file task_prompt.txt \
    --initial-file dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json \
    --eval-file eval_program_modal.py \
    --log-level INFO
