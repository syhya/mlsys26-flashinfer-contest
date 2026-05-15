#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY=sk-... before running.}"
: "${LLM_BASE_URL:?LLM_BASE_URL not set. export LLM_BASE_URL=https://api.chatanywhere.tech/v1 before running.}"
export LLM_API_KEY LLM_BASE_URL

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/agent"
TASK_DIR="$PROJECT_ROOT/agents/math_agent/cuda_task/mlsys26/dsa_sparse_attn"
RUN_DIR="$SCRIPT_DIR/dsa_sparse_attn_run"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

RENDERED_CONFIG="$(mktemp -t task_config.XXXXXX).yaml"
trap 'rm -f "$RENDERED_CONFIG"' EXIT
envsubst '${LLM_API_KEY} ${LLM_BASE_URL}' < "$TASK_DIR/task_config.yaml" > "$RENDERED_CONFIG"

echo "[LoongFlow] Starting DSA sparse attention contest task"

"$PYTHON_BIN" "$PROJECT_ROOT/agents/math_agent/math_evolve_agent.py" \
    --config       "$RENDERED_CONFIG" \
    --task-file    "$TASK_DIR/task_prompt.txt" \
    --initial-file "$TASK_DIR/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json" \
    --eval-file    "$TASK_DIR/eval_program_modal.py" \
    --log-level INFO
