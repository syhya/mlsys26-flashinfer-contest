#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY=sk-... before running.}"
export LLM_API_KEY

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/agent"
TASK_DIR="$PROJECT_ROOT/agents/math_agent/cuda_task/mlsys26/gdn_decode"
RUN_DIR="$SCRIPT_DIR/gdn_decode_run"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

RENDERED_CONFIG="$(mktemp -t task_config.XXXXXX).yaml"
trap 'rm -f "$RENDERED_CONFIG"' EXIT
envsubst '${LLM_API_KEY}' < "$TASK_DIR/task_config.yaml" > "$RENDERED_CONFIG"

echo "[LoongFlow] Starting GDN decode contest task"

"$PYTHON_BIN" "$PROJECT_ROOT/agents/math_agent/math_evolve_agent.py" \
    --config       "$RENDERED_CONFIG" \
    --task-file    "$TASK_DIR/task_prompt.txt" \
    --initial-file "$TASK_DIR/gdn_decode_qk4_v8_d128_k_last.json" \
    --eval-file    "$TASK_DIR/eval_program_modal.py" \
    --log-level INFO
