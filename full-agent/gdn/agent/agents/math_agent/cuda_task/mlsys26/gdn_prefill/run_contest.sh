SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

cd "$SCRIPT_DIR" || exit 1

echo "[LoongFlow] Starting GDN prefill contest task"

python ../../../math_evolve_agent.py \
    --config task_config.yaml \
    --task-file task_prompt_new.txt \
    --initial-file gdn_prefill_qk4_v8_d128_k_last.json \
    --eval-file eval_program_modal.py \
    --log-level INFO
