SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

cd "$SCRIPT_DIR" || exit 1

echo "[LoongFlow] Starting MoE FP8 block-scale contest task (Triton)"

python ../../../math_evolve_agent.py \
    --config task_config.yaml \
    --task-file task_prompt.txt \
    --initial-file moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json \
    --eval-file eval_program_modal.py \
    --log-level INFO
