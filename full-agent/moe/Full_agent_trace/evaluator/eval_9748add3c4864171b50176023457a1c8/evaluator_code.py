"""
LoongFlow-compatible CUDA Kernel Evaluator using Modal + flashinfer-bench.

Usage:
    1. Deploy Modal function (one-time):
       python -m modal deploy modal_eval_deploy.py

    2. Upload dataset (one-time):
       python -m modal run modal_eval_deploy.py::upload_dataset \
           --local-path /path/to/datasets/mlsys26-contest

    3. Use this file as --eval-file in LoongFlow:
       python ../../math_evolve_agent.py \
         --config task_config.yaml \
         --eval-file eval_program_modal.py \
         --initial-file task_definition.json \
         ...

Requires: modal package installed locally (pip install modal)
"""
import concurrent.futures
import json
import os
import sys
import time
import traceback
from typing import Dict, Any

import modal

# ============================================================================
# CONFIGURATION - modify these per task
# ============================================================================

# Modal app and function name (must match modal_eval_deploy_triton.py)
MODAL_APP_NAME = "loongflow-triton-eval"
MODAL_FUNCTION_NAME = "remote_eval_triton"

# Task ID - the problem definition name in the flashinfer-bench dataset
# This must match the "name" field in your JSON task definition
TASK_ID = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# Dataset name in the Modal Volume
DATASET_NAME = "mlsys26-contest"

# Number of parallel GPU shards for evaluation.
# Each shard gets a separate B200 container and processes workloads[shard_idx::NUM_SHARDS].
# Set to 1 to disable sharding (single container, sequential evaluation).
NUM_EVAL_SHARDS = 10

# Path to track best score across evaluations (enables NCU on new-best candidates)
_BEST_SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".best_score.json")


def _load_best_score() -> float:
    """Load the best score seen so far (0.0 if no record exists)."""
    try:
        with open(_BEST_SCORE_FILE) as f:
            return float(json.load(f).get("score", 0.0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return 0.0


def _save_best_score(score: float) -> None:
    """Persist the new best score."""
    with open(_BEST_SCORE_FILE, "w") as f:
        json.dump({"score": score}, f)


def _aggregate_shard_results(shard_results: list) -> dict:
    """Merge per-shard result dicts into a single aggregated result.

    Strategy (mirrors run_modal_multiple_gpus.py build_summary):
    - Any compilation error → return that error immediately.
    - Any correctness error → return that error immediately.
    - Otherwise: concatenate per_workload lists and recompute mean stats.
    """
    # Fast-fail on compilation/correctness errors.
    for r in shard_results:
        if r.get("error") and not r.get("compiled", True):
            return r  # compile error
    for r in shard_results:
        if r.get("error") and not r.get("correct", True):
            return r  # correctness error

    # Aggregate all per_workload entries from shards that produced results.
    all_per_workload = []
    task_id = None
    for r in shard_results:
        if r.get("error"):
            return r  # any other error
        task_id = task_id or r.get("task_id")
        all_per_workload.extend(r.get("per_workload", []))

    if not all_per_workload:
        return {"task_id": task_id, "error": "No evaluation results from any shard"}

    n = len(all_per_workload)
    return {
        "compiled": True,
        "correct": True,
        "task_id": task_id,
        "speedup": sum(w["speedup"] for w in all_per_workload) / n,
        "latency_ms": sum(w["latency_ms"] for w in all_per_workload) / n,
        "stats": {
            "reference_latency_ms": sum(w["reference_latency_ms"] for w in all_per_workload) / n,
            "max_relative_error": max(w["max_relative_error"] for w in all_per_workload),
            "max_absolute_error": max(w["max_absolute_error"] for w in all_per_workload),
            "total_workloads": n,
        },
        "per_workload": all_per_workload,
    }


# ============================================================================
# Core evaluate() function - called by LoongFlow framework
# ============================================================================

def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate a kernel via Modal remote GPU + flashinfer-bench.

    Conforms to the LoongFlow evaluate() interface:
    - Input: program_path (str) - path to file with kernel code (run() function)
    - Output: dict with {status, summary, score, metrics, artifacts}

    Also handles JSON task definitions gracefully (returns score=0.0).
    """
    start = time.time()

    try:
        with open(program_path, 'r') as f:
            content = f.read().strip()

        # If the file is a JSON task definition (not kernel code), skip evaluation
        if content.startswith('{'):
            try:
                task_def = json.loads(content)
                task_name = task_def.get('name', 'unknown')
                return {
                    "status": "success",
                    "summary": f"Task definition loaded: {task_name} (no kernel to evaluate)",
                    "score": 0.0,
                    "metrics": {
                        "correctness": 0.0,
                        "speedup": 0.0,
                        "exec_time_ms": 0.0,
                        "baseline_time_ms": 0.0,
                        "eval_time": time.time() - start,
                    },
                    "artifacts": {
                        "task_name": task_name,
                        "eval_backend": "modal (skipped - task definition)",
                    },
                }
            except json.JSONDecodeError:
                pass  # Not valid JSON, treat as kernel code

        kernel_code = content
        print(f"Starting Modal evaluation: {program_path}")
        print(f"  task_id={TASK_ID}, dataset={DATASET_NAME}, shards={NUM_EVAL_SHARDS}")

        # Look up the deployed Modal function
        try:
            remote_fn = modal.Function.from_name(
                MODAL_APP_NAME, MODAL_FUNCTION_NAME
            )
        except Exception as e:
            return _make_error_result(
                f"Cannot find Modal function '{MODAL_APP_NAME}/{MODAL_FUNCTION_NAME}'. "
                f"Run 'python -m modal deploy modal_eval_deploy.py' first. Error: {e}",
                time.time() - start,
                failure_stage="modal_lookup",
            )

        # ── Parallel shard dispatch ───────────────────────────────────────────
        # Spawn NUM_EVAL_SHARDS Modal containers concurrently.  Each container
        # processes workloads[shard_idx::NUM_EVAL_SHARDS] independently.
        # This mirrors run_modal_multiple_gpus.py's ThreadPoolExecutor pattern.
        print(f"-> Dispatching {NUM_EVAL_SHARDS} parallel shard(s) to Modal B200 GPUs...")

        def _call_shard(shard_idx: int) -> dict:
            return remote_fn.remote(
                kernel_code=kernel_code,
                task_id=TASK_ID,
                dataset_name=DATASET_NAME,
                shard_idx=shard_idx,
                num_shards=NUM_EVAL_SHARDS,
            )

        shard_results = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_EVAL_SHARDS) as executor:
                futures = {executor.submit(_call_shard, i): i for i in range(NUM_EVAL_SHARDS)}
                for fut in concurrent.futures.as_completed(futures):
                    shard_idx = futures[fut]
                    try:
                        shard_results.append(fut.result())
                        print(f"  Shard {shard_idx}/{NUM_EVAL_SHARDS} done")
                    except Exception as e:
                        shard_results.append({
                            "task_id": TASK_ID,
                            "error": f"Shard {shard_idx} failed: {e}",
                            "compiled": False,
                            "correct": False,
                        })
                        print(f"  Shard {shard_idx}/{NUM_EVAL_SHARDS} error: {e}", file=sys.stderr)
        except Exception as e:
            return _make_error_result(
                f"Modal parallel dispatch failed: {e}",
                time.time() - start,
                failure_stage="modal_remote_call",
            )

        eval_time = time.time() - start
        print(f"<- All shards returned in {eval_time:.2f}s")

        raw_result = _aggregate_shard_results(shard_results)
        print(f"[Aggregated result] {json.dumps(raw_result, indent=2, default=str)}")

        converted = _convert_result(raw_result, eval_time)

        # ── NCU profiling for new-best candidates ────────────────────────────
        # Run NCU only when this result beats the historical best score.
        if converted["score"] > 0:
            prev_best = _load_best_score()
            if converted["score"] > prev_best:
                print(
                    f"New best score {converted['score']:.4f} > {prev_best:.4f}. "
                    "Running NCU profiling..."
                )
                try:
                    ncu_raw = remote_fn.remote(
                        kernel_code=kernel_code,
                        task_id=TASK_ID,
                        dataset_name=DATASET_NAME,
                        include_ncu=True,
                        ncu_workload_count=1,
                        ncu_set="full",
                        shard_idx=0,
                        num_shards=1,
                    )
                    ncu_profile = ncu_raw.get("ncu_profile")
                    if ncu_profile:
                        converted["artifacts"]["ncu_profile"] = ncu_profile
                        print("NCU profile attached to result artifacts.")
                    else:
                        print("NCU call returned no profile data.")
                except Exception as ncu_err:
                    print(f"NCU profiling failed (non-fatal): {ncu_err}", file=sys.stderr)
                _save_best_score(converted["score"])

        return converted

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc()
        return {
            "status": "execution_failed",
            "summary": error_msg,
            "score": 0.0,
            "metrics": {"correctness": 0.0, "speedup": 0.0, "eval_time": 0.0},
            "artifacts": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "failure_stage": "evaluation_framework",
            },
        }


# ============================================================================
# Result conversion
# ============================================================================

def _convert_result(modal_result: dict, eval_time: float) -> Dict[str, Any]:
    """Convert flashinfer-bench result dict to LoongFlow format."""
    compiled = modal_result.get("compiled", False)
    correct = modal_result.get("correct", False)
    speedup = modal_result.get("speedup", 0.0)
    latency_ms = modal_result.get("latency_ms", 0.0)
    task_id = modal_result.get("task_id", TASK_ID)
    error = modal_result.get("error")
    stats = modal_result.get("stats", {})

    ref_latency_ms = stats.get("reference_latency_ms", 0.0) if stats else 0.0
    correctness_score = 1.0 if (compiled and correct) else 0.0
    combined_score = float(speedup) * 0.1 if speedup and speedup > 0 and correctness_score > 0 else 0.0

    # Determine status and summary
    if error and not compiled:
        status = "execution_failed"
        summary = f"Compilation failed: {error}"
    elif error and not correct:
        status = "validation_failed"
        summary = f"Validation failed: {error}"
    elif error:
        status = "execution_failed"
        summary = f"Evaluation error: {error}"
    elif correct and speedup and speedup >= 1.0:
        status = "success"
        summary = (
            f"Optimization successful! Speedup: {speedup:.2f}x "
            f"(ref: {ref_latency_ms:.3f}ms, kernel: {latency_ms:.3f}ms)"
        )
    elif correct:
        status = "success"
        summary = f"Kernel correct but slower. Speedup: {speedup:.2f}x, Latency: {latency_ms:.3f}ms"
    else:
        status = "execution_failed"
        summary = "Unknown evaluation result"

    # Build artifacts
    artifacts = {
        "eval_backend": "modal+flashinfer-bench",
        "task_id": task_id,
        "execution_time": f"{eval_time:.2f}s",
    }
    if latency_ms:
        artifacts["kernel_latency_ms"] = f"{latency_ms:.3f}ms"
    if ref_latency_ms:
        artifacts["reference_latency_ms"] = f"{ref_latency_ms:.3f}ms"
    if speedup:
        artifacts["speedup"] = f"{speedup:.2f}x"
    if stats:
        artifacts["bench_stats"] = stats
    if compiled and correct:
        artifacts["correctness_check"] = "Passed"
    elif error:
        artifacts["error"] = error

    print(
        f"Evaluation complete: compiled={compiled}, correct={correct}, "
        f"speedup={speedup:.2f}x, score={combined_score:.4f}"
    )

    return {
        "status": status,
        "summary": summary,
        "score": float(combined_score),
        "metrics": {
            "correctness": float(correctness_score),
            "speedup": float(speedup) if speedup else 0.0,
            "exec_time_ms": float(latency_ms) if latency_ms else 0.0,
            "baseline_time_ms": float(ref_latency_ms),
            "compile_time": 0.0,
            "eval_time": float(eval_time),
        },
        "artifacts": artifacts,
    }


def _make_error_result(
    error: str, eval_time: float, failure_stage: str = "unknown"
) -> Dict[str, Any]:
    """Build a LoongFlow-compatible error result."""
    print(f"Error: {error}", file=sys.stderr)
    return {
        "status": "execution_failed",
        "summary": f"Evaluation failed: {error}",
        "score": 0.0,
        "metrics": {
            "correctness": 0.0,
            "speedup": 0.0,
            "exec_time_ms": 0.0,
            "baseline_time_ms": 0.0,
            "compile_time": 0.0,
            "eval_time": float(eval_time),
        },
        "artifacts": {
            "error": error,
            "failure_stage": failure_stage,
            "eval_backend": "modal+flashinfer-bench",
        },
    }


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "./test.py"
    if os.path.exists(test_file):
        result = evaluate(test_file)
        print(f"\nStatus: {result['status']}")
        print(f"Summary: {result['summary']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Metrics: {result['metrics']}")
    else:
        print(f"File not found: {test_file}")

