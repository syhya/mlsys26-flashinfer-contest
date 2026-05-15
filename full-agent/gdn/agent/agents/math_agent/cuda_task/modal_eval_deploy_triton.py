"""
Modal remote GPU evaluation service for LoongFlow — Triton kernel variant.

Aligned with the official contest evaluation environment:
  - Image:   flashinfer/flashinfer-ci-cu132:latest  (official contest image)
  - Package: language=triton, binding=torch          (Triton submission format)
  - Params:  warmup=10, iterations=50, trials=3      (official benchmark params)
  - Runner:  use_isolated_runner=True                (matches leaderboard)

Deploy once:
    python -m modal deploy modal_eval_deploy_triton.py

Dataset upload is shared with the CUDA deploy — if already uploaded, skip:
    python -m modal run modal_eval_deploy_triton.py::upload_dataset \
        --local-path /path/to/datasets/mlsys26-contest
"""

import os

import modal

MODAL_APP_NAME = "loongflow-triton-eval"
VOLUME_NAME = "flashinfer-trace-chenyu"
DATASET_PATH = "/data"

# Official contest image — matches the leaderboard evaluation environment
OFFICIAL_IMAGE = "flashinfer/flashinfer-ci-cu132:20260401-2c675fb"
OFFICIAL_FLASHINFER_BENCH_GIT = (
    "git+https://github.com/flashinfer-ai/flashinfer-bench.git"
    "@80f40d45968c65840d05872516befd9691ec9fd8"
)
FLASHINFER_BENCH_PACKAGE = os.environ.get("FLASHINFER_BENCH_PACKAGE", OFFICIAL_FLASHINFER_BENCH_GIT)

# Official benchmark parameters per definition
OFFICIAL_DEFAULT = dict(warmup_runs=10, iterations=50, num_trials=3)
OFFICIAL_CONFIGS = {
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": dict(
        warmup_runs=10, iterations=50, num_trials=3,
        atol=1.0, rtol=0.3, required_matched_ratio=0.9,
    ),
}

# Triton entry-point table: task_id prefix → "kernel.py::func_name"
_TASK_ENTRY_POINTS_TRITON = {
    "dsa": "kernel.py::run",
    "gdn": "kernel.py::run",
    "moe": "kernel.py::run",
}

app = modal.App(MODAL_APP_NAME)

image = (
    modal.Image.from_registry(OFFICIAL_IMAGE, add_python=None)
    .pip_install(FLASHINFER_BENCH_PACKAGE, "cupti-python")
)

dataset_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    volumes={DATASET_PATH: dataset_vol},
    timeout=600,
    serialized=True,
)
def diagnose_bench(
    kernel_code: str,
    task_id: str,
    dataset_name: str = "mlsys26-contest",
) -> dict:
    """Diagnostic: manually step through the runner to find the exact failure."""
    import uuid
    import traceback as _tb
    import inspect

    diag = {"version": "v7"}

    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import (
        BuildSpec, EvaluationStatus, Solution, SourceFile, TraceSet,
    )

    dataset_root = os.path.join(DATASET_PATH, dataset_name)
    trace_set = TraceSet.from_path(dataset_root)
    definition = trace_set.definitions.get(task_id)
    workloads = trace_set.workloads.get(task_id, [])

    entry_point = _TASK_ENTRY_POINTS_TRITON.get(task_id.split("_")[0], "kernel.py::run")
    solution = Solution(
        name=f"diag_{uuid.uuid4().hex[:8]}",
        definition=task_id,
        author="diag",
        spec=BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point=entry_point,
            dependencies=[],
            binding="torch",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content=kernel_code)],
    )

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={task_id: definition},
        solutions={task_id: [solution]},
        workloads={task_id: workloads[:1]},  # just 1 workload
        traces={task_id: []},
    )

    config = BenchmarkConfig(
        warmup_runs=1, iterations=1, num_trials=1,
        use_isolated_runner=True,
        atol=1.0, rtol=0.3,
        timeout_seconds=300,
    )

    benchmark = Benchmark(bench_ts, config)

    # Get runner and try manual evaluation
    runner = benchmark._runner
    diag["runner_type"] = str(type(runner))

    # Get the source of run_all to understand the full flow
    try:
        src = inspect.getsource(Benchmark.run_all)
        diag["run_all_source_len"] = len(src)
        # Get the SECOND half of the source
        diag["run_all_source_part2"] = src[2000:]
    except Exception as e:
        diag["run_all_source_error"] = str(e)

    # Try to manually evaluate one workload
    wl_trace = workloads[0]
    wl = wl_trace.workload
    diag["workload_type"] = str(type(wl))
    diag["workload_uuid"] = str(getattr(wl, 'uuid', 'N/A'))

    # Try runner.evaluate directly
    try:
        eval_src = inspect.getsource(type(runner).evaluate)
        diag["runner_evaluate_source"] = eval_src[:4000]
    except Exception as e:
        diag["runner_evaluate_error"] = str(e)

    # Try runner.run_workload directly
    try:
        results = runner.run_workload(
            definition, wl, [solution], config, trace_set.root
        )
        diag["manual_run_result_type"] = str(type(results))
        diag["manual_run_keys"] = list(results.keys()) if isinstance(results, dict) else "not dict"
        for k, v in (results or {}).items():
            diag[f"manual_run_{k}"] = str(v)[:1000]
    except Exception as e:
        diag["manual_run_error"] = str(e)
        diag["manual_run_traceback"] = _tb.format_exc()[:3000]
        diag["manual_run_error_type"] = type(e).__name__

    benchmark.close()
    return diag


@app.function(
    image=image,
    gpu="B200",
    volumes={DATASET_PATH: dataset_vol},
    timeout=3600,
    serialized=True,
)
def remote_eval_triton(
    kernel_code: str,
    task_id: str,
    dataset_name: str = "mlsys26-contest",
    include_ncu: bool = False,
    ncu_workload_count: int = 1,
    ncu_set: str = "full",
    ncu_page: str = "details",
    ncu_timeout: int = 900,
    ncu_kernel_name: str = None,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> dict:
    """
    [v3-debug] Evaluate a Triton kernel on Modal B200 using the official flashinfer-bench pipeline.

    kernel_code must be a Python file that exposes the forward function, e.g.:
        def moe_forward(hidden_states, w1, w2, topk_weights, topk_ids,
                        w1_scale, w2_scale, a1_scale, a2_scale) -> torch.Tensor:
            ...
    """
    import uuid
    import tempfile
    import subprocess
    from pathlib import Path

    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import (
        BuildSpec,
        EvaluationStatus,
        Solution,
        SourceFile,
        TraceSet,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────────
    dataset_root = os.path.join(DATASET_PATH, dataset_name)
    trace_set = TraceSet.from_path(dataset_root)

    definition = trace_set.definitions.get(task_id)
    if definition is None:
        return {"task_id": task_id, "error": f"[v3] Definition '{task_id}' not found in dataset"}

    workloads = trace_set.workloads.get(task_id, [])
    if not workloads:
        return {"task_id": task_id, "error": f"[v3] No workloads found for '{task_id}'"}

    # Shard workloads for parallel multi-GPU evaluation
    if num_shards > 1:
        workloads = workloads[shard_idx::num_shards]
        if not workloads:
            return {
                "task_id": task_id, "compiled": True, "correct": True,
                "speedup": 0.0, "latency_ms": 0.0, "per_workload": [],
                "stats": {"reference_latency_ms": 0.0, "max_relative_error": 0.0,
                          "max_absolute_error": 0.0, "total_workloads": 0},
            }

    # ── Build Triton Solution ─────────────────────────────────────────────────────
    task_prefix = task_id.split("_")[0]
    entry_point = _TASK_ENTRY_POINTS_TRITON.get(task_prefix, f"kernel.py::{task_prefix}_forward")

    solution_name = f"agent_{uuid.uuid4().hex[:8]}"
    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="agent",
        spec=BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point=entry_point,
            dependencies=[],
            binding="torch",
            destination_passing_style=False,
        ),
        sources=[
            SourceFile(path="kernel.py", content=kernel_code),
        ],
    )

    # Filtered TraceSet with only this task + solution
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={task_id: definition},
        solutions={task_id: [solution]},
        workloads={task_id: workloads},
        traces={task_id: []},
    )

    # Official benchmark parameters
    bench_kwargs = dict(OFFICIAL_DEFAULT)
    bench_kwargs.update(OFFICIAL_CONFIGS.get(task_id, {}))
    config_kw = dict(
        warmup_runs=bench_kwargs["warmup_runs"],
        iterations=bench_kwargs["iterations"],
        num_trials=bench_kwargs["num_trials"],
        use_isolated_runner=True,
        atol=bench_kwargs.get("atol", 0.01),
        rtol=bench_kwargs.get("rtol", 0.01),
    )
    if bench_kwargs.get("required_matched_ratio", 0.0) > 0:
        config_kw["required_matched_ratio"] = bench_kwargs["required_matched_ratio"]
    config = BenchmarkConfig(**config_kw)

    benchmark = Benchmark(bench_trace_set, config)
    try:
        result_ts = benchmark.run_all(dump_traces=False)
    except Exception as bench_exc:
        import traceback as _tb
        benchmark.close()
        return {
            "task_id": task_id,
            "compiled": False,
            "correct": False,
            "error": f"Benchmark execution failed: {bench_exc}",
            "traceback": _tb.format_exc(),
        }
    finally:
        benchmark.close()

    # ── Debug: collect result_ts structure info ─────────────────────────────────
    _debug = {}
    _debug["result_ts_type"] = str(type(result_ts))
    _debug["traces_keys"] = list(result_ts.traces.keys()) if result_ts.traces else []
    _debug["definitions_keys"] = list(result_ts.definitions.keys()) if result_ts.definitions else []
    _debug["solutions_keys"] = list(result_ts.solutions.keys()) if result_ts.solutions else []
    _debug["workloads_keys"] = list(result_ts.workloads.keys()) if result_ts.workloads else []
    _trace_debug = []
    for _tk, _tlist in (result_ts.traces or {}).items():
        for _i, _tr in enumerate(_tlist[:5]):
            _ev = getattr(_tr, 'evaluation', None)
            _td = {"task": _tk, "idx": _i, "has_eval": _ev is not None}
            if _ev:
                _td["status"] = str(getattr(_ev, 'status', None))
                _td["log"] = str(getattr(_ev, 'log', ''))[:500]
            _trace_debug.append(_td)
    _debug["trace_details"] = _trace_debug
    for _wk, _wlist in (result_ts.workloads or {}).items():
        _debug[f"workloads_{_wk}_count"] = len(_wlist)
    for _sk, _slist in (result_ts.solutions or {}).items():
        _debug[f"solutions_{_sk}_count"] = len(_slist)

    traces = result_ts.traces.get(task_id, [])

    # ── Result parsing helpers ────────────────────────────────────────────────────
    def _maybe_dump(model_like):
        if model_like is None:
            return None
        try:
            if hasattr(model_like, "model_dump"):
                return model_like.model_dump()
        except Exception:
            pass
        try:
            return dict(model_like)
        except Exception:
            return str(model_like)

    def _trace_axes(trace):
        workload = getattr(trace, "workload", None)
        inner_workload = getattr(workload, "workload", None) or workload
        axes_raw = getattr(inner_workload, "axes", {}) or {}
        return _maybe_dump(axes_raw) or {}

    def _error_detail(trace, ev):
        detail = {
            "status": getattr(ev.status, "value", str(ev.status)),
            "log": ev.log,
            "axes": _trace_axes(trace),
        }
        correctness = getattr(ev, "correctness", None)
        correctness_dump = _maybe_dump(correctness)
        if correctness_dump is not None:
            detail["correctness"] = correctness_dump
        performance = getattr(ev, "performance", None)
        performance_dump = _maybe_dump(performance)
        if performance_dump is not None:
            detail["performance"] = performance_dump
        return detail

    error_statuses = {
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.TIMEOUT,
    }
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status in error_statuses:
            return {
                "compiled": ev.status != EvaluationStatus.COMPILE_ERROR,
                "correct": False,
                "task_id": task_id,
                "error": f"{ev.status.value}: {ev.log}",
                "error_detail": _error_detail(trace, ev),
            }

    # ── NCU profiling (optional) ──────────────────────────────────────────────────
    def _run_ncu_profile(passing_traces, dataset_root_, solution_):
        import sys as _sys
        rows = []
        for trace in passing_traces[:ncu_workload_count]:
            workload = getattr(trace, "workload", None)
            wuuid = str(getattr(workload, "uuid", uuid.uuid4()))
            inner_workload = getattr(workload, "workload", None) or workload
            axes_raw = getattr(inner_workload, "axes", {}) or {}
            try:
                axes = axes_raw.model_dump() if hasattr(axes_raw, "model_dump") else dict(axes_raw)
            except Exception:
                axes = {}

            ncu_output = ""
            with tempfile.TemporaryDirectory(prefix="fib_ncu_") as build_dir:
                build_path = Path(build_dir)
                defn = trace_set.definitions.get(task_id)
                if defn:
                    (build_path / "definition.json").write_text(defn.model_dump_json())
                (build_path / "solution.json").write_text(solution_.model_dump_json())
                (build_path / "workload.json").write_text(inner_workload.model_dump_json())

                cmd = ["ncu", "--page", ncu_page, "--set", ncu_set, "-f"]
                if ncu_kernel_name:
                    cmd.extend(["--kernel-name", ncu_kernel_name])
                cmd.extend([
                    _sys.executable, "-u", "-m",
                    "flashinfer_bench.agents._solution_runner",
                    "--data-dir", str(build_path),
                    "--device", "cuda:0",
                    "--trace-set-path", dataset_root_,
                ])

                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=ncu_timeout,
                    )
                    ncu_output = (proc.stdout or "") + (proc.stderr or "")
                except subprocess.TimeoutExpired:
                    ncu_output = f"[NCU timeout after {ncu_timeout}s]"
                except Exception as exc:
                    ncu_output = f"[NCU error: {exc}]"

            rows.append({"workload_uuid": wuuid, "axes": axes, "output": ncu_output})

        return {"ncu_set": ncu_set, "ncu_page": ncu_page, "ncu_kernel_name": ncu_kernel_name, "rows": rows}

    # ── Aggregate passing results ─────────────────────────────────────────────────
    passing_traces = []
    per_workload = []
    unhandled_traces = []
    for trace in traces:
        ev = trace.evaluation
        if ev is None:
            unhandled_traces.append(trace)
            continue
        if ev.status == EvaluationStatus.PASSED:
            passing_traces.append(trace)
            per_workload.append({
                "latency_ms": ev.performance.latency_ms,
                "reference_latency_ms": ev.performance.reference_latency_ms,
                "speedup": ev.performance.speedup_factor,
                "max_relative_error": ev.correctness.max_relative_error,
                "max_absolute_error": ev.correctness.max_absolute_error,
            })
        else:
            unhandled_traces.append(trace)

    if not per_workload:
        # Try to extract error info from unhandled traces
        error_details = []
        for trace in unhandled_traces:
            ev = trace.evaluation
            if ev is not None:
                error_details.append(f"{ev.status.value}: {ev.log}")
            else:
                error_details.append("evaluation=None (runner subprocess crashed)")
        if not traces:
            error_msg = "[v3] No traces returned from benchmark (dataset/solution mismatch?)"
        elif error_details:
            error_msg = "; ".join(error_details[:3])
        else:
            error_msg = "No evaluation results"
        return {
            "task_id": task_id,
            "compiled": False,
            "correct": False,
            "error": error_msg,
            "_debug": _debug,
        }

    n = len(per_workload)
    result = {
        "compiled": True,
        "correct": True,
        "speedup": sum(w["speedup"] for w in per_workload) / n,
        "latency_ms": sum(w["latency_ms"] for w in per_workload) / n,
        "task_id": task_id,
        "stats": {
            "reference_latency_ms": sum(w["reference_latency_ms"] for w in per_workload) / n,
            "max_relative_error": max(w["max_relative_error"] for w in per_workload),
            "max_absolute_error": max(w["max_absolute_error"] for w in per_workload),
            "total_workloads": n,
        },
        "per_workload": per_workload,
    }

    if include_ncu and passing_traces:
        result["ncu_profile"] = _run_ncu_profile(passing_traces, dataset_root, solution)

    return result


@app.local_entrypoint()
def upload_dataset(local_path: str, dataset_name: str = "mlsys26-contest", force: bool = False):
    """Upload local dataset to Modal Volume (shared with CUDA deploy).

    Usage:
        python -m modal run modal_eval_deploy_triton.py::upload_dataset \
            --local-path /path/to/datasets/mlsys26-contest \
            --dataset-name mlsys26-contest
    """
    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Not a directory: {local_path}")

    if not force:
        try:
            if dataset_vol.listdir(f"/{dataset_name}"):
                print(f"Dataset '{dataset_name}' already exists in Volume, skipping.")
                return
        except Exception:
            pass

    print(f"Uploading '{local_path}' -> Volume:/{dataset_name} (force={force}) ...")
    with dataset_vol.batch_upload(force=force) as batch:
        batch.put_directory(local_path, f"/{dataset_name}")
    print("Upload complete.")


@app.local_entrypoint()
def upload_moe_blobs(local_path: str):
    """Upload only MOE blob files to overwrite LFS pointers in Volume.

    Usage:
        python -m modal run modal_eval_deploy_triton.py::upload_moe_blobs \
            --local-path /root/datasets/mlsys26-contest
    """
    import pathlib
    moe_blob_dir = pathlib.Path(local_path) / "blob" / "workloads" / "moe"
    if not moe_blob_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {moe_blob_dir}")

    print(f"Uploading MOE blobs from '{moe_blob_dir}' ...")
    with dataset_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(moe_blob_dir), "/mlsys26-contest/blob/workloads/moe")
    print("MOE blob upload complete.")
