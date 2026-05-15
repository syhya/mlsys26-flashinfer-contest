"""
Modal remote GPU evaluation service for LoongFlow using flashinfer-bench.

Aligned with the official contest evaluation environment:
  - Image:   flashinfer/flashinfer-ci-cu132:latest  (official contest image)
  - Package: language=cuda, binding=torch           (matches submission format)
  - Params:  warmup=10, iterations=50, trials=3     (official benchmark params)
  - Runner:  use_isolated_runner=True               (matches leaderboard)

Deploy once:
    python -m modal deploy modal_eval_deploy.py

Ensure dataset is uploaded first:
    python -m modal run modal_eval_deploy.py::upload_dataset \
        --local-path /path/to/datasets/mlsys26-contest
"""

import os

import modal

MODAL_APP_NAME = "loongflow-cuda-eval"
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
    "gdn_prefill_qk4_v8_d128_k_last": dict(warmup_runs=1, iterations=5, num_trials=3),
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": dict(
        warmup_runs=10, iterations=50, num_trials=3,
        atol=1.0, rtol=0.3, required_matched_ratio=0.9,
    ),
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
    timeout=3600,   # 1 hour: official params + 23 workloads takes ~20-40 min
    serialized=True,
)
def remote_eval_kernel(
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
    Evaluate a CUDA kernel on Modal B200 using the official flashinfer-bench pipeline.

    Uses official contest image, benchmark parameters (warmup=10, iter=50, trials=3),
    CUDA + torch binding packaging, and isolated runner — matching leaderboard eval.

    kernel_code must expose:
        std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
            torch::Tensor q_nope, torch::Tensor q_pe,
            torch::Tensor ckv_cache, torch::Tensor kpe_cache,
            torch::Tensor sparse_indices, float sm_scale)
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

    # ── CUTLASS / CuTe include path setup (needed for kernels using cute/*.hpp) ──
    def _setup_cutlass_env():
        for module_name, rel_paths in [
            ("flashinfer", [("data", "cutlass", "include"), ("3rdparty", "cutlass", "include")]),
            ("tilelang", [("3rdparty", "cutlass", "include"), ("data", "cutlass", "include")]),
        ]:
            try:
                pkg_root = Path(__import__(module_name).__file__).resolve().parent
            except Exception:
                continue
            for rel_path in rel_paths:
                candidate = pkg_root.joinpath(*rel_path)
                if candidate.is_dir() and (candidate / "cute" / "tensor.hpp").exists():
                    inc = str(candidate)
                    os.environ["CUTLASS_INCLUDE"] = inc
                    for env_var in ("CPLUS_INCLUDE_PATH", "CPATH"):
                        cur = os.environ.get(env_var, "")
                        if inc not in cur.split(":"):
                            os.environ[env_var] = f"{inc}:{cur}" if cur else inc
                    flag = f"-I{inc}"
                    nvcc = os.environ.get("NVCC_PREPEND_FLAGS", "")
                    if flag not in nvcc:
                        os.environ["NVCC_PREPEND_FLAGS"] = f"{flag} --expt-relaxed-constexpr {nvcc}".strip()
                    return

    _setup_cutlass_env()

    # ── NCU profiling helper (inline to avoid cloudpickle module-reference issues) ──
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

    # ── Load dataset ──────────────────────────────────────────────────────────────
    dataset_root = os.path.join(DATASET_PATH, dataset_name)
    trace_set = TraceSet.from_path(dataset_root)

    definition = trace_set.definitions.get(task_id)
    if definition is None:
        return {"task_id": task_id, "error": f"Definition '{task_id}' not found in dataset"}

    workloads = trace_set.workloads.get(task_id, [])
    if not workloads:
        return {"task_id": task_id, "error": f"No workloads found for '{task_id}'"}

    # Shard workloads for parallel multi-GPU evaluation.
    # shard_idx=i, num_shards=N → this container processes workloads[i::N]
    if num_shards > 1:
        workloads = workloads[shard_idx::num_shards]
        if not workloads:
            # This shard has no workloads (e.g. more shards than workloads).
            return {
                "task_id": task_id, "compiled": True, "correct": True,
                "speedup": 0.0, "latency_ms": 0.0, "per_workload": [],
                "stats": {"reference_latency_ms": 0.0, "max_relative_error": 0.0,
                          "max_absolute_error": 0.0, "total_workloads": 0},
            }

    # ── Build Solution (official CUDA + torch binding packaging) ─────────────────
    # Derive entry point from task_id prefix (dsa_* → dsa_forward, etc.)
    _TASK_ENTRY_POINTS = {
        "dsa": "kernel.cu::dsa_forward",
        "gdn": "kernel.cu::gdn_forward",
        "moe": "kernel.cu::moe_forward",
    }
    task_prefix = task_id.split("_")[0]
    entry_point = _TASK_ENTRY_POINTS.get(task_prefix, f"kernel.cu::{task_prefix}_forward")

    solution_name = f"agent_{uuid.uuid4().hex[:8]}"
    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="agent",
        spec=BuildSpec(
            language="cuda",
            target_hardware=["cuda"],
            entry_point=entry_point,
            dependencies=[],
            binding="torch",
            destination_passing_style=False,  # kernel returns tuple, not in-place
        ),
        sources=[
            SourceFile(path="kernel.cu", content=kernel_code),
        ],
    )

    # Filtered TraceSet with only this task + solution (matches run_modal.py)
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={task_id: definition},
        solutions={task_id: [solution]},
        workloads={task_id: workloads},
        traces={task_id: []},
    )

    # Official benchmark parameters (definition-aware)
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
    finally:
        benchmark.close()

    traces = result_ts.traces.get(task_id, [])

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

    passing_traces = []
    per_workload = []
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status == EvaluationStatus.PASSED:
            passing_traces.append(trace)
            per_workload.append({
                "latency_ms": ev.performance.latency_ms,
                "reference_latency_ms": ev.performance.reference_latency_ms,
                "speedup": ev.performance.speedup_factor,
                "max_relative_error": ev.correctness.max_relative_error,
                "max_absolute_error": ev.correctness.max_absolute_error,
            })

    if not per_workload:
        return {"task_id": task_id, "error": "No evaluation results"}

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
def upload_dataset(local_path: str, dataset_name: str = "mlsys26-contest"):
    """Upload local dataset to Modal Volume.

    Usage:
        python -m modal run modal_eval_deploy.py::upload_dataset \
            --local-path /path/to/datasets/mlsys26-contest \
            --dataset-name mlsys26-contest
    """
    import os

    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Not a directory: {local_path}")

    try:
        if dataset_vol.listdir(f"/{dataset_name}"):
            print(f"Dataset '{dataset_name}' already exists in Volume, skipping.")
            return
    except Exception:
        pass

    print(f"Uploading '{local_path}' -> Volume:/{dataset_name} ...")
    with dataset_vol.batch_upload() as batch:
        batch.put_directory(local_path, f"/{dataset_name}")
    print("Upload complete.")
