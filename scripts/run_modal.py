from __future__ import annotations

"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal. Optionally profiles selected workloads
with torch.profiler and stores the profiler output in the JSON report.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import os
import sys
import json
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load project-level .env before importing modal.
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

import modal

def ensure_triton_testing() -> None:
    try:
        import triton.testing  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import torch

    triton_mod = types.ModuleType("triton")
    testing_mod = types.ModuleType("triton.testing")

    def do_bench(fn, warmup=25, rep=100, *args, **kwargs):
        del args, kwargs
        for _ in range(max(0, int(warmup))):
            fn()

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(max(1, int(rep))):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / max(1, int(rep))

        t0 = time.perf_counter()
        for _ in range(max(1, int(rep))):
            fn()
        return ((time.perf_counter() - t0) * 1000.0) / max(1, int(rep))

    testing_mod.do_bench = do_bench
    triton_mod.testing = testing_mod
    sys.modules.setdefault("triton", triton_mod)
    sys.modules["triton.testing"] = testing_mod

ensure_triton_testing()
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.compile import BuilderRegistry, BuildError

OFFICIAL_IMAGE = "flashinfer/flashinfer-ci-cu132:20260401-2c675fb"
# Always install flashinfer-bench from the current upstream main branch.
OFFICIAL_FLASHINFER_BENCH_GIT = (
    "git+https://github.com/flashinfer-ai/flashinfer-bench.git@main"
)
OFFICIAL_CONFIGS = {
    "gdn_prefill_qk4_v8_d128_k_last": dict(warmup_runs=1, iterations=5, num_trials=3),
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": dict(
        warmup_runs=10, iterations=50, num_trials=3,
        atol=1.0, rtol=0.3, required_matched_ratio=0.9,
    ),
}
OFFICIAL_DEFAULT = dict(warmup_runs=10, iterations=50, num_trials=3)
DISPLAY_SIGFIGS = 15

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        OFFICIAL_IMAGE,
        add_python=None,
    )
    .pip_install(OFFICIAL_FLASHINFER_BENCH_GIT, "cupti-python", force_build=True)
)


def _extract_workload_uuid(workload) -> str | None:
    uid = getattr(workload, "uuid", None)
    if uid is not None:
        return uid
    inner = getattr(workload, "workload", None)
    if inner is not None:
        return getattr(inner, "uuid", None)
    return None


def _extract_workload_axes(workload) -> dict:
    axes = getattr(workload, "axes", None)
    if axes is not None:
        return axes
    inner = getattr(workload, "workload", None)
    if inner is not None:
        return getattr(inner, "axes", {}) or {}
    return {}


def _extract_inner_workload(workload):
    inner = getattr(workload, "workload", None)
    return inner if inner is not None else workload


def _load_definition_and_workloads(trace_set: TraceSet, solution: Solution):
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")
    return definition, workloads


def _prepend_env_path(name: str, value: str) -> None:
    current = os.environ.get(name, "")
    parts = [part for part in current.split(":") if part]
    if value in parts:
        return
    os.environ[name] = f"{value}:{current}" if current else value


def _discover_cutlass_include() -> str | None:
    candidates: list[Path] = []
    for module_name, rel_paths in [
        ("flashinfer", [("data", "cutlass", "include"), ("3rdparty", "cutlass", "include")]),
        ("tilelang", [("3rdparty", "cutlass", "include"), ("data", "cutlass", "include")]),
    ]:
        try:
            pkg_root = Path(__import__(module_name).__file__).resolve().parent
        except Exception:
            continue
        for rel_path in rel_paths:
            candidates.append(pkg_root.joinpath(*rel_path))

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "cute" / "tensor.hpp").exists():
            return str(candidate)
    return None


def _setup_cutlass_env() -> None:
    """Set up CUTLASS/CuTe include paths for CUDA kernel compilation."""
    cutlass_inc = _discover_cutlass_include()
    if not cutlass_inc:
        return

    os.environ["CUTLASS_INCLUDE"] = cutlass_inc
    _prepend_env_path("CPLUS_INCLUDE_PATH", cutlass_inc)
    _prepend_env_path("CPATH", cutlass_inc)

    nvcc_flags = os.environ.get("NVCC_PREPEND_FLAGS", "")
    include_flag = f"-I{cutlass_inc}"
    if include_flag not in nvcc_flags:
        extra = f"{include_flag} --expt-relaxed-constexpr"
        os.environ["NVCC_PREPEND_FLAGS"] = f"{extra} {nvcc_flags}".strip()


def _preflight_build(definition, solution: Solution) -> None:
    _setup_cutlass_env()
    try:
        registry = BuilderRegistry.get_instance()
        registry.build(definition, solution)
    except BuildError as e:
        raise RuntimeError(f"Preflight compilation failed: {e}") from e


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        kwargs = dict(OFFICIAL_DEFAULT)
        kwargs.update(OFFICIAL_CONFIGS.get(solution.definition, {}))
        config = BenchmarkConfig(
            **kwargs,
            use_isolated_runner=True,
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    definition, workloads = _load_definition_and_workloads(trace_set, solution)

    # Preflight compile in the current process so compile failures return clear errors.
    _preflight_build(definition, solution)

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            workload_uuid = getattr(getattr(trace, "workload", None), "uuid", None)
            if not workload_uuid:
                workload_uuid = f"unknown_workload_uuid_{len(results[definition.name])}"
            entry = {
                "status": trace.evaluation.status.value,
                "solution": getattr(trace.solution, "name", str(trace.solution)),
            }
            eval_log = getattr(trace.evaluation, "log", None)
            if eval_log:
                entry["log"] = eval_log
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error

            results[definition.name][workload_uuid] = entry

    return results


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_profile(
    solution: Solution,
    workload_ids: list[str],
    sort_by: str = "cuda_time_total",
    row_limit: int = 40,
    warmup_runs: int = 3,
) -> dict:
    """Run torch.profiler for selected workloads on Modal B200."""
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("PATH", f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}")

    import torch
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition, workloads = _load_definition_and_workloads(trace_set, solution)
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)
    try:
        workload_map = {}
        for workload in workloads:
            workload_uuid = _extract_workload_uuid(workload)
            if workload_uuid is not None:
                workload_map[workload_uuid] = workload

        rows = []
        missing_workloads = []
        for workload_id in workload_ids:
            workload = workload_map.get(workload_id)
            if workload is None:
                missing_workloads.append(workload_id)
                continue

            inner_workload = _extract_inner_workload(workload)
            safe_tensors = None
            if any(inp.type == "safetensors" for inp in inner_workload.inputs.values()):
                safe_tensors = load_safetensors(definition, inner_workload, Path(TRACE_SET_PATH))
            inputs = gen_inputs(definition, inner_workload, "cuda:0", safe_tensors)
            outputs = allocate_outputs(definition, inputs, "cuda:0")

            with torch.no_grad():
                for _ in range(max(1, warmup_runs)):
                    runnable.call_destination_passing(*inputs, *outputs)
                torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                with torch.no_grad():
                    runnable.call_destination_passing(*inputs, *outputs)
                torch.cuda.synchronize()

            table = prof.key_averages().table(
                sort_by=sort_by,
                row_limit=row_limit,
                max_src_column_width=120,
                max_name_column_width=80,
            )
            rows.append(
                {
                    "workload_uuid": workload_id,
                    "axes": _extract_workload_axes(workload),
                    "sort_by": sort_by,
                    "row_limit": row_limit,
                    "warmup_runs": warmup_runs,
                    "table": table,
                }
            )
    finally:
        runnable.cleanup()

    return {
        "requested_workloads": workload_ids,
        "profiled_workloads": [row["workload_uuid"] for row in rows],
        "missing_workloads": missing_workloads,
        "profiler": "torch.profiler",
        "sort_by": sort_by,
        "row_limit": row_limit,
        "warmup_runs": warmup_runs,
        "rows": rows,
    }


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu(
    solution: Solution,
    workload_ids: list[str],
    ncu_set: str = "detailed",
    ncu_page: str = "details",
    ncu_timeout: int = 240,
    kernel_name: str | None = None,
) -> dict:
    """Run NCU profiling for selected workloads on Modal B200.

    Uses direct subprocess invocation (bypasses NVTX filtering which is
    broken on Modal containers).
    """
    import subprocess
    import tempfile

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("PATH", f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}")

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition, workloads = _load_definition_and_workloads(trace_set, solution)

    workload_map = {}
    for workload in workloads:
        workload_uuid = _extract_workload_uuid(workload)
        if workload_uuid is not None:
            workload_map[workload_uuid] = workload

    rows = []
    missing_workloads = []
    for workload_id in workload_ids:
        workload = workload_map.get(workload_id)
        if workload is None:
            missing_workloads.append(workload_id)
            continue

        inner_workload = _extract_inner_workload(workload)

        with tempfile.TemporaryDirectory(prefix="fib_ncu_") as build_dir:
            build_path = Path(build_dir)
            (build_path / "definition.json").write_text(definition.model_dump_json())
            (build_path / "solution.json").write_text(solution.model_dump_json())
            (build_path / "workload.json").write_text(inner_workload.model_dump_json())

            cmd = [
                "ncu",
                "--page", ncu_page,
                "--set", ncu_set,
                "-f",
            ]
            if kernel_name:
                cmd.extend(["--kernel-name", kernel_name])
            cmd.extend([
                sys.executable, "-u", "-m",
                "flashinfer_bench.agents._solution_runner",
                "--data-dir", str(build_path),
                "--device", "cuda:0",
                "--trace-set-path", TRACE_SET_PATH,
            ])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=ncu_timeout,
                )
                output = result.stdout + result.stderr
                if result.returncode != 0:
                    output = f"ERROR: NCU exited with code {result.returncode}:\n{output}"
            except subprocess.TimeoutExpired:
                output = f"ERROR: NCU profiling timed out after {ncu_timeout}s"

        rows.append({
            "workload_uuid": workload_id,
            "axes": _extract_workload_axes(workload),
            "output": output,
        })

    return {
        "requested_workloads": workload_ids,
        "profiled_workloads": [row["workload_uuid"] for row in rows],
        "missing_workloads": missing_workloads,
        "profiler": "ncu",
        "ncu_set": ncu_set,
        "ncu_page": ncu_page,
        "kernel_name": kernel_name,
        "timeout": ncu_timeout,
        "rows": rows,
    }


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {_fmt_optional(result['latency_ms'])} ms", end="")

            if result.get("reference_latency_ms") is not None:
                print(f" | ref {_fmt_optional(result['reference_latency_ms'])} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {_fmt_optional(result['speedup_factor'])}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            if status in {"COMPILE_ERROR", "RUNTIME_ERROR"} and result.get("log"):
                log = result["log"].strip()
                snippet = log if len(log) <= 1200 else (log[:1200] + "...[truncated]")
                print("    log:")
                for line in snippet.splitlines():
                    print(f"      {line}")

        aggregate = _compute_aggregate_metrics(traces)
        print(
            "  Overall:"
            f" passed={aggregate['passed_workloads']}/{aggregate['total_workloads']}"
            f" | avg={_fmt_optional(aggregate['avg_latency_ms'])} ms"
            f" | median={_fmt_optional(aggregate['median_latency_ms'])} ms"
            f" | p95={_fmt_optional(aggregate['p95_latency_ms'])} ms"
            f" | avg_ref={_fmt_optional(aggregate['avg_reference_latency_ms'])} ms"
            f" | avg_speedup={_fmt_optional(aggregate['avg_speedup_factor'])}x"
        )


def print_ncu_results(ncu_report: dict | None, max_lines: int = 220) -> None:
    """Print NCU profiling results (truncated for console)."""
    if not ncu_report:
        return

    rows = ncu_report.get("rows", [])
    print("\nNCU Profiler:")
    print(
        "  Summary:"
        f" profiled={len(rows)}"
        f" | missing={len(ncu_report.get('missing_workloads', []))}"
        f" | set={ncu_report.get('ncu_set')}"
        f" | page={ncu_report.get('ncu_page')}"
        f" | kernel_name={ncu_report.get('kernel_name', 'all')}"
    )

    for row in rows:
        print(f"  Workload {row['workload_uuid'][:8]}... | axes={row.get('axes', {})}")
        output = (row.get("output") or "").strip()
        if not output:
            print("    <empty NCU output>")
            continue
        out_lines = output.splitlines()
        if max_lines > 0 and len(out_lines) > max_lines:
            out_lines = out_lines[:max_lines]
            out_lines.append(f"...[truncated, {len(output.splitlines()) - max_lines} more lines]")
        for line in out_lines:
            print(f"    {line}")

    for workload_id in ncu_report.get("missing_workloads", []):
        print(f"  Workload {workload_id[:8]}... | missing from trace set")


def print_profile_results(profile_report: dict | None) -> None:
    """Print torch.profiler results in a compact way."""
    if not profile_report:
        return

    rows = profile_report.get("rows", [])
    print("\nTorch Profiler:")
    print(
        "  Summary:"
        f" profiled={len(rows)}"
        f" | missing={len(profile_report.get('missing_workloads', []))}"
        f" | sort_by={profile_report.get('sort_by')}"
        f" | row_limit={profile_report.get('row_limit')}"
    )

    for row in rows:
        print(f"  Workload {row['workload_uuid'][:8]}... | axes={row.get('axes', {})}")
        table = (row.get("table") or "").strip()
        if not table:
            print("    <empty profiler output>")
            continue
        for line in table.splitlines():
            print(f"    {line}")

    for workload_id in profile_report.get("missing_workloads", []):
        print(f"  Workload {workload_id[:8]}... | missing from trace set")


def _fmt_optional(value, sigfigs: int = DISPLAY_SIGFIGS) -> str:
    if value is None:
        return "n/a"
    return format(float(value), f".{sigfigs}g")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(round((len(sorted_vals) - 1) * q))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def _compute_aggregate_metrics(traces: dict) -> dict:
    total = len(traces)
    passed = 0
    latencies = []
    ref_latencies = []
    speedups = []

    for item in traces.values():
        status = item.get("status")
        if status == "PASSED":
            passed += 1
        if status == "PASSED" and item.get("latency_ms") is not None:
            latencies.append(float(item["latency_ms"]))
        if status == "PASSED" and item.get("reference_latency_ms") is not None:
            ref_latencies.append(float(item["reference_latency_ms"]))
        if status == "PASSED" and item.get("speedup_factor") is not None:
            speedups.append(float(item["speedup_factor"]))

    return {
        "total_workloads": total,
        "passed_workloads": passed,
        "failed_workloads": total - passed,
        "passed_with_latency": len(latencies),
        "passed_with_reference_latency": len(ref_latencies),
        "passed_with_speedup": len(speedups),
        "avg_latency_ms": mean(latencies) if latencies else None,
        "median_latency_ms": median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "avg_reference_latency_ms": mean(ref_latencies) if ref_latencies else None,
        "median_reference_latency_ms": median(ref_latencies) if ref_latencies else None,
        "p95_reference_latency_ms": _percentile(ref_latencies, 0.95),
        "avg_speedup_factor": mean(speedups) if speedups else None,
        "median_speedup_factor": median(speedups) if speedups else None,
        "p95_speedup_factor": _percentile(speedups, 0.95),
        "min_speedup_factor": min(speedups) if speedups else None,
        "max_speedup_factor": max(speedups) if speedups else None,
    }

def summarize_results(results: dict) -> dict:
    """Build summary with pass/fail and aggregate performance metrics."""
    summary = {}
    for def_name, traces in results.items():
        summary[def_name] = _compute_aggregate_metrics(traces)
    return summary


def _parse_workload_ids(raw_workload_ids: str) -> list[str]:
    if not raw_workload_ids:
        return []
    return [item.strip() for item in raw_workload_ids.split(",") if item.strip()]


def _select_profile_workloads(
    results: dict,
    requested_workload_ids: list[str],
    max_workloads: int,
) -> list[str]:
    if requested_workload_ids:
        return requested_workload_ids

    selected = []
    for traces in results.values():
        for workload_uuid, item in traces.items():
            if item.get("status") == "PASSED":
                selected.append(workload_uuid)
    if max_workloads <= 0:
        return selected
    return selected[:max_workloads]


def save_json_report(
    results: dict,
    solution: Solution,
    output_path: Path,
    profile_report: dict | None = None,
    ncu_report: dict | None = None,
) -> None:
    """Persist detailed benchmark outputs into one JSON file."""
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "solution": {
            "name": solution.name,
            "definition": solution.definition,
            "author": solution.author,
        },
        "summary": summarize_results(results),
        "results": results,
    }
    if profile_report is not None:
        payload["torch_profile"] = profile_report
    if ncu_report is not None:
        payload["ncu_profile"] = ncu_report
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_ncu_markdown_report(
    ncu_report: dict,
    solution: Solution,
    output_path: Path,
) -> None:
    """Save NCU profiling results as a formatted markdown file."""
    lines = [
        "# NCU Profiling Report",
        "",
        f"- **Solution**: {solution.name}",
        f"- **Definition**: {solution.definition}",
        f"- **NCU Set**: {ncu_report.get('ncu_set', 'n/a')}",
        f"- **NCU Page**: {ncu_report.get('ncu_page', 'n/a')}",
        f"- **Kernel Filter**: {ncu_report.get('kernel_name') or 'all'}",
        f"- **Timeout**: {ncu_report.get('timeout', 'n/a')}s",
        f"- **Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    rows = ncu_report.get("rows", [])
    missing = ncu_report.get("missing_workloads", [])
    lines.append(f"**Profiled**: {len(rows)} workload(s) | **Missing**: {len(missing)}")
    lines.append("")

    for row in rows:
        wid = row.get("workload_uuid", "unknown")
        axes = row.get("axes", {})
        lines.append(f"## Workload `{wid}`")
        if axes:
            axes_str = ", ".join(f"{k}={v}" for k, v in axes.items())
            lines.append(f"**Axes**: {axes_str}")
        lines.append("")
        output = (row.get("output") or "").strip()
        if output:
            lines.append("```")
            lines.append(output)
            lines.append("```")
        else:
            lines.append("*No NCU output captured.*")
        lines.append("")

    if missing:
        lines.append("## Missing Workloads")
        lines.append("")
        for wid in missing:
            lines.append(f"- `{wid}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


@app.local_entrypoint()
def main(
    profile_torch: bool = True,
    profile_workload_ids: str = "",
    profile_max_workloads: int = 0,
    profile_sort_by: str = "cuda_time_total",
    profile_row_limit: int = 40,
    profile_warmup_runs: int = 3,
    profile_ncu: bool = False,
    ncu_workload_ids: str = "",
    ncu_max_workloads: int = 0,
    ncu_set: str = "detailed",
    ncu_page: str = "details",
    ncu_timeout: int = 240,
    ncu_max_lines: int = 220,
    ncu_kernel_name: str = "",
    config_path: str = "",
    solution_dir: str = "",
):
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution(
        config_path=Path(config_path) if config_path else None,
        solution_dir=Path(solution_dir) if solution_dir else None,
    )

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)

    profile_report = None
    if profile_torch:
        selected_workload_ids = _select_profile_workloads(
            results=results,
            requested_workload_ids=_parse_workload_ids(profile_workload_ids),
            max_workloads=profile_max_workloads,
        )
        if not selected_workload_ids:
            print("\nTorch profiler skipped: no workloads selected.")
        else:
            print(
                "\nRunning torch.profiler on Modal B200..."
                f" workloads={selected_workload_ids}"
            )
            profile_report = run_profile.remote(
                solution,
                selected_workload_ids,
                profile_sort_by,
                profile_row_limit,
                profile_warmup_runs,
            )
            print_profile_results(profile_report)

    ncu_report = None
    if profile_ncu:
        selected_ncu_ids = _select_profile_workloads(
            results=results,
            requested_workload_ids=_parse_workload_ids(ncu_workload_ids),
            max_workloads=ncu_max_workloads,
        )
        if not selected_ncu_ids:
            print("\nNCU profiler skipped: no workloads selected.")
        else:
            print(
                "\nRunning NCU profiler on Modal B200..."
                f" workloads={selected_ncu_ids}"
            )
            ncu_report = run_ncu.remote(
                solution,
                selected_ncu_ids,
                ncu_set,
                ncu_page,
                ncu_timeout,
                ncu_kernel_name or None,
            )
            print_ncu_results(ncu_report, max_lines=ncu_max_lines)
            ncu_md_path = PROJECT_ROOT / "ncu_profile_report.md"
            save_ncu_markdown_report(ncu_report, solution, ncu_md_path)
            print(f"  NCU markdown report saved to: {ncu_md_path}")

    report_path = PROJECT_ROOT / "benchmark_detailed_results.json"
    save_json_report(
        results, solution, report_path,
        profile_report=profile_report,
        ncu_report=ncu_report,
    )
    print(f"\nDetailed JSON report saved to: {report_path}")


if __name__ == "__main__":
    print(
        "This script must be started with Modal CLI:\n"
        "  modal run scripts/run_modal.py\n"
        "  modal run scripts/run_modal.py --profile-torch --profile-max-workloads 0\n"
        "  modal run scripts/run_modal.py --profile-ncu --ncu-max-workloads 3\n"
        "  modal run scripts/run_modal.py --profile-ncu --ncu-kernel-name my_kernel\n"
        "Running with plain `python` will not trigger @app.local_entrypoint()."
    )
