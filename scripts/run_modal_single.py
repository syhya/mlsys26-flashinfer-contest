from __future__ import annotations

"""
FlashInfer-Bench single-workload Modal runner.

Useful for fast correctness/debug iterations before launching the full 19-case sweep.
"""

import os
import sys
import json
import time
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


app = modal.App("flashinfer-bench-single")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        OFFICIAL_IMAGE,
        add_python=None,
    )
    # Modal's registry image does not expose flashinfer_bench as an importable
    # Python package. Install GitHub main and force this layer to rebuild so
    # moving main is resolved again on every benchmark run.
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


def _resolve_workload(workloads, workload_selector: str):
    if workload_selector.isdigit():
        idx = int(workload_selector)
        zero_based = idx if 0 <= idx < len(workloads) else idx - 1
        if 0 <= zero_based < len(workloads):
            return workloads[zero_based]
        raise ValueError(
            f"Workload index '{workload_selector}' is out of range for {len(workloads)} workloads"
        )

    for workload in workloads:
        workload_uuid = _extract_workload_uuid(workload)
        if workload_uuid == workload_selector:
            return workload

    raise ValueError(f"Workload '{workload_selector}' not found")


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_single_benchmark(
    solution_json: str,
    workload_selector: str,
    config_kwargs: dict | None = None,
) -> dict:
    solution = Solution.model_validate_json(solution_json)
    if config_kwargs is None:
        config_kwargs = dict(warmup_runs=10, iterations=50, num_trials=3, use_isolated_runner=True)
    config = BenchmarkConfig(**config_kwargs)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition, workloads = _load_definition_and_workloads(trace_set, solution)

    workload = _resolve_workload(workloads, workload_selector)
    _preflight_build(definition, solution)

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: [workload]},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if not trace.evaluation:
            continue
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
        results[definition.name][trace.workload.uuid] = entry

    return results


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_single_profile(
    solution_json: str,
    workload_selector: str,
    sort_by: str = "cuda_time_total",
    row_limit: int = 40,
    warmup_runs: int = 3,
) -> dict:
    """Run torch.profiler for a single selected workload on Modal B200."""
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("PATH", f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}")
    _setup_cutlass_env()

    import torch
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition, workloads = _load_definition_and_workloads(trace_set, solution)
    workload = _resolve_workload(workloads, workload_selector)

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)
    try:
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

        workload_uuid = _extract_workload_uuid(workload) or workload_selector
        return {
            "requested_workload": workload_selector,
            "profiled_workloads": [workload_uuid],
            "missing_workloads": [],
            "profiler": "torch.profiler",
            "sort_by": sort_by,
            "row_limit": row_limit,
            "warmup_runs": warmup_runs,
            "rows": [
                {
                    "workload_uuid": workload_uuid,
                    "axes": _extract_workload_axes(workload),
                    "sort_by": sort_by,
                    "row_limit": row_limit,
                    "warmup_runs": warmup_runs,
                    "table": prof.key_averages().table(
                        sort_by=sort_by,
                        row_limit=row_limit,
                        max_src_column_width=120,
                        max_name_column_width=80,
                    ),
                }
            ],
        }
    finally:
        runnable.cleanup()


def _save_json_report(
    results: dict,
    solution_meta: dict,
    output_path: Path,
    profile_report: dict | None = None,
    ncu_report: dict | None = None,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "solution": {
            "name": solution_meta["name"],
            "definition": solution_meta["definition"],
            "author": solution_meta["author"],
        },
        "results": results,
    }
    if profile_report is not None:
        payload["torch_profile"] = profile_report
    if ncu_report is not None:
        payload["ncu_profile"] = ncu_report
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_single_ncu(
    solution_json: str,
    workload_selector: str,
    ncu_set: str = "detailed",
    ncu_page: str = "details",
    ncu_timeout: int = 240,
    kernel_name: str | None = None,
) -> dict:
    """Run NCU profiling for a single workload on Modal B200.

    Uses direct subprocess invocation (bypasses NVTX filtering which is
    broken on Modal containers).
    """
    import subprocess
    import tempfile

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("PATH", f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}")
    _setup_cutlass_env()

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition, workloads = _load_definition_and_workloads(trace_set, solution)
    workload = _resolve_workload(workloads, workload_selector)

    inner_workload = _extract_inner_workload(workload)
    workload_uuid = _extract_workload_uuid(workload) or workload_selector

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

    return {
        "requested_workload": workload_selector,
        "profiled_workloads": [workload_uuid],
        "missing_workloads": [],
        "profiler": "ncu",
        "ncu_set": ncu_set,
        "ncu_page": ncu_page,
        "kernel_name": kernel_name,
        "timeout": ncu_timeout,
        "rows": [
            {
                "workload_uuid": workload_uuid,
                "axes": _extract_workload_axes(workload),
                "output": output,
            }
        ],
    }


def _print_results(results: dict):
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            print(f"  Workload {workload_uuid}: {result.get('status')}", end="")
            if result.get("latency_ms") is not None:
                print(f" | {_fmt_metric(result['latency_ms'])} ms", end="")
            if result.get("reference_latency_ms") is not None:
                print(f" | ref {_fmt_metric(result['reference_latency_ms'])} ms", end="")
            if result.get("speedup_factor") is not None:
                print(f" | {_fmt_metric(result['speedup_factor'])}x", end="")
            if result.get("max_abs_error") is not None:
                print(
                    f" | abs_err={result['max_abs_error']:.2e},"
                    f" rel_err={result.get('max_rel_error', 0):.2e}",
                    end="",
                )
            print()
            if result.get("log"):
                snippet = result["log"]
                if len(snippet) > 1200:
                    snippet = snippet[:1200] + "...[truncated]"
                print("    log:")
                for line in snippet.splitlines():
                    print(f"      {line}")


def _fmt_metric(value: float | None, sigfigs: int = DISPLAY_SIGFIGS) -> str:
    if value is None:
        return "n/a"
    return format(float(value), f".{sigfigs}g")


def _save_ncu_markdown_report(
    ncu_report: dict,
    solution_meta: dict,
    output_path: Path,
) -> None:
    """Save NCU profiling results as a formatted markdown file."""
    lines = [
        "# NCU Profiling Report",
        "",
        f"- **Solution**: {solution_meta['name']}",
        f"- **Definition**: {solution_meta['definition']}",
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


def _print_ncu_results(ncu_report: dict | None, max_lines: int = 220) -> None:
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
        print(f"  Workload {row['workload_uuid']} | axes={row.get('axes', {})}")
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


def _print_profile_results(profile_report: dict | None) -> None:
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
        print(f"  Workload {row['workload_uuid']} | axes={row.get('axes', {})}")
        table = (row.get("table") or "").strip()
        if not table:
            print("    <empty profiler output>")
            continue
        for line in table.splitlines():
            print(f"    {line}")


@app.local_entrypoint()
def main(
    workload_uuid: str = "",
    warmup_runs: int = 10,
    iterations: int = 50,
    num_trials: int = 3,
    use_isolated_runner: bool = True,
    official: bool = False,
    atol: float = 0.01,
    rtol: float = 0.01,
    required_matched_ratio: float = 0.0,
    profile_torch: bool = False,
    profile_sort_by: str = "cuda_time_total",
    profile_row_limit: int = 40,
    profile_warmup_runs: int = 3,
    profile_ncu: bool = False,
    ncu_set: str = "detailed",
    ncu_page: str = "details",
    ncu_timeout: int = 240,
    ncu_max_lines: int = 220,
    ncu_kernel_name: str = "",
    config_path: str = "",
    solution_dir: str = "",
):
    from scripts.pack_solution import pack_solution

    if not workload_uuid:
        raise ValueError("workload_uuid is required unless you are only checking runtime alignment")

    print("Packing solution from source files...")
    packed_solution_path = (
        Path(tempfile.gettempdir()) /
        f"flashinfer_single_{os.getpid()}_{workload_uuid.replace('-', '_')}.json"
    )
    solution_path = pack_solution(
        output_path=packed_solution_path,
        config_path=Path(config_path) if config_path else None,
        solution_dir=Path(solution_dir) if solution_dir else None,
    )

    solution_json = solution_path.read_text(encoding="utf-8")
    solution_meta = json.loads(solution_json)

    if official:
        oc = dict(OFFICIAL_DEFAULT)
        oc.update(OFFICIAL_CONFIGS.get(solution_meta["definition"], {}))
        warmup_runs = oc["warmup_runs"]
        iterations = oc["iterations"]
        num_trials = oc["num_trials"]
        use_isolated_runner = True
        atol = oc.get("atol", 0.01)
        rtol = oc.get("rtol", 0.01)
        required_matched_ratio = oc.get("required_matched_ratio", 0.0)

    config_kwargs = dict(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
        use_isolated_runner=use_isolated_runner,
        atol=atol,
        rtol=rtol,
    )
    if required_matched_ratio > 0:
        config_kwargs["required_matched_ratio"] = required_matched_ratio

    print(
        f"\nRunning single-workload benchmark on Modal B200"
        f" (selector={workload_uuid}, warmup={warmup_runs}, iterations={iterations}, trials={num_trials})..."
    )
    results = run_single_benchmark.remote(solution_json, workload_uuid, config_kwargs)
    _print_results(results)

    profile_report = None
    if profile_torch:
        print(
            "\nRunning torch.profiler on Modal B200"
            f" (selector={workload_uuid}, sort_by={profile_sort_by}, row_limit={profile_row_limit})..."
        )
        profile_report = run_single_profile.remote(
            solution_json,
            workload_uuid,
            profile_sort_by,
            profile_row_limit,
            profile_warmup_runs,
        )
        _print_profile_results(profile_report)

    ncu_report = None
    if profile_ncu:
        print(
            "\nRunning NCU profiler on Modal B200"
            f" (selector={workload_uuid}, set={ncu_set}, page={ncu_page})..."
        )
        ncu_report = run_single_ncu.remote(
            solution_json,
            workload_uuid,
            ncu_set,
            ncu_page,
            ncu_timeout,
            ncu_kernel_name or None,
        )
        _print_ncu_results(ncu_report, max_lines=ncu_max_lines)
        ncu_md_path = PROJECT_ROOT / "ncu_profile_report_single.md"
        _save_ncu_markdown_report(ncu_report, solution_meta, ncu_md_path)
        print(f"NCU markdown report saved to: {ncu_md_path}")

    report_path = PROJECT_ROOT / "benchmark_detailed_result_single.json"
    _save_json_report(
        results, solution_meta, report_path,
        profile_report=profile_report,
        ncu_report=ncu_report,
    )
    print(f"\nDetailed JSON report saved to: {report_path}")


if __name__ == "__main__":
    print(
        "This script must be started with Modal CLI:\n"
        "  modal run scripts/run_modal_single.py --workload-uuid 1\n"
        "  modal run scripts/run_modal_single.py --workload-uuid 1 --profile-torch\n"
        "  modal run scripts/run_modal_single.py --workload-uuid 1 --profile-ncu\n"
        "  modal run scripts/run_modal_single.py --workload-uuid 1 --profile-ncu --ncu-kernel-name my_kernel\n"
    )
