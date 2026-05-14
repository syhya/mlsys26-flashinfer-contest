"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution

OFFICIAL_CONFIGS = {
    "gdn_prefill_qk4_v8_d128_k_last": dict(warmup_runs=1, iterations=5, num_trials=3),
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": dict(
        warmup_runs=10, iterations=50, num_trials=3,
        atol=1.0, rtol=0.3, required_matched_ratio=0.9,
    ),
}
OFFICIAL_DEFAULT = dict(warmup_runs=10, iterations=50, num_trials=3)
DISPLAY_SIGFIGS = 15


def resolve_official_benchmark_kwargs(definition: str) -> dict:
    kwargs = dict(OFFICIAL_DEFAULT)
    kwargs.update(OFFICIAL_CONFIGS.get(definition, {}))
    return kwargs


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


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    if config is None:
        config = BenchmarkConfig(
            **resolve_official_benchmark_kwargs(solution.definition),
            use_isolated_runner=True,
        )

    _setup_cutlass_env()
    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

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
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {_fmt_metric(result['latency_ms'])} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {_fmt_metric(result['speedup_factor'])}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def _fmt_metric(value: float | None, sigfigs: int = DISPLAY_SIGFIGS) -> str:
    if value is None:
        return "n/a"
    return format(float(value), f".{sigfigs}g")


def main():
    """Pack solution and run benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack and run a local FlashInfer-Bench benchmark")
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to a definition config.toml, for example gdn_decode_qk4_v8_d128_k_last/config.toml",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=None,
        help="Optional source directory override. Defaults to <config-dir>/solution/<language>.",
    )
    args = parser.parse_args()

    print("Packing solution from source files...")
    solution_path = pack_solution(config_path=args.config_path, solution_dir=args.solution_dir)

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    results = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()
