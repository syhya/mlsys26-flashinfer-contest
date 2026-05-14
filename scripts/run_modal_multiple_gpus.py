#!/usr/bin/env python3
"""
Parallel evaluation of all workloads for any contest problem via run_modal_single.py.

This script is the generalized parallel benchmarking harness for the MLSYS26
FlashInfer contest.  It supports all 5 contest problems (and any future ones)
by dynamically resolving the definition, workload JSONL, solution source
directory, and axis metadata at runtime.

Each workload is dispatched to its own Modal container running
``run_modal_single.py``, so up to ``--workers`` workloads execute in parallel.
A real-time progress bar, per-workload result lines, and incremental JSON
snapshots are emitted as results arrive.

Outputs (written to ``--out-dir``, defaults to project root):
    - ``benchmark_detailed_results.json`` – machine-readable results + summary
    - ``retained_run.log`` – human-readable log identical to the console output

Usage:
    # Use definition from root config.toml (default):
    conda run -n fi-bench python scripts/run_modal_multiple_gpus.py

    # Specify a definition explicitly:
    conda run -n fi-bench python scripts/run_modal_multiple_gpus.py \\
        --definition gdn_decode_qk4_v8_d128_k_last

    # With custom workers and output directory:
    conda run -n fi-bench python scripts/run_modal_multiple_gpus.py \\
        --definition gdn_prefill_qk4_v8_d128_k_last --workers 10 \\
        --out-dir optimize_ops/gdn_prefill_qk4_v8_d128_k_last

    # Use a per-problem solution directory instead of root solution/:
    conda run -n fi-bench python scripts/run_modal_multiple_gpus.py \\
        --definition gdn_decode_qk4_v8_d128_k_last \\
        --solution-dir gdn_decode_qk4_v8_d128_k_last/solution/cuda
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

# tomllib is built-in from Python 3.11+; fall back to the backport for 3.10.
try:
    import tomllib
except ImportError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parent.parent

OFFICIAL_CONFIGS = {
    "gdn_prefill_qk4_v8_d128_k_last": dict(warmup_runs=1, iterations=5, num_trials=3),
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": dict(
        warmup_runs=10, iterations=50, num_trials=3,
        atol=1.0, rtol=0.3, required_matched_ratio=0.9,
    ),
}
OFFICIAL_DEFAULT = dict(warmup_runs=10, iterations=50, num_trials=3)
DISPLAY_SIGFIGS = 15
LATENCY_COL_WIDTH = 16
REFERENCE_COL_WIDTH = 16
SPEEDUP_COL_WIDTH = 16


def resolve_official_benchmark_kwargs(definition: str) -> dict:
    kwargs = dict(OFFICIAL_DEFAULT)
    kwargs.update(OFFICIAL_CONFIGS.get(definition, {}))
    return kwargs


# ===========================================================================
# Section 1: Workload Discovery
# ===========================================================================
# Workload JSONL files are searched in priority order.  The first match wins.
# ``mlsys26-contest/workloads/`` is checked first because it contains the
# official contest evaluation data; ``flashinfer-trace/workloads/`` serves
# as a fallback with a broader (but unofficial) workload set.
# ===========================================================================

def workload_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    dataset_root = os.environ.get("FIB_DATASET_PATH")
    if dataset_root:
        root = Path(dataset_root).expanduser()
        dirs.extend([root / "workloads", root])
    dirs.extend(
        [
            PROJECT_ROOT / "mlsys26-contest" / "workloads",
            PROJECT_ROOT / "flashinfer-trace" / "workloads",
        ]
    )
    return dirs


def find_workload_jsonl(definition: str) -> Path:
    """Locate the workload JSONL file for *definition* by scanning search dirs.

    The search is recursive (``rglob``) so the file can live under an
    arbitrary category sub-directory (e.g. ``gdn/``, ``moe/``).

    Raises:
        FileNotFoundError: if no matching ``.jsonl`` is found anywhere.
    """
    search_dirs = workload_search_dirs()
    for base in search_dirs:
        for jsonl in base.rglob(f"{definition}.jsonl"):
            return jsonl
    raise FileNotFoundError(
        f"Workload file '{definition}.jsonl' not found in: "
        + ", ".join(str(d) for d in search_dirs)
    )


def load_workloads(definition: str) -> tuple[list[dict], list[str]]:
    """Parse all workloads from the JSONL file for *definition*.

    Returns:
        A 2-tuple ``(workloads, axes_names)`` where:
        - *workloads* is a list of dicts, each containing ``"uuid"`` plus
          every axis key-value pair (e.g. ``total_seq_len``, ``batch_size``).
        - *axes_names* is the ordered list of axis key names extracted from
          the first workload line (used later for dynamic log column headers).
    """
    jsonl_path = find_workload_jsonl(definition)
    workloads = []
    axes_names: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            w = json.loads(line)
            axes = w["workload"]["axes"]
            # Capture axis names from the first workload (all workloads
            # within the same definition share the same axes schema).
            if not axes_names:
                axes_names = list(axes.keys())
            entry = {"uuid": w["workload"]["uuid"]}
            entry.update(axes)
            workloads.append(entry)
    return workloads, axes_names


# ===========================================================================
# Section 2: Config / Solution Resolution
# ===========================================================================
# ``pack_solution.py`` can read definition-local ``config.toml`` files and
# source directories.  The public repo keeps one directory per official
# definition, so this resolver prefers ``<definition>/config.toml`` and
# ``<definition>/solution/<language>/``.
# ===========================================================================


def read_definition_from_config(config_path: Path) -> str:
    """Read the ``[solution].definition`` field from a TOML config file."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["solution"]["definition"]


def read_config(config_path: Path) -> dict:
    """Read a TOML config file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_and_solution(
    definition: str, solution_dir: str | None, config_path: str | None = None
) -> tuple[Path, Path]:
    """Determine the config file and solution source directory to use.

    Resolution strategy:
      1. Use ``--config-path`` when provided, else ``<definition>/config.toml``.
      2. Read ``[build].language`` from that config.
      3. Use ``--solution-dir`` when provided, else ``<config-dir>/solution/<language>/``.

    Returns:
        ``(config_toml_path, solution_source_dir)``
    """
    if config_path:
        cfg_path = resolve_project_path(config_path)
    else:
        cfg_path = PROJECT_ROOT / definition / "config.toml"
        if not cfg_path.exists():
            cfg_path = PROJECT_ROOT / "config.toml"

    cfg = read_config(cfg_path)
    language = cfg["build"]["language"]

    if solution_dir:
        sol_dir = resolve_project_path(solution_dir)
    else:
        candidates = [
            cfg_path.parent / "solution" / language,
            PROJECT_ROOT / definition / "solution" / language,
            PROJECT_ROOT / "solution" / language,
        ]
        for candidate in candidates:
            if candidate.exists():
                sol_dir = candidate
                break
        else:
            raise FileNotFoundError(
                "Solution directory not found. Expected one of: "
                + ", ".join(str(candidate) for candidate in candidates)
            )

    return cfg_path, sol_dir


def pack_solution_for_definition(
    definition: str,
    solution_dir: str | None,
    config_path: str | None,
    python: str,
) -> tuple[str, Path, Path]:
    """Pack a candidate solution via explicit config / source paths.

    Returns:
        A tuple of ``(stdout, config_path, solution_source_dir)``.
    """
    config_path, sol_dir = resolve_config_and_solution(definition, solution_dir, config_path)
    proc = subprocess.run(
        [
            python,
            "scripts/pack_solution.py",
            "--config-path",
            str(config_path),
            "--solution-dir",
            str(sol_dir),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip(), config_path, sol_dir


# ===========================================================================
# Section 3: Modal Result Parsing
# ===========================================================================
# ``run_modal_single.py`` prints benchmark results to stdout in a fixed
# format.  We use regular expressions to extract the structured fields.
# Two patterns are needed:
#   - _RESULT_RE: matches a successful "PASSED" line with latency metrics.
#   - _FAIL_RE:   matches a "FAILED" or "ERROR" status line.
# ===========================================================================

# Regex for a successful result line.  Named groups capture:
#   uuid     – workload UUID
#   status   – "PASSED" (or other success token)
#   latency  – solution latency in milliseconds
#   ref      – reference (torch baseline) latency in milliseconds
#   speedup  – speedup factor (ref / solution)
#   abs_err  – maximum absolute error (optional, only if correctness checked)
#   rel_err  – maximum relative error (optional)
_RESULT_RE = re.compile(
    r"Workload\s+(?P<uuid>[\w-]+):\s+(?P<status>\S+)"
    r"\s*\|\s*(?P<latency>[\d.eE+-]+)\s*ms"
    r"\s*\|\s*ref\s+(?P<ref>[\d.eE+-]+)\s*ms"
    r"\s*\|\s*(?P<speedup>[\d.eE+-]+)x"
    r"(?:\s*\|\s*abs_err=(?P<abs_err>[\d.eE+-]+),\s*rel_err=(?P<rel_err>[\d.eE+-]+))?"
)

# Regex for a failure line (no latency fields).
_FAIL_RE = re.compile(
    r"Workload\s+(?P<uuid>[\w-]+):\s+(?P<status>FAILED|ERROR)"
)

# Lock for thread-safe console/log output from concurrent workers.
_print_lock = threading.Lock()


def run_single_workload(
    uuid: str,
    python: str,
    config_path: str | None = None,
    solution_dir: str | None = None,
    timeout: int = 300,
    warmup_runs: int = 10,
    iterations: int = 50,
    num_trials: int = 3,
    use_isolated_runner: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
    required_matched_ratio: float = 0.0,
) -> dict:
    """Spawn ``modal run scripts/run_modal_single.py`` for one workload.

    This function is executed inside a thread-pool worker.  It launches a
    subprocess that boots a Modal container with a B200 GPU, compiles the
    solution, runs the benchmark for the single workload identified by
    *uuid*, and captures the stdout/stderr output.

    The output is then parsed with the regexes above to extract structured
    metrics.

    Args:
        uuid:    Workload UUID string (passed as ``--workload-uuid``).
        python:  Path to the Python interpreter (must have ``modal`` installed).
        timeout: Maximum wall-clock seconds before the subprocess is killed.
        warmup_runs: Number of warmup runs before timing.
        iterations: Number of timed iterations per trial.
        num_trials: Number of independent trials.
        use_isolated_runner: Use isolated runner (matches official eval).
        atol: Absolute tolerance for correctness (official MoE: 1.0).
        rtol: Relative tolerance for correctness (official MoE: 0.3).
        required_matched_ratio: Required matched ratio (official MoE: 0.9, 0.0 = not set).

    Returns:
        A dict with at least ``"status"`` and, on success, ``"latency_ms"``,
        ``"reference_latency_ms"``, ``"speedup_factor"``, and optionally
        ``"max_abs_error"`` / ``"max_rel_error"``.  On failure, ``"error"``
        contains a diagnostic string.  The raw Modal log is stashed in
        ``"_modal_log"`` (stripped before persisting to JSON).
    """
    # Prefer the current interpreter's Modal module so the local launcher uses
    # the same environment as this script (not an unrelated system install).
    base_cmd = [python, "-m", "modal", "run"]
    cmd = [
        *base_cmd,
        "scripts/run_modal_single.py",
        "--workload-uuid", uuid,
        "--warmup-runs", str(warmup_runs),
        "--iterations", str(iterations),
        "--num-trials", str(num_trials),
        "--atol", str(atol),
        "--rtol", str(rtol),
        "--required-matched-ratio", str(required_matched_ratio),
    ]
    if config_path:
        cmd.extend(["--config-path", config_path])
    if solution_dir:
        cmd.extend(["--solution-dir", solution_dir])
    if use_isolated_runner:
        cmd.append("--use-isolated-runner")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        output = proc.stdout + proc.stderr

        # Try to match a successful result line first.
        m = _RESULT_RE.search(output)
        if m:
            result: dict = {
                "status": m.group("status"),
                "latency_ms": float(m.group("latency")),
                "reference_latency_ms": float(m.group("ref")),
                "speedup_factor": float(m.group("speedup")),
            }
            if m.group("abs_err"):
                result["max_abs_error"] = float(m.group("abs_err"))
            if m.group("rel_err"):
                result["max_rel_error"] = float(m.group("rel_err"))
            # Keep the raw log for debugging but prefix with "_" so it is
            # stripped before writing to the output JSON.
            result["_modal_log"] = output
            return result

        # Fall back to matching a failure line.
        fm = _FAIL_RE.search(output)
        if fm:
            return {"status": fm.group("status"), "error": output[-500:], "_modal_log": output}

        # If neither regex matched, flag it as a parse error.
        return {"status": "PARSE_ERROR", "error": f"exit_code={proc.returncode}", "_modal_log": output}

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": f"timed out after {timeout}s"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ===========================================================================
# Section 4: Dynamic Log Formatting
# ===========================================================================
# The log table adapts its columns to the axes present in the workload
# (e.g. ``batch_size`` for decode, ``total_seq_len + num_seqs`` for prefill,
# ``num_tokens + num_pages`` for DSA, etc.).  This avoids hard-coding
# problem-specific column layouts.
# ===========================================================================


def build_log_header(axes_names: list[str]) -> tuple[str, str]:
    """Build the table header and separator lines for the given axes.

    Each axis gets an 8-character right-aligned column.  The rest of the
    columns (UUID, Status, Latency, Ref, Speedup, errors) are fixed-width.

    Returns:
        ``(header_string, separator_string)``
    """
    axes_cols = "".join(f" {name:>8s}" for name in axes_names)
    header = (
        f"  #                                 UUID"
        f"{axes_cols}"
        f"   {'Status':>6s}"
        f"   {'Latency(ms)':>{LATENCY_COL_WIDTH}s}"
        f"   {'Ref(ms)':>{REFERENCE_COL_WIDTH}s}"
        f"   {'Speedup':>{SPEEDUP_COL_WIDTH}s}"
        f"     abs_err     rel_err"
    )
    sep = "-" * len(header)
    return header, sep


def _fmt_metric(value: float | None, sigfigs: int = DISPLAY_SIGFIGS) -> str:
    """Format floats with enough precision to make small latency deltas visible."""
    if value is None:
        return "n/a"
    return format(float(value), f".{sigfigs}g")


def _fmt_metric_cell(value: float | None, width: int) -> str:
    return f"{_fmt_metric(value):>{width}s}"


def format_log_line(idx: int, w: dict, result: dict, axes_names: list[str]) -> str:
    """Format a single workload result as a fixed-width table row.

    Args:
        idx:        0-based workload index (for ordering in logs).
        w:          Workload dict with ``"uuid"`` and axis values.
        result:     Result dict from ``run_single_workload``.
        axes_names: Axis key names for column rendering.

    Returns:
        A single formatted line string (no trailing newline).
    """
    uuid = w["uuid"]
    # Render each axis value as an 8-char integer column.
    axes_vals = "".join(f" {w.get(name, 0):8d}" for name in axes_names)
    status = result.get("status", "?")

    if status == "PASSED":
        lat = result.get("latency_ms", 0)
        ref = result.get("reference_latency_ms", 0)
        spd = result.get("speedup_factor", 0)
        ae = result.get("max_abs_error")
        re_ = result.get("max_rel_error")
        ae_s = f"{ae:.2e}" if ae is not None else "       N/A"
        re_s = f"{re_:.2e}" if re_ is not None else "       N/A"
        return (
            f"{idx:3d} {uuid}{axes_vals}"
            f"   {status:>6s}"
            f"   {_fmt_metric_cell(lat, LATENCY_COL_WIDTH)}"
            f"   {_fmt_metric_cell(ref, REFERENCE_COL_WIDTH)}"
            f"   {(_fmt_metric(spd) + 'x'):>{SPEEDUP_COL_WIDTH}s}"
            f"    {ae_s}"
            f"    {re_s}"
        )
    else:
        # For non-PASSED results, show a truncated error snippet instead of
        # latency columns (which are unavailable).
        err = result.get("error", "")[:50]
        return f"{idx:3d} {uuid}{axes_vals}   {status:>6s}   {err}"


def progress_bar(done: int, total: int, passed: int, failed: int,
                 elapsed: float, width: int = 40) -> str:
    """Render a carriage-return-prefixed progress bar for stderr.

    The bar overwrites itself in-place on the terminal so that intermediate
    result lines printed to stdout remain readable above it.
    """
    pct = done / total if total else 0
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    eta = (elapsed / done * (total - done)) if done > 0 else 0
    return (
        f"\r[{bar}] {done}/{total} ({pct*100:.0f}%)"
        f"  passed:{passed} failed:{failed}"
        f"  elapsed:{elapsed:.0f}s  ETA:{eta:.0f}s"
    )


# ===========================================================================
# Section 5: Summary / Statistics Helpers
# ===========================================================================


def _percentile(values: list[float], q: float) -> float | None:
    """Compute the *q*-th percentile (0..1) using nearest-rank interpolation.

    Returns ``None`` for an empty input list.
    """
    if not values:
        return None
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


def build_summary(all_results: dict[str, dict]) -> dict:
    """Aggregate per-workload results into a summary statistics dict.

    The output schema matches the format used by ``run_modal.py`` so that
    downstream tooling (archive helpers, round summaries) can consume either
    source interchangeably.

    Computed metrics include avg / median / p95 / min / max for latency,
    reference latency, and speedup, plus pass/fail counts.
    """
    total = len(all_results)
    passed = 0
    lat, ref, spd = [], [], []
    for r in all_results.values():
        if r.get("status") == "PASSED":
            passed += 1
            if r.get("latency_ms") is not None:
                lat.append(float(r["latency_ms"]))
            if r.get("reference_latency_ms") is not None:
                ref.append(float(r["reference_latency_ms"]))
            if r.get("speedup_factor") is not None:
                spd.append(float(r["speedup_factor"]))

    summary: dict = {
        "total_workloads": total,
        "passed_workloads": passed,
        "failed_workloads": total - passed,
        "passed_with_latency": len(lat),
        "passed_with_reference_latency": len(ref),
        "passed_with_speedup": len(spd),
    }
    if lat:
        summary["avg_latency_ms"] = mean(lat)
        summary["median_latency_ms"] = median(lat)
        summary["p95_latency_ms"] = _percentile(lat, 0.95)
        summary["min_latency_ms"] = min(lat)
        summary["max_latency_ms"] = max(lat)
    if ref:
        summary["avg_reference_latency_ms"] = mean(ref)
        summary["median_reference_latency_ms"] = median(ref)
        summary["p95_reference_latency_ms"] = _percentile(ref, 0.95)
    if spd:
        summary["avg_speedup_factor"] = mean(spd)
        summary["median_speedup_factor"] = median(spd)
        summary["p95_speedup_factor"] = _percentile(spd, 0.95)
        summary["min_speedup_factor"] = min(spd)
        summary["max_speedup_factor"] = max(spd)
    return summary


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``Xm YYs`` or ``Xh YYm ZZs``."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


# ===========================================================================
# Section 6: Main Entry Point
# ===========================================================================


def main():
    # ------------------------------------------------------------------
    # 6a. Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Parallel workload evaluation via Modal (supports all contest problems)"
    )
    parser.add_argument(
        "--definition", default=None,
        help="Definition name (e.g. gdn_decode_qk4_v8_d128_k_last). "
             "If omitted, reads from --config-path or root config.toml.",
    )
    parser.add_argument(
        "--solution-dir", default=None,
        help="Path to solution source dir relative to project root "
             "(e.g. gdn_decode_qk4_v8_d128_k_last/solution/cuda). "
             "If omitted, auto-detected from <definition>/solution/<language>/.",
    )
    parser.add_argument(
        "--config-path", default=None,
        help="Path to config.toml relative to project root. "
             "If provided, overrides config auto-discovery.",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Max concurrent Modal containers (default: 10)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for JSON and log files (default: project root)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-workload timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--retry", action="store_true",
        help="Retry mode: skip workloads already PASSED in "
             "--out-dir/benchmark_detailed_results.json",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=None,
        help="Override warmup runs (default: 10, or 1 for gdn_prefill)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override iterations (default: 50, or 5 for gdn_prefill)",
    )
    parser.add_argument(
        "--num-trials", type=int, default=None,
        help="Override number of trials (default: 3)",
    )
    parser.add_argument(
        "--use-isolated-runner", action="store_true", default=True,
        help="Use isolated runner (default: True, matches official eval)",
    )
    parser.add_argument(
        "--no-isolated-runner", action="store_false", dest="use_isolated_runner",
        help="Disable isolated runner",
    )
    parser.add_argument(
        "--atol", type=float, default=None,
        help="Override absolute tolerance for correctness (official MoE: 1.0)",
    )
    parser.add_argument(
        "--rtol", type=float, default=None,
        help="Override relative tolerance for correctness (official MoE: 0.3)",
    )
    parser.add_argument(
        "--required-matched-ratio", type=float, default=None,
        help="Override required matched ratio (official MoE: 0.9)",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter path (must have 'modal' installed)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 6b. Resolve which definition to benchmark
    # ------------------------------------------------------------------
    if args.definition:
        definition = args.definition
    elif args.config_path:
        definition = read_definition_from_config(resolve_project_path(args.config_path))
    else:
        root_config = PROJECT_ROOT / "config.toml"
        if not root_config.exists():
            raise FileNotFoundError(
                "No root config.toml exists in this release layout. "
                "Pass --definition <definition> or --config-path <definition>/config.toml."
            )
        definition = read_definition_from_config(root_config)

    # Resolve benchmark parameters (definition-aware defaults matching official eval).
    _resolved = resolve_official_benchmark_kwargs(definition)
    bench_warmup = args.warmup_runs if args.warmup_runs is not None else _resolved["warmup_runs"]
    bench_iterations = args.iterations if args.iterations is not None else _resolved["iterations"]
    bench_num_trials = args.num_trials if args.num_trials is not None else _resolved["num_trials"]
    bench_use_isolated_runner = args.use_isolated_runner
    bench_atol = args.atol if args.atol is not None else _resolved.get("atol", 0.01)
    bench_rtol = args.rtol if args.rtol is not None else _resolved.get("rtol", 0.01)
    bench_required_matched_ratio = args.required_matched_ratio if args.required_matched_ratio is not None else _resolved.get("required_matched_ratio", 0.0)

    # Prepare output paths.
    out_dir = PROJECT_ROOT / (args.out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_detailed_results.json"
    log_path = out_dir / "retained_run.log"

    # ------------------------------------------------------------------
    # 6c. Pack the solution into solution.json
    # ------------------------------------------------------------------
    # This step compiles the CUDA/Triton source files into a portable JSON
    # bundle that Modal containers can deserialize and build on the fly.
    print(f"Packing solution for: {definition}")
    pack_output, resolved_config_path, resolved_solution_dir = pack_solution_for_definition(
        definition, args.solution_dir, args.config_path, args.python
    )
    print(pack_output)

    # ------------------------------------------------------------------
    # 6d. Load workload metadata
    # ------------------------------------------------------------------
    # We only need the UUID and axes here; the actual tensor data lives on
    # the Modal volume and is loaded inside the container at runtime.
    workloads, axes_names = load_workloads(definition)
    total = len(workloads)
    LOG_HEADER, LOG_SEP = build_log_header(axes_names)

    # ------------------------------------------------------------------
    # 6e. Retry mode: load prior results and skip already-PASSED workloads
    # ------------------------------------------------------------------
    # This is useful when a previous run was interrupted or had transient
    # failures.  We reload the JSON, mark PASSED workloads as skipped, and
    # only dispatch the remaining ones to Modal.
    all_results: dict[str, dict] = {}
    skipped_indices: set[int] = set()

    if args.retry and json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
            all_results = existing["results"].get(definition, {})
        except Exception:
            all_results = {}

        for i, w in enumerate(workloads):
            prev = all_results.get(w["uuid"])
            if prev and prev.get("status") == "PASSED":
                skipped_indices.add(i)

    to_run = total - len(skipped_indices)

    # Rough time estimate: ~90s per workload divided across workers.
    est_per_workload = 90
    est_total = est_per_workload * max(to_run, 1) / args.workers
    start_time = datetime.now(timezone.utc)

    # Print run plan to console.
    print(f"\nDefinition: {definition}")
    print(f"Config:     {resolved_config_path}")
    print(f"Solution:   {resolved_solution_dir}")
    print(f"Workloads:  {total} total, {to_run} to run, {len(skipped_indices)} skipped (already PASSED)")
    print(f"Axes:       {', '.join(axes_names)}")
    print(f"Benchmark:  warmup={bench_warmup}, iter={bench_iterations}, trials={bench_num_trials}, isolated={bench_use_isolated_runner}")
    print(f"Tolerance:  atol={bench_atol}, rtol={bench_rtol}, required_matched_ratio={bench_required_matched_ratio or 'default'}")
    print(f"Workers:    {args.workers}")
    print(f"Estimated:  {fmt_duration(est_total)}")
    print(f"Started:    {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # ------------------------------------------------------------------
    # 6f. Open the log file and write the header
    # ------------------------------------------------------------------
    is_retry = args.retry and len(skipped_indices) > 0
    log_f = open(log_path, "a" if is_retry else "w")
    if is_retry:
        log_f.write(f"\n=== Retry started at {start_time.isoformat()} ===\n")
        log_f.write(f"Definition: {definition}\n")
        log_f.write(f"Skipped {len(skipped_indices)} already-PASSED workloads, running {to_run}\n\n")
    else:
        log_f.write(f"=== Run started at {start_time.isoformat()} ===\n")
        log_f.write(f"Definition: {definition}\n")
        log_f.write(f"Solution:  {pack_output}\n")
        log_f.write(f"Workers:   {args.workers}\n")
        log_f.write(f"Estimated: {fmt_duration(est_total)}\n\n")
    log_f.write(LOG_HEADER + "\n")
    log_f.write(LOG_SEP + "\n")
    log_f.flush()

    # Print the same header to the console.
    print(LOG_HEADER)
    print(LOG_SEP)

    # ------------------------------------------------------------------
    # 6g. Initialize counters (pre-populated from skipped/retry results)
    # ------------------------------------------------------------------
    result_lines: list[tuple[int, str]] = []  # (index, formatted_line) for ordered log
    done = 0
    passed = 0
    failed = 0
    latencies: list[float] = []
    speedups: list[float] = []
    per_workload_times: list[float] = []  # wall-clock seconds per dispatched workload

    # Count skipped (already-PASSED) workloads into the totals so that the
    # progress bar starts at the right position.
    for i in skipped_indices:
        w = workloads[i]
        r = all_results[w["uuid"]]
        passed += 1
        done += 1
        if r.get("latency_ms"):
            latencies.append(float(r["latency_ms"]))
        if r.get("speedup_factor"):
            speedups.append(float(r["speedup_factor"]))
        result_lines.append((i, format_log_line(i, w, r, axes_names)))

    t_start = time.time()

    # ------------------------------------------------------------------
    # 6h. Callback invoked when a worker thread finishes a workload
    # ------------------------------------------------------------------
    def on_result(idx: int, w: dict, result: dict, wall_s: float):
        """Process a completed workload result.

        This callback is invoked from the main thread (inside the
        ``as_completed`` loop).  It updates the global counters, writes to
        the log file, refreshes the progress bar, and saves an intermediate
        JSON snapshot so that partial results survive crashes.
        """
        nonlocal done, passed, failed
        done += 1
        per_workload_times.append(wall_s)
        status = result.get("status", "?")
        if status == "PASSED":
            passed += 1
            if result.get("latency_ms"):
                latencies.append(float(result["latency_ms"]))
            if result.get("speedup_factor"):
                speedups.append(float(result["speedup_factor"]))
        else:
            failed += 1

        # Strip internal fields (prefixed with "_") before persisting.
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        all_results[w["uuid"]] = clean

        line = format_log_line(idx, w, result, axes_names)
        result_lines.append((idx, line))

        with _print_lock:
            # Clear the progress bar, print the result line, then redraw
            # the bar below it.  This keeps results scrolling upward while
            # the bar stays at the bottom of the terminal.
            sys.stderr.write("\r" + " " * 120 + "\r")
            sys.stderr.flush()
            print(line)
            elapsed = time.time() - t_start
            bar = progress_bar(done, total, passed, failed, elapsed)
            sys.stderr.write(bar)
            sys.stderr.flush()

            # Append to the human-readable log (arrival order).
            log_f.write(line + "\n")
            log_f.flush()

            # Write an intermediate JSON snapshot so that partial results
            # are recoverable if the process is killed or interrupted.
            summary = build_summary(all_results)
            report = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "definition": definition,
                "results": {definition: all_results},
                "summary": {definition: summary},
                "meta": {
                    "completed": done, "total": total,
                    "passed": passed, "failed": failed,
                },
            }
            json_path.write_text(json.dumps(report, indent=2))

    # ------------------------------------------------------------------
    # 6i. Dispatch workloads to the thread pool
    # ------------------------------------------------------------------
    # Each worker thread spawns a subprocess that calls ``modal run``,
    # which in turn boots a cloud container on a B200 GPU.  The thread
    # pool size (``--workers``) limits the number of simultaneous Modal
    # containers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[concurrent.futures.Future, tuple[int, dict, float]] = {}
        for i, w in enumerate(workloads):
            if i in skipped_indices:
                continue
            fut = executor.submit(
                run_single_workload,
                w["uuid"],
                args.python,
                str(resolved_config_path),
                str(resolved_solution_dir),
                args.timeout,
                warmup_runs=bench_warmup,
                iterations=bench_iterations,
                num_trials=bench_num_trials,
                use_isolated_runner=bench_use_isolated_runner,
                atol=bench_atol,
                rtol=bench_rtol,
                required_matched_ratio=bench_required_matched_ratio,
            )
            futures[fut] = (i, w, time.time())

        # Show the initial (empty) progress bar.
        sys.stderr.write(progress_bar(done, total, passed, failed, 0))
        sys.stderr.flush()

        # Process results as they arrive (not necessarily in submission order).
        for fut in concurrent.futures.as_completed(futures):
            idx, w, t0 = futures[fut]
            wall_s = time.time() - t0
            try:
                result = fut.result()
            except Exception as e:
                result = {"status": "ERROR", "error": str(e)}
            on_result(idx, w, result, wall_s)

    # Clear the progress bar after all workloads are done.
    sys.stderr.write("\r" + " " * 120 + "\r")
    sys.stderr.flush()

    # ------------------------------------------------------------------
    # 6j. Print and persist the final summary
    # ------------------------------------------------------------------
    end_time = datetime.now(timezone.utc)
    elapsed_total = time.time() - t_start
    avg_wall = mean(per_workload_times) if per_workload_times else 0

    skipped_count = len(skipped_indices)
    summary = build_summary(all_results)
    s = summary  # shorthand alias

    summary_lines = [
        LOG_SEP,
        f"DEFINITION: {definition}",
        f"RESULT: {s['passed_workloads']}/{s['total_workloads']} PASSED  |  {s['failed_workloads']} FAILED"
        + (f"  |  (skipped {skipped_count} already-PASSED, ran {to_run})" if skipped_count else ""),
        LOG_SEP,
        "",
        "LATENCY:",
        f"  Avg latency:    {_fmt_metric(s.get('avg_latency_ms'))} ms",
        f"  Median latency: {_fmt_metric(s.get('median_latency_ms'))} ms",
        f"  P95 latency:    {_fmt_metric(s.get('p95_latency_ms'))} ms",
        f"  Min latency:    {_fmt_metric(s.get('min_latency_ms'))} ms",
        f"  Max latency:    {_fmt_metric(s.get('max_latency_ms'))} ms",
        "",
        "REFERENCE LATENCY:",
        f"  Avg ref:        {_fmt_metric(s.get('avg_reference_latency_ms'))} ms",
        f"  Median ref:     {_fmt_metric(s.get('median_reference_latency_ms'))} ms",
        f"  P95 ref:        {_fmt_metric(s.get('p95_reference_latency_ms'))} ms",
        "",
        "SPEEDUP:",
        f"  Avg speedup:    {_fmt_metric(s.get('avg_speedup_factor'))}x",
        f"  Median speedup: {_fmt_metric(s.get('median_speedup_factor'))}x",
        f"  P95 speedup:    {_fmt_metric(s.get('p95_speedup_factor'))}x",
        f"  Min speedup:    {_fmt_metric(s.get('min_speedup_factor'))}x",
        f"  Max speedup:    {_fmt_metric(s.get('max_speedup_factor'))}x",
        "",
        "TIMING:",
        f"  Estimated time: {fmt_duration(est_total)}",
        f"  Real time:      {fmt_duration(elapsed_total)}",
        f"  Avg per workload (wall): {avg_wall:.1f}s",
        f"  Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"  Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        LOG_SEP,
    ]

    # Print summary to console.
    for line in summary_lines:
        print(line)

    # Append summary to the log file.
    log_f.write("\n")
    for line in summary_lines:
        log_f.write(line + "\n")

    # Append an index-ordered result table to the log for easy scanning.
    # (The main log body above is in arrival order, which is non-deterministic.)
    log_f.write("\n\n=== Results ordered by workload index ===\n")
    log_f.write(LOG_HEADER + "\n")
    log_f.write(LOG_SEP + "\n")
    for idx, line in sorted(result_lines, key=lambda x: x[0]):
        log_f.write(line + "\n")
    log_f.write(LOG_SEP + "\n")

    log_f.write(f"\n=== Run finished at {end_time.isoformat()} ===\n")
    log_f.close()

    # Write the final JSON report (overwrites the last intermediate snapshot).
    final_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": definition,
        "results": {definition: all_results},
        "summary": {definition: summary},
        "meta": {
            "completed": done, "total": total,
            "passed": s["passed_workloads"], "failed": s["failed_workloads"],
            "estimated_time_s": est_total,
            "real_time_s": elapsed_total,
            "started": start_time.isoformat(),
            "finished": end_time.isoformat(),
        },
    }
    json_path.write_text(json.dumps(final_report, indent=2))

    print(f"\nResults: {json_path}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()
