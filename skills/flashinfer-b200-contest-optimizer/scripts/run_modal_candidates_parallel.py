#!/usr/bin/env python3
"""
Parallel Modal evaluator for FlashInfer candidate kernels.

This script evaluates multiple candidate kernels concurrently by creating
isolated workspace copies, then running the workspace's current
`scripts/run_modal.py` in each copy.

Workflow overview:
  1. Read a JSON file listing candidate kernels (each with a name, kernel
     source path, solution metadata, etc.).
  2. For every candidate, rsync-clone the project tree into a temporary
     working directory, drop the candidate's kernel.cu into the clone,
     write a matching config.toml, and launch `modal run scripts/run_modal.py`
     inside that clone.
  3. All candidates run in parallel via a ThreadPoolExecutor (controlled by
     --max-workers).
  4. After all runs finish, aggregate benchmark results, extract any
     torch-profiler and NCU-profiler artifacts, and write a combined
     summary (JSON + Markdown) into the output directory.

Exit code: 0 if every candidate passed, 1 if any candidate failed or
partially failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


# ---------------------------------------------------------------------------
# Data classes – typed containers for candidate metadata and profiling options
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """Describes a single kernel candidate to be evaluated.

    Attributes:
        name:          Human-readable label for this candidate (used in logs
                       and report filenames).
        kernel_path:   Absolute path to the .cu source file on the local
                       filesystem.
        solution_name: Name written into config.toml's [solution] section so
                       the Modal runner can identify this solution.
        definition:    The operator definition name that the benchmark
                       framework should load (e.g.
                       "gdn_decode_qk4_v8_d128_k_last").
        author:        Author field written to config.toml.
        entry_point:   Build entry point in the form "file::symbol" that the
                       CUDA build system uses to locate the kernel.
        binding:       Binding type for the compiled kernel (e.g. "tvm-ffi").
    """
    name: str
    kernel_path: Path
    solution_name: str
    definition: str
    author: str
    entry_point: str = "kernel.cu::kernel_cuda_v4"
    binding: str = "tvm-ffi"


@dataclass(frozen=True)
class ProfileOptions:
    """Configuration knobs forwarded to the Modal runner for torch.profiler.

    Attributes:
        enabled:       Whether torch profiling is turned on.
        workload_ids:  Comma-separated list of workload UUIDs to profile
                       (empty string means "all eligible").
        max_workloads: Cap on how many workloads to profile (0 = unlimited).
        sort_by:       Column name used to sort profiler table rows.
        row_limit:     Maximum rows shown in the profiler output table.
        warmup_runs:   Number of warm-up iterations before profiling starts.
    """
    enabled: bool = True
    workload_ids: str = ""
    max_workloads: int = 0
    sort_by: str = "cuda_time_total"
    row_limit: int = 40
    warmup_runs: int = 3


@dataclass(frozen=True)
class NcuOptions:
    """Configuration knobs forwarded to the Modal runner for NVIDIA Nsight
    Compute (NCU) profiling.

    Attributes:
        enabled:       Whether NCU profiling is turned on.
        workload_ids:  Comma-separated workload UUIDs to profile (empty =
                       all eligible).
        max_workloads: Cap on workloads to profile (0 = unlimited).
        ncu_set:       NCU metric set to collect (e.g. "detailed", "full").
        ncu_page:      NCU report page to render (e.g. "details", "raw").
        ncu_timeout:   Per-workload NCU timeout in seconds.
        max_lines:     Max output lines to capture from NCU stdout.
        kernel_name:   Optional regex filter for which GPU kernel(s) NCU
                       should instrument (empty = instrument all).
    """
    enabled: bool = False
    workload_ids: str = ""
    max_workloads: int = 0
    ncu_set: str = "detailed"
    ncu_page: str = "details"
    ncu_timeout: int = 240
    max_lines: int = 220
    kernel_name: str = ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Convert an arbitrary string into a filesystem-safe token.

    Replaces any character that is not alphanumeric, hyphen, dot, or
    underscore with a hyphen, then strips leading/trailing hyphens.
    """
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in name).strip("-")


def _percentile(values: list[float], q: float) -> float | None:
    """Compute the q-th percentile (0.0–1.0) of a list of floats.

    Uses nearest-rank interpolation. Returns None for empty input.
    """
    if not values:
        return None
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


# ---------------------------------------------------------------------------
# Candidate loading from JSON
# ---------------------------------------------------------------------------

def _load_candidates(path: Path) -> list[Candidate]:
    """Parse the candidates JSON file and return a list of Candidate objects.

    The JSON file must be a non-empty list of objects.  Each object must
    contain at least:
        - name          (str)
        - kernel_path   (str – path to the .cu file, may contain ~)
        - solution_name (str)

    Optional fields (with defaults):
        - definition   (default "gdn_decode_qk4_v8_d128_k_last")
        - author       (default "yue-shui")
        - entry_point  (default "kernel.cu::kernel_cuda_v4")
        - binding      (default "tvm-ffi")

    Raises:
        ValueError:        If the file doesn't contain a non-empty list or a
                           required field is missing.
        FileNotFoundError: If a candidate's kernel_path does not exist.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("candidates json must be a non-empty list")

    out: list[Candidate] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"candidate[{idx}] is not an object")
        try:
            c = Candidate(
                name=str(item["name"]),
                kernel_path=Path(item["kernel_path"]).expanduser().resolve(),
                solution_name=str(item["solution_name"]),
                definition=str(item.get("definition", "gdn_decode_qk4_v8_d128_k_last")),
                author=str(item.get("author", "yue-shui")),
                entry_point=str(item.get("entry_point", "kernel.cu::kernel_cuda_v4")),
                binding=str(item.get("binding", "tvm-ffi")),
            )
        except KeyError as e:
            raise ValueError(f"candidate[{idx}] missing required field: {e}") from e

        if not c.kernel_path.exists():
            raise FileNotFoundError(f"kernel not found: {c.kernel_path}")
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Workspace isolation – each candidate gets its own copy of the project
# ---------------------------------------------------------------------------

def _copy_workspace(project_root: Path, dst: Path) -> None:
    """Create an isolated workspace clone of the project at *dst* using rsync.

    The clone excludes heavyweight directories that are not needed for the
    Modal run (.git, __pycache__, and several large reference/contest
    sub-trees) to speed up the copy.
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-a",                   # archive mode: preserves permissions, timestamps, etc.
        "--delete",             # remove extraneous files from the destination
        "--exclude=.git",
        "--exclude=.DS_Store",
        "--exclude=__pycache__",
        "--exclude=flashinfer-competition-codebase-reference",
        "--exclude=mlsys26-agent-baseline",
        "--exclude=mlsys26-contest",
        str(project_root) + "/",
        str(dst) + "/",
    ]
    subprocess.run(cmd, check=True)


def _write_config(workdir: Path, c: Candidate) -> None:
    """Write a config.toml for the candidate into the workspace clone.

    The config is written to both the project root *and*
    ``solution/cuda/config.toml`` so that every layer of the build /
    evaluation pipeline can find it.
    """
    config = (
        "[solution]\n"
        f'name = "{c.solution_name}"\n'
        f'definition = "{c.definition}"\n'
        f'author = "{c.author}"\n\n'
        "[build]\n"
        'language = "cuda"\n'
        f'entry_point = "{c.entry_point}"\n'
        f'binding = "{c.binding}"\n'
    )
    (workdir / "config.toml").write_text(config, encoding="utf-8")
    (workdir / "solution" / "cuda" / "config.toml").write_text(config, encoding="utf-8")


# ---------------------------------------------------------------------------
# Benchmark JSON aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_from_benchmark_json(path: Path) -> dict[str, Any]:
    """Read a benchmark_detailed_results.json and compute aggregate metrics.

    Iterates over all trace entries for the first (and typically only)
    definition.  Collects latency, reference latency, and speedup numbers
    for every PASSED workload, then returns a dict with:
        - definition, total_workloads, passed_workloads, failed_workloads
        - avg / median / p95 / min / max latency_ms
        - avg_reference_latency_ms, avg_speedup_factor
    """
    payload = _load_benchmark_payload(path)
    # The benchmark JSON nests results under a definition name key.
    def_name = next(iter(payload["results"].keys()))
    traces = payload["results"][def_name]

    lat = []   # latencies of passed workloads
    ref = []   # reference latencies of passed workloads
    spd = []   # speedup factors of passed workloads
    passed = 0
    for item in traces.values():
        if item.get("status") == "PASSED":
            passed += 1
            if item.get("latency_ms") is not None:
                lat.append(float(item["latency_ms"]))
            if item.get("reference_latency_ms") is not None:
                ref.append(float(item["reference_latency_ms"]))
            if item.get("speedup_factor") is not None:
                spd.append(float(item["speedup_factor"]))

    return {
        "definition": def_name,
        "total_workloads": len(traces),
        "passed_workloads": passed,
        "failed_workloads": len(traces) - passed,
        "avg_latency_ms": mean(lat) if lat else None,
        "median_latency_ms": median(lat) if lat else None,
        "p95_latency_ms": _percentile(lat, 0.95),
        "min_latency_ms": min(lat) if lat else None,
        "max_latency_ms": max(lat) if lat else None,
        "avg_reference_latency_ms": mean(ref) if ref else None,
        "avg_speedup_factor": mean(spd) if spd else None,
    }


def _load_benchmark_payload(path: Path) -> dict[str, Any]:
    """Load and return the full JSON payload from a benchmark results file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Torch profiler artifact extraction
# ---------------------------------------------------------------------------

def _render_profile_report(profile_report: dict[str, Any]) -> str:
    """Convert a torch-profiler report dict into a human-readable text block.

    The output starts with a summary line (profiled count, missing count,
    sort column, row limit), followed by per-workload profiler tables.
    """
    lines = []
    rows = profile_report.get("rows", [])
    lines.append("Torch Profiler:")
    lines.append(
        "  Summary:"
        f" profiled={len(rows)}"
        f" | missing={len(profile_report.get('missing_workloads', []))}"
        f" | sort_by={profile_report.get('sort_by')}"
        f" | row_limit={profile_report.get('row_limit')}"
    )

    for row in rows:
        # Show the first 8 characters of the workload UUID and its axes
        lines.append(f"  Workload {row['workload_uuid'][:8]}... | axes={row.get('axes', {})}")
        table = (row.get("table") or "").strip()
        if not table:
            lines.append("    <empty profiler output>")
            continue
        for line in table.splitlines():
            lines.append(f"    {line}")

    # List any workloads that were requested but not found in the trace set
    for workload_id in profile_report.get("missing_workloads", []):
        lines.append(f"  Workload {workload_id[:8]}... | missing from trace set")

    return "\n".join(lines) + "\n"


def _extract_profile_artifacts(
    benchmark_payload: dict[str, Any],
    out_dir: Path,
    safe_name: str,
) -> dict[str, Any]:
    """Extract torch-profiler data from the benchmark payload and persist it.

    Writes two files to *out_dir*:
        - <safe_name>.torch_profile.json  (raw profiler JSON)
        - <safe_name>.torch_profile.txt   (rendered human-readable text)

    Returns a dict of metadata keys that will be merged into the
    candidate's result row.  Returns an empty dict if no torch_profile
    section exists in the payload.
    """
    profile_report = benchmark_payload.get("torch_profile")
    if not profile_report:
        return {}

    profile_json_path = out_dir / f"{safe_name}.torch_profile.json"
    profile_txt_path = out_dir / f"{safe_name}.torch_profile.txt"
    profile_json_path.write_text(
        json.dumps(profile_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    profile_txt_path.write_text(_render_profile_report(profile_report), encoding="utf-8")

    return {
        "torch_profile_json": str(profile_json_path),
        "torch_profile_text": str(profile_txt_path),
        "profiled_torch_workloads": profile_report.get("profiled_workloads", []),
        "missing_torch_workloads": profile_report.get("missing_workloads", []),
        "torch_profiler": profile_report.get("profiler", "torch.profiler"),
    }


# ---------------------------------------------------------------------------
# NCU profiler artifact extraction
# ---------------------------------------------------------------------------

def _render_ncu_markdown_report(ncu_report: dict[str, Any]) -> str:
    """Render an NCU profiling report dict as a Markdown document.

    Produces a top-level heading, metadata bullet list, per-workload
    sections with fenced code blocks for NCU output, and a trailing
    section listing any missing workloads.
    """
    lines = [
        "# NCU Profiling Report",
        "",
        f"- **NCU Set**: {ncu_report.get('ncu_set', 'n/a')}",
        f"- **NCU Page**: {ncu_report.get('ncu_page', 'n/a')}",
        f"- **Kernel Filter**: {ncu_report.get('kernel_name') or 'all'}",
        f"- **Timeout**: {ncu_report.get('timeout', 'n/a')}s",
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

    return "\n".join(lines) + "\n"


def _extract_ncu_artifacts(
    benchmark_payload: dict[str, Any],
    run_dir: Path,
    out_dir: Path,
    safe_name: str,
) -> dict[str, Any]:
    """Extract NCU profiling data from the benchmark payload and persist it.

    Writes two files to *out_dir*:
        - <safe_name>.ncu_profile.json  (raw NCU JSON)
        - <safe_name>.ncu_profile.md    (Markdown report – either copied
          from the run directory if the Modal runner already generated it,
          or rendered from the JSON data)

    Returns a dict of metadata keys merged into the candidate result row,
    or an empty dict if no ncu_profile section exists.
    """
    ncu_report = benchmark_payload.get("ncu_profile")
    if not ncu_report:
        return {}

    ncu_json_path = out_dir / f"{safe_name}.ncu_profile.json"
    ncu_md_path = out_dir / f"{safe_name}.ncu_profile.md"
    ncu_json_path.write_text(
        json.dumps(ncu_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Prefer the Markdown report already generated by the Modal runner;
    # fall back to rendering one ourselves from the JSON data.
    source_md = run_dir / "ncu_profile_report.md"
    if source_md.exists():
        shutil.copy2(source_md, ncu_md_path)
    else:
        ncu_md_path.write_text(_render_ncu_markdown_report(ncu_report), encoding="utf-8")

    return {
        "ncu_profile_json": str(ncu_json_path),
        "ncu_profile_markdown": str(ncu_md_path),
        "profiled_ncu_workloads": ncu_report.get("profiled_workloads", []),
        "missing_ncu_workloads": ncu_report.get("missing_workloads", []),
        "ncu_profiler": ncu_report.get("profiler", "ncu"),
        "ncu_set": ncu_report.get("ncu_set"),
        "ncu_page": ncu_report.get("ncu_page"),
        "ncu_kernel_name": ncu_report.get("kernel_name"),
    }


# ---------------------------------------------------------------------------
# Modal URL extraction helper
# ---------------------------------------------------------------------------

def _extract_modal_url(stdout: str) -> str:
    """Scan the Modal runner's stdout for a Modal app URL and return it.

    The Modal CLI typically prints a line like
    ``https://modal.com/apps/...`` which links to the run dashboard.
    Returns an empty string if no such line is found.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("https://modal.com/apps/"):
            return line
    return ""


# ---------------------------------------------------------------------------
# Core per-candidate evaluation logic
# ---------------------------------------------------------------------------

def _run_candidate(
    project_root: Path,
    work_root: Path,
    out_dir: Path,
    conda_env: str,
    timeout_s: int,
    keep_workdir: bool,
    profile_options: ProfileOptions,
    ncu_options: NcuOptions,
    c: Candidate,
) -> dict[str, Any]:
    """Evaluate a single candidate kernel end-to-end.

    Steps performed:
      1. Clone the project tree into ``work_root/<sanitized-name>``.
      2. Copy the candidate's kernel.cu into the clone's solution/cuda/.
      3. Write a matching config.toml.
      4. Build the ``modal run`` command, forwarding all profile / NCU
         options from the caller.
      5. Execute the command via ``conda run -n <env>``, capturing
         stdout + stderr.
      6. If the run succeeded and benchmark_detailed_results.json exists,
         aggregate metrics and extract profiler artifacts.
      7. Clean up the temporary workspace (unless --keep-workdir).

    Returns:
        A dict describing the outcome, including status (PASSED /
        PARTIAL / FAILED), aggregate latency metrics, paths to saved
        artifacts, and the Modal run URL.
    """
    safe_name = _sanitize(c.name)
    run_dir = work_root / safe_name           # isolated workspace clone
    log_path = out_dir / f"{safe_name}.modal.log"
    bench_copy = out_dir / f"{safe_name}.benchmark.json"

    try:
        # ---- 1. Prepare the workspace ----
        _copy_workspace(project_root, run_dir)
        shutil.copy2(c.kernel_path, run_dir / "solution" / "cuda" / "kernel.cu")
        _write_config(run_dir, c)

        # ---- 2. Build the modal run command ----
        cmd = ["conda", "run", "-n", conda_env, "modal", "run", "scripts/run_modal.py"]

        # Torch profiler flags
        cmd.append("--profile-torch" if profile_options.enabled else "--no-profile-torch")
        if profile_options.workload_ids:
            cmd.extend(["--profile-workload-ids", profile_options.workload_ids])
        if profile_options.max_workloads != 0:
            cmd.extend(["--profile-max-workloads", str(profile_options.max_workloads)])
        if profile_options.sort_by != "cuda_time_total":
            cmd.extend(["--profile-sort-by", profile_options.sort_by])
        if profile_options.row_limit != 40:
            cmd.extend(["--profile-row-limit", str(profile_options.row_limit)])
        if profile_options.warmup_runs != 3:
            cmd.extend(["--profile-warmup-runs", str(profile_options.warmup_runs)])

        # NCU profiler flags
        if ncu_options.enabled:
            cmd.append("--profile-ncu")
        if ncu_options.workload_ids:
            cmd.extend(["--ncu-workload-ids", ncu_options.workload_ids])
        if ncu_options.max_workloads != 0:
            cmd.extend(["--ncu-max-workloads", str(ncu_options.max_workloads)])
        if ncu_options.ncu_set != "detailed":
            cmd.extend(["--ncu-set", ncu_options.ncu_set])
        if ncu_options.ncu_page != "details":
            cmd.extend(["--ncu-page", ncu_options.ncu_page])
        if ncu_options.ncu_timeout != 240:
            cmd.extend(["--ncu-timeout", str(ncu_options.ncu_timeout)])
        if ncu_options.max_lines != 220:
            cmd.extend(["--ncu-max-lines", str(ncu_options.max_lines)])
        if ncu_options.kernel_name:
            cmd.extend(["--ncu-kernel-name", ncu_options.kernel_name])

        # ---- 3. Execute ----
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,       # we handle non-zero return codes ourselves
        )
        stdout = proc.stdout
        stderr = proc.stderr
        # Persist the full log for later debugging
        log_path.write_text(stdout + "\n" + stderr, encoding="utf-8")

        # ---- 4. Build the initial result dict ----
        result = {
            "candidate": c.name,
            "status": "FAILED",
            "return_code": proc.returncode,
            "modal_run_url": _extract_modal_url(stdout),
            "workdir": str(run_dir),
            "log_path": str(log_path),
        }

        # ---- 5. If benchmark succeeded, aggregate metrics & artifacts ----
        bench_json = run_dir / "benchmark_detailed_results.json"
        if proc.returncode == 0 and bench_json.exists():
            shutil.copy2(bench_json, bench_copy)
            bench_payload = _load_benchmark_payload(bench_json)
            agg = _aggregate_from_benchmark_json(bench_json)
            result.update(agg)
            result.update(_extract_profile_artifacts(bench_payload, out_dir, safe_name))
            result.update(_extract_ncu_artifacts(bench_payload, run_dir, out_dir, safe_name))
            # PASSED = all workloads passed; PARTIAL = some failed
            result["status"] = "PASSED" if agg["failed_workloads"] == 0 else "PARTIAL"
            result["benchmark_json"] = str(bench_copy)
        else:
            # Capture the last 40 lines of output for quick error diagnosis
            tail = "\n".join((stdout + "\n" + stderr).splitlines()[-40:])
            result["error_tail"] = tail

        return result
    finally:
        # ---- 6. Clean up ----
        if not keep_workdir and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Summary / report generation
# ---------------------------------------------------------------------------

def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Produce the combined parallel-evaluation summary files.

    Outputs:
        parallel_results.json         – full JSON array of all candidate
                                        results.
        parallel_results.md           – Markdown table ranked by average
                                        latency (lowest = best).
        parallel_torch_profiles.json  – index of all torch-profiler
                                        artifacts across candidates.
        parallel_ncu_profiles.json    – index of all NCU-profiler artifacts
                                        across candidates.
    """
    out_json = out_dir / "parallel_results.json"
    profile_bundle_path = out_dir / "parallel_torch_profiles.json"
    ncu_bundle_path = out_dir / "parallel_ncu_profiles.json"

    # --- 1. Main results JSON ---
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": rows,
        "torch_profile_bundle": str(profile_bundle_path),
        "ncu_profile_bundle": str(ncu_bundle_path),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 2. Torch profiler bundle JSON ---
    # Collects per-candidate profiler paths into a single index file so
    # downstream tooling can iterate over all profiles without parsing
    # parallel_results.json.
    profile_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profiles": [
            {
                "candidate": row.get("candidate"),
                "status": row.get("status"),
                "benchmark_json": row.get("benchmark_json"),
                "torch_profile_json": row.get("torch_profile_json"),
                "torch_profile_text": row.get("torch_profile_text"),
                "profiled_workloads": row.get("profiled_torch_workloads", []),
                "missing_profile_workloads": row.get("missing_torch_workloads", []),
            }
            for row in rows
            if row.get("torch_profile_json")
        ],
    }
    profile_bundle_path.write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- 3. NCU profiler bundle JSON ---
    ncu_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profiles": [
            {
                "candidate": row.get("candidate"),
                "status": row.get("status"),
                "benchmark_json": row.get("benchmark_json"),
                "ncu_profile_json": row.get("ncu_profile_json"),
                "ncu_profile_markdown": row.get("ncu_profile_markdown"),
                "profiled_workloads": row.get("profiled_ncu_workloads", []),
                "missing_profile_workloads": row.get("missing_ncu_workloads", []),
                "ncu_set": row.get("ncu_set"),
                "ncu_page": row.get("ncu_page"),
                "ncu_kernel_name": row.get("ncu_kernel_name"),
            }
            for row in rows
            if row.get("ncu_profile_json")
        ],
    }
    ncu_bundle_path.write_text(
        json.dumps(ncu_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- 4. Markdown leaderboard ---
    # Sort candidates by average latency ascending; candidates with no
    # latency data (i.e. failures) are pushed to the bottom.
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("avg_latency_ms") is None,       # None → sorts last
            math.inf if r.get("avg_latency_ms") is None else r["avg_latency_ms"],
        ),
    )

    md = []
    md.append("# Parallel Modal Candidate Results")
    md.append("")
    md.append("| rank | candidate | status | avg latency (ms) | median (ms) | p95 (ms) | avg speedup | passed | torch profile | ncu profile | modal url |")
    md.append("|---:|---|---|---:|---:|---:|---:|---:|---|---|---|")

    rank = 0
    for r in sorted_rows:
        # Only assign a numeric rank to candidates that produced latency data
        if r.get("avg_latency_ms") is not None:
            rank += 1
            rank_str = str(rank)
        else:
            rank_str = "-"

        # Build a concise torch-profiler summary cell
        profile_summary = "-"
        if r.get("torch_profile_json"):
            profile_summary = (
                f"{len(r.get('profiled_torch_workloads', []))} profiled, "
                f"{len(r.get('missing_torch_workloads', []))} missing"
            )

        # Build a concise NCU-profiler summary cell
        ncu_summary = "-"
        if r.get("ncu_profile_json"):
            ncu_summary = (
                f"{len(r.get('profiled_ncu_workloads', []))} profiled, "
                f"{len(r.get('missing_ncu_workloads', []))} missing"
            )

        md.append(
            "| {rank} | {name} | {status} | {avg} | {med} | {p95} | {spd} | {passed}/{total} | {profile} | {ncu} | {url} |".format(
                rank=rank_str,
                name=r.get("candidate", "-"),
                status=r.get("status", "-"),
                avg=(f"{r['avg_latency_ms']:.9f}" if r.get("avg_latency_ms") is not None else "-"),
                med=(f"{r['median_latency_ms']:.9f}" if r.get("median_latency_ms") is not None else "-"),
                p95=(f"{r['p95_latency_ms']:.9f}" if r.get("p95_latency_ms") is not None else "-"),
                spd=(f"{r['avg_speedup_factor']:.3f}x" if r.get("avg_speedup_factor") is not None else "-"),
                passed=r.get("passed_workloads", 0),
                total=r.get("total_workloads", 0),
                profile=profile_summary,
                ncu=ncu_summary,
                url=(r.get("modal_run_url") or "-"),
            )
        )

    md.append("")
    md.append(f"- Combined torch profiler index: `{profile_bundle_path}`")
    md.append(f"- Combined NCU profiler index: `{ncu_bundle_path}`")

    out_md = out_dir / "parallel_results.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments and orchestrate the parallel evaluation.

    Returns 0 on full success, 1 if any candidate is FAILED or PARTIAL.
    """
    parser = argparse.ArgumentParser(description="Parallel Modal evaluator for FlashInfer candidates")

    # -- Required arguments --
    parser.add_argument("--project-root", type=Path, required=True,
                        help="Path to the FlashInfer contest project root to clone for each candidate")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="JSON list of candidate configs (see _load_candidates for schema)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory where all result artifacts are written")

    # -- Parallelism & environment --
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Max concurrent candidate evaluations (ThreadPoolExecutor workers)")
    parser.add_argument("--conda-env", default="fi-bench",
                        help="Conda environment used to invoke `modal run`")
    parser.add_argument("--timeout-s", type=int, default=5400,
                        help="Per-candidate wall-clock timeout in seconds (default 90 min)")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Do not delete temporary workspace clones after evaluation")

    # -- Torch profiler options --
    parser.add_argument(
        "--profile-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.profiler inside each Modal run",
    )
    parser.add_argument("--profile-workload-ids", default="",
                        help="Comma-separated workload UUIDs to torch-profile")
    parser.add_argument("--profile-max-workloads", type=int, default=0,
                        help="Max workloads to torch-profile (0 = unlimited)")
    parser.add_argument("--profile-sort-by", default="cuda_time_total",
                        help="Sort column for torch profiler table")
    parser.add_argument("--profile-row-limit", type=int, default=40,
                        help="Max rows in torch profiler table")
    parser.add_argument("--profile-warmup-runs", type=int, default=3,
                        help="Warm-up iterations before torch profiling starts")

    # -- NCU profiler options --
    parser.add_argument(
        "--profile-ncu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable NCU profiling inside each Modal run",
    )
    parser.add_argument("--ncu-workload-ids", default="",
                        help="Comma-separated workload UUIDs to NCU-profile")
    parser.add_argument("--ncu-max-workloads", type=int, default=0,
                        help="Max workloads to NCU-profile (0 = unlimited)")
    parser.add_argument("--ncu-set", default="detailed",
                        help="NCU metric set (e.g. detailed, full)")
    parser.add_argument("--ncu-page", default="details",
                        help="NCU report page to render (e.g. details, raw)")
    parser.add_argument("--ncu-timeout", type=int, default=240,
                        help="Per-workload NCU timeout in seconds")
    parser.add_argument("--ncu-max-lines", type=int, default=220,
                        help="Max stdout lines captured from NCU")
    parser.add_argument("--ncu-kernel-name", default="",
                        help="Optional regex to filter which GPU kernel(s) NCU instruments")
    args = parser.parse_args()

    # Resolve all paths to absolute form
    project_root = args.project_root.expanduser().resolve()
    candidates = _load_candidates(args.candidates.expanduser().resolve())

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamped temporary root for workspace clones
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    work_root = Path(tempfile.gettempdir()) / f"fi_parallel_eval_{stamp}"
    work_root.mkdir(parents=True, exist_ok=True)

    # Bundle profiling options into typed containers
    profile_options = ProfileOptions(
        enabled=args.profile_torch,
        workload_ids=args.profile_workload_ids,
        max_workloads=args.profile_max_workloads,
        sort_by=args.profile_sort_by,
        row_limit=args.profile_row_limit,
        warmup_runs=args.profile_warmup_runs,
    )
    ncu_options = NcuOptions(
        enabled=args.profile_ncu,
        workload_ids=args.ncu_workload_ids,
        max_workloads=args.ncu_max_workloads,
        ncu_set=args.ncu_set,
        ncu_page=args.ncu_page,
        ncu_timeout=args.ncu_timeout,
        max_lines=args.ncu_max_lines,
        kernel_name=args.ncu_kernel_name,
    )

    # --- Launch all candidates in parallel ---
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        # Submit one future per candidate
        futs = [
            ex.submit(
                _run_candidate,
                project_root,
                work_root,
                out_dir,
                args.conda_env,
                args.timeout_s,
                args.keep_workdir,
                profile_options,
                ncu_options,
                c,
            )
            for c in candidates
        ]

        # Collect results as they complete (order may differ from submission)
        for fut in concurrent.futures.as_completed(futs):
            row = fut.result()
            rows.append(row)
            avg = row.get("avg_latency_ms")
            avg_s = f"{avg:.9f} ms" if avg is not None else "-"
            print(f"[{row.get('status')}] {row.get('candidate')} avg={avg_s} url={row.get('modal_run_url') or '-'}")

    # --- Write combined summary artifacts ---
    _write_summary(out_dir, rows)
    print(f"Saved: {out_dir / 'parallel_results.json'}")
    print(f"Saved: {out_dir / 'parallel_results.md'}")
    print(f"Saved: {out_dir / 'parallel_torch_profiles.json'}")
    print(f"Saved: {out_dir / 'parallel_ncu_profiles.json'}")

    # Return non-zero exit code if any candidate failed or partially failed
    has_fail = any(r.get("status") in {"FAILED", "PARTIAL"} for r in rows)
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
