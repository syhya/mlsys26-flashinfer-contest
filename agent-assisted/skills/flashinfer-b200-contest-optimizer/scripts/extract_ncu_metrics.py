#!/usr/bin/env python3
"""
Extract compact kernel metrics from FlashInfer benchmark NCU payloads.

This script reads the `ncu_profile.rows[*].output` text embedded in
`benchmark_detailed_result_single.json` or `benchmark_detailed_results.json`,
parses the most decision-relevant metrics, and emits a compact table or JSON
payload that is easier to compare in round notes than the raw Nsight Compute
markdown dump.

Workflow overview:
  1. Load `ncu_profile.rows` from the benchmark JSON.
  2. Split each workload's raw NCU text into per-kernel blocks.
  3. Parse only the metrics that are typically used for shape-aware tuning
     decisions: duration, throughput, occupancy, registers, shared memory,
     and waves per SM.
  4. Keep one representative record per kernel name (the slowest instance),
     then print either Markdown or JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants – metric allowlist and regexes used during raw text parsing
# ---------------------------------------------------------------------------

TARGET_METRICS = {
    "Duration",
    "Compute (SM) Throughput",
    "Memory Throughput",
    "Registers Per Thread",
    "Dynamic Shared Memory Per Block",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Waves Per SM",
}

KERNEL_HEADER_RE = re.compile(
    r"^  (?P<header>.+, Context \d+, Stream \d+, Device \d+, CC [^\n]+)$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for workload, kernel, and output-format filters."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract a compact kernel table from benchmark_detailed_result_single.json "
            "or benchmark_detailed_results.json with embedded ncu_profile data."
        )
    )
    parser.add_argument("benchmark_json", type=Path, help="Benchmark JSON with ncu_profile")
    parser.add_argument(
        "--workload",
        help="Optional workload UUID filter. Defaults to all profiled workloads.",
    )
    parser.add_argument(
        "--kernel-substring",
        default="",
        help="Optional case-insensitive kernel substring filter.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many kernels to show per workload. Use 0 for all. Default: 5.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Default: table.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Benchmark JSON loading
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    """Load `ncu_profile.rows` from a benchmark JSON file.

    Raises:
        ValueError: If the file does not contain embedded NCU rows.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("ncu_profile", {}).get("rows", [])
    if not rows:
        raise ValueError(f"No ncu_profile.rows found in {path}")
    return rows


# ---------------------------------------------------------------------------
# Unit normalization helpers
# ---------------------------------------------------------------------------

def convert_duration_to_us(value: float, unit: str) -> float:
    """Normalize an NCU duration metric to microseconds."""
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "ns":
        return value / 1000.0
    raise ValueError(f"Unsupported duration unit: {unit}")


def convert_smem_to_kb(value: float, unit: str) -> float:
    """Normalize dynamic shared memory units to KB per block."""
    if unit == "Kbyte/block":
        return value
    if unit == "byte/block":
        return value / 1024.0
    if unit == "Mbyte/block":
        return value * 1024.0
    raise ValueError(f"Unsupported shared-memory unit: {unit}")


def metric_value(metrics: dict[str, dict[str, float | str]], name: str) -> float | None:
    """Read one parsed metric and normalize units where needed."""
    metric = metrics.get(name)
    if not metric:
        return None
    value = float(metric["value"])
    unit = str(metric["unit"])
    if name == "Duration":
        return convert_duration_to_us(value, unit)
    if name == "Dynamic Shared Memory Per Block":
        return convert_smem_to_kb(value, unit)
    return value


# ---------------------------------------------------------------------------
# Raw NCU text parsing
# ---------------------------------------------------------------------------

def shorten_kernel_name(name: str, limit: int = 96) -> str:
    """Truncate very long demangled kernel names for Markdown readability."""
    return name if len(name) <= limit else f"{name[:limit - 3]}..."


def split_kernel_blocks(output: str) -> list[tuple[str, str]]:
    """Split one raw NCU report string into `(header, block_text)` tuples."""
    matches = list(KERNEL_HEADER_RE.finditer(output))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        blocks.append((match.group("header"), output[start:end]))
    return blocks


def strip_launch_dims(launch_prefix: str) -> str:
    """Remove trailing `(grid)x(block)` launch dimensions from a header line."""
    match = re.match(r"^(?P<kernel>.*) \((?P<grid>[^()]*)\)x\((?P<block>[^()]*)\)$", launch_prefix)
    if match:
        return match.group("kernel").strip()
    return launch_prefix.strip()


def parse_metric_lines(block_text: str) -> dict[str, dict[str, float | str]]:
    """Parse the small subset of metric rows used for tuning decisions.

    NCU tables are mostly fixed-width text. Some rows include both a unit and a
    numeric value, while others only show the value column (for example
    `Waves Per SM`). This parser accepts both layouts and ignores every metric
    that is not in `TARGET_METRICS`.
    """
    metrics: dict[str, dict[str, float | str]] = {}
    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("==PROF==", "[", "Section:", "OPT", "INF", "-", "Metric Name")):
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2:
            continue
        metric_name = parts[0]
        if len(parts) >= 3:
            unit = parts[-2]
            value = parts[-1]
        else:
            unit = ""
            value = parts[-1]
        if metric_name not in TARGET_METRICS:
            continue
        try:
            parsed_value = float(value)
        except ValueError:
            continue
        metrics[metric_name] = {"value": parsed_value, "unit": unit}
    return metrics


def parse_kernel_metrics(row: dict, kernel_filter: str) -> list[dict]:
    """Extract compact per-kernel metrics for one profiled workload.

    If the same kernel name appears multiple times in one report, keep the
    slowest instance so the output stays compact while preserving the dominant
    path for that workload.
    """
    output = row.get("output", "")
    kernel_filter = kernel_filter.lower()
    best_by_name: dict[str, dict] = {}
    for header_line, block_text in split_kernel_blocks(output):
        launch_prefix = header_line.split(", Context", 1)[0].strip()
        kernel_name = strip_launch_dims(launch_prefix)
        if kernel_filter and kernel_filter not in kernel_name.lower():
            continue
        metrics = parse_metric_lines(block_text)
        duration_us = metric_value(metrics, "Duration")
        if duration_us is None:
            continue
        # Keep the small, comparison-friendly field set that round notes use.
        kernel_record = {
            "kernel_name": kernel_name,
            "duration_us": duration_us,
            "compute_throughput_pct": metric_value(metrics, "Compute (SM) Throughput"),
            "memory_throughput_pct": metric_value(metrics, "Memory Throughput"),
            "registers_per_thread": metric_value(metrics, "Registers Per Thread"),
            "dynamic_shared_memory_kb": metric_value(metrics, "Dynamic Shared Memory Per Block"),
            "theoretical_occupancy_pct": metric_value(metrics, "Theoretical Occupancy"),
            "achieved_occupancy_pct": metric_value(metrics, "Achieved Occupancy"),
            "waves_per_sm": metric_value(metrics, "Waves Per SM"),
        }
        previous = best_by_name.get(kernel_name)
        # Prefer the slowest instance because it is the dominant one to fix.
        if previous is None or kernel_record["duration_us"] > previous["duration_us"]:
            best_by_name[kernel_name] = kernel_record
    return sorted(best_by_name.values(), key=lambda item: item["duration_us"], reverse=True)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def axes_to_text(axes: dict) -> str:
    """Render workload axes as a stable `key=value` comma-separated string."""
    if not axes:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(axes.items()))


def trim_kernels(kernels: list[dict], top: int) -> list[dict]:
    """Apply the `--top` limit, where `0` means "show all kernels"."""
    if top == 0:
        return kernels
    return kernels[:top]


def render_markdown(rows: list[dict], top: int) -> str:
    """Render the extracted metrics as a Markdown report grouped by workload."""
    lines: list[str] = []
    for row in rows:
        kernels = trim_kernels(row["kernels"], top)
        lines.append(f"## Workload `{row['workload_uuid']}`")
        lines.append(f"- axes: `{axes_to_text(row['axes'])}`")
        if not kernels:
            lines.append("- no matching kernels")
            lines.append("")
            continue
        lines.append("")
        lines.append(
            "| rank | kernel | duration_us | compute_pct | memory_pct | regs/thread | dyn_smem_kb | theor_occ_pct | achieved_occ_pct | waves_per_sm |"
        )
        lines.append(
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for rank, kernel in enumerate(kernels, start=1):
            lines.append(
                "| {rank} | {kernel} | {duration:.2f} | {compute:.2f} | {memory:.2f} | {regs:.0f} | {smem:.2f} | {theor:.2f} | {achieved:.2f} | {waves:.2f} |".format(
                    rank=rank,
                    kernel=shorten_kernel_name(kernel["kernel_name"]).replace("|", "\\|"),
                    duration=kernel["duration_us"],
                    compute=kernel["compute_throughput_pct"] or 0.0,
                    memory=kernel["memory_throughput_pct"] or 0.0,
                    regs=kernel["registers_per_thread"] or 0.0,
                    smem=kernel["dynamic_shared_memory_kb"] or 0.0,
                    theor=kernel["theoretical_occupancy_pct"] or 0.0,
                    achieved=kernel["achieved_occupancy_pct"] or 0.0,
                    waves=kernel["waves_per_sm"] or 0.0,
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Load, filter, parse, and print compact NCU metrics."""
    args = parse_args()
    rows = load_rows(args.benchmark_json)
    selected = []
    for row in rows:
        if args.workload and row.get("workload_uuid") != args.workload:
            continue
        # Apply the kernel-name filter after the workload filter so callers can
        # narrow the output to one representative path when needed.
        kernels = parse_kernel_metrics(row, args.kernel_substring)
        selected.append(
            {
                "workload_uuid": row.get("workload_uuid", "unknown"),
                "axes": row.get("axes", {}),
                "kernels": kernels,
            }
        )

    if not selected:
        print("No matching workloads found.", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(selected, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(selected, args.top), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
