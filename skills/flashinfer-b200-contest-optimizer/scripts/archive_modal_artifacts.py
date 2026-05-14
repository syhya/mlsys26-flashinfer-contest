#!/usr/bin/env python3
"""
Archive scratch Modal benchmark artifacts into optimize_ops/.

This script copies the files that FlashInfer Modal runs tend to overwrite in
the project root, then snapshots them under `optimize_ops/` using a
hypothesis-driven candidate id. It is intended to make archival a
single-command step instead of a repetitive manual copy checklist.

Workflow overview:
  1. Resolve the project root, candidate id, and current kernel path.
  2. Create the standard archive directories under `optimize_ops/`.
  3. Copy the current kernel snapshot plus whichever scratch benchmark / NCU
     files exist in the root directory.
  4. Avoid clobbering earlier archives by adding `_run2`, `_run3`, etc. when
     the target filename already exists.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the project root, candidate id, and optional workload label."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy full/single benchmark scratch files, NCU markdown reports, and the "
            "current kernel snapshot out of the project root into optimize_ops/."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root that contains optimize_ops/ and scratch benchmark files. Default: cwd.",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Hypothesis-driven candidate id such as v25c_t14107_gemm2_ns2.",
    )
    parser.add_argument(
        "--kernel-path",
        required=True,
        help="Kernel path relative to project root, for example solution/triton/kernel.py.",
    )
    parser.add_argument(
        "--workload-label",
        default="",
        help="Optional workload label for single-workload artifacts, for example long_14107.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Optional explicit run index for full benchmark JSONs. Default: first unused run index.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Naming and path helpers
# ---------------------------------------------------------------------------

def sanitize_token(text: str) -> str:
    """Convert arbitrary text into a conservative filesystem-safe token."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text).strip("._")


def unique_destination(path: Path) -> Path:
    """Return `path` or the next free `*_runN` variant if it already exists."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_run{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def next_full_run_path(bench_dir: Path, candidate_id: str, run_index: int) -> Path:
    """Choose the destination filename for a full benchmark archive."""
    if run_index > 0:
        return bench_dir / f"benchmark_{candidate_id}_run{run_index}.json"
    index = 1
    while True:
        candidate = bench_dir / f"benchmark_{candidate_id}_run{index}.json"
        if not candidate.exists():
            return candidate
        index += 1


def infer_version_group(candidate_id: str) -> str | None:
    """Infer `v25`, `v99`, etc. from a candidate id for kernel snapshot layout."""
    match = re.match(r"^(v\d+)", candidate_id)
    return match.group(1) if match else None


def infer_workload_label(single_benchmark_json: Path) -> str:
    """Infer a stable workload label from a single-workload benchmark JSON.

    Prefer axis values because they are easier to read in round notes. Fall
    back to a shortened workload UUID if axis metadata is missing.
    """
    payload = json.loads(single_benchmark_json.read_text(encoding="utf-8"))
    rows = payload.get("ncu_profile", {}).get("rows", [])
    if rows:
        axes = rows[0].get("axes", {})
        if axes:
            tokens = [f"{sanitize_token(str(key))}_{sanitize_token(str(value))}" for key, value in sorted(axes.items())]
            return "__".join(tokens)

    results = payload.get("results", {})
    if results:
        definition = next(iter(results.values()))
        if definition:
            workload_uuid = next(iter(definition.keys()))
            return f"uuid_{sanitize_token(workload_uuid[:8])}"

    return "single"


# ---------------------------------------------------------------------------
# File-copy helper
# ---------------------------------------------------------------------------

def copy_if_exists(source: Path, destination: Path, copied: list[tuple[Path, Path]]) -> None:
    """Copy one artifact if it exists and record the source/destination pair."""
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append((source, destination))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Archive the current scratch artifacts and print what was copied."""
    args = parse_args()
    project_root = args.project_root.resolve()
    candidate_id = sanitize_token(args.candidate_id)
    kernel_path = (project_root / args.kernel_path).resolve()
    if not kernel_path.exists():
        raise FileNotFoundError(kernel_path)

    optimize_root = project_root / "optimize_ops"
    bench_dir = optimize_root / "benchmarks"
    version_group = infer_version_group(candidate_id)
    kernel_dir = optimize_root / version_group if version_group else optimize_root / "kernels"
    bench_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []

    # Snapshot the current kernel first so the implementation that produced the
    # scratch benchmark files is preserved before the next experiment starts.
    kernel_target = unique_destination(kernel_dir / f"kernel_{candidate_id}{kernel_path.suffix}")
    copy_if_exists(kernel_path, kernel_target, copied)

    # Full-benchmark output is tracked separately because it is the promotion
    # surface used for mean-latency comparisons across repeats.
    full_benchmark = project_root / "benchmark_detailed_results.json"
    full_benchmark_target = next_full_run_path(bench_dir, candidate_id, args.run_index)
    copy_if_exists(full_benchmark, full_benchmark_target, copied)

    # Single-workload artifacts are optional, but they are critical for
    # representative-band gating and paired baseline comparisons.
    single_benchmark = project_root / "benchmark_detailed_result_single.json"
    workload_label = sanitize_token(args.workload_label) or (
        infer_workload_label(single_benchmark) if single_benchmark.exists() else ""
    )
    if workload_label:
        single_benchmark_target = unique_destination(
            bench_dir / f"benchmark_single_{candidate_id}_{workload_label}.json"
        )
        copy_if_exists(single_benchmark, single_benchmark_target, copied)

    # Copy whichever NCU markdown reports exist so the raw text stays available
    # even when round notes rely on the compact extractor output instead.
    full_ncu_report = project_root / "ncu_profile_report.md"
    full_ncu_target = unique_destination(bench_dir / f"ncu_{candidate_id}.md")
    copy_if_exists(full_ncu_report, full_ncu_target, copied)

    if workload_label:
        single_ncu_report = project_root / "ncu_profile_report_single.md"
        single_ncu_target = unique_destination(bench_dir / f"ncu_{candidate_id}_{workload_label}.md")
        copy_if_exists(single_ncu_report, single_ncu_target, copied)

    if not copied:
        raise RuntimeError(
            "No artifacts were copied. Expected at least one scratch benchmark file or kernel snapshot."
        )

    print("Archived artifacts:")
    for source, destination in copied:
        print(f"- {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
