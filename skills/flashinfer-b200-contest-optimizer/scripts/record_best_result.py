#!/usr/bin/env python3
"""
Record a benchmark result and maintain a best-kernel registry.

This script is called after a candidate kernel has been evaluated via Modal.
It performs the following:
  1. Reads benchmark metrics (latency, speedup, pass rate) from the
     benchmark JSON produced by the Modal runner.
  2. Snapshots the benchmark JSON and kernel source into a timestamped
     artifacts directory for historical tracking.
  3. Appends the new entry to a persistent registry (registry.json), sorted
     by average latency (ascending), so the first entry is always the
     current best.
  4. Copies the best kernel and its benchmark JSON to well-known paths so
     that downstream tooling (e.g. the contest submission pipeline) can
     always find the latest winner.
  5. Writes a Markdown summary table comparing all recorded variants.

Usage example:
    python record_best_result.py \
        --benchmark-json /tmp/run/benchmark_detailed_results.json \
        --kernel-path    /tmp/run/solution/cuda/kernel.cu \
        --operator       gdn_decode_qk4_v8_d128_k_last \
        --model          B200 \
        --out-dir        ./results \
        --variant        extreme_v5 \
        --modal-run-url  https://modal.com/apps/...
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sanitize_token(text: str) -> str:
    """Convert an arbitrary string to a filesystem-safe token.

    Replaces any character that is not alphanumeric, hyphen, dot, or
    underscore with a hyphen, and strips leading/trailing hyphens.
    """
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in text).strip("-")


# ---------------------------------------------------------------------------
# Metrics extraction from benchmark JSON
# ---------------------------------------------------------------------------

def load_metrics(benchmark_json: Path):
    """Parse a benchmark_detailed_results.json and compute aggregate metrics.

    Reads the first (typically only) definition from the "results" mapping.
    For each trace entry, collects latency_ms, speedup_factor, and status.

    Returns a dict with:
        - definition           – operator definition name
        - solution_name        – name from the [solution] section
        - total_workloads      – total trace entries
        - passed_workloads     – count of entries with status == "PASSED"
        - avg_latency_ms       – arithmetic mean of all latencies
        - median_latency_ms    – median latency
        - p95_latency_ms       – 95th-percentile latency (nearest-rank)
        - avg_speedup          – arithmetic mean of all speedup factors
        - median_speedup       – median speedup factor

    Raises:
        ValueError:       If the JSON has no results or is missing required
                          numeric fields.
        FileNotFoundError: Propagated if *benchmark_json* does not exist.
    """
    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    if "results" not in payload or not payload["results"]:
        raise ValueError(f"No results in {benchmark_json}")

    # The benchmark nests per-workload traces under a definition name key.
    definition = next(iter(payload["results"].keys()))
    traces = payload["results"][definition]
    if not traces:
        raise ValueError(f"No trace entries for definition {definition}")

    # Collect per-workload numeric fields
    latencies = [float(v["latency_ms"]) for v in traces.values() if "latency_ms" in v]
    speedups = [float(v["speedup_factor"]) for v in traces.values() if "speedup_factor" in v]
    statuses = [v.get("status", "UNKNOWN") for v in traces.values()]

    if not latencies or not speedups:
        raise ValueError("Missing latency_ms or speedup_factor in benchmark json")

    # Compute p95 using nearest-rank: pick the element at the 95th percentile index
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    sorted_lat = sorted(latencies)

    return {
        "definition": definition,
        "solution_name": payload.get("solution", {}).get("name", "unknown-solution"),
        "total_workloads": len(statuses),
        "passed_workloads": sum(1 for s in statuses if s == "PASSED"),
        "avg_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": sorted_lat[p95_idx],
        "avg_speedup": mean(speedups),
        "median_speedup": median(speedups),
    }


# ---------------------------------------------------------------------------
# Registry persistence – a simple JSON-file-based leaderboard
# ---------------------------------------------------------------------------

def load_registry(path: Path):
    """Load the existing registry JSON array, or return [] if it doesn't exist."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(path: Path, entries):
    """Write the registry entries list to *path* as pretty-printed JSON.

    Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown summary generation
# ---------------------------------------------------------------------------

def write_summary_md(out_dir: Path, operator: str, model: str, entries):
    """Write a Markdown leaderboard table comparing all recorded variants.

    The table is sorted in the same order as the registry (ascending by
    average latency), so rank 1 is always the best.

    Args:
        out_dir:   Directory to write the .md file into.
        operator:  Operator name used in the heading and filename.
        model:     Model / GPU label used in the heading metadata.
        entries:   The full sorted registry list.

    Returns:
        The Path of the written Markdown file.
    """
    md_path = out_dir / f"{operator}_{sanitize_token(model)}_benchmark_summary.md"
    lines = [
        f"# {operator} Benchmark Summary",
        "",
        f"- model: `{model}`",
        "",
        "| rank | variant | passed | avg latency (ms) | p95 latency (ms) | avg speedup |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(entries, start=1):
        lines.append(
            f"| {idx} | {item['variant']} | {item['passed_workloads']}/{item['total_workloads']} | "
            f"{item['avg_latency_ms']:.6f} | {item['p95_latency_ms']:.6f} | {item['avg_speedup']:.2f}x |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments, record the benchmark, and update best-kernel artifacts.

    After recording:
        - ``<out-dir>/artifacts/registry.json`` is updated with the new entry.
        - ``<out-dir>/artifacts/<timestamp>_<variant>_benchmark.json`` and
          ``<out-dir>/artifacts/<timestamp>_<variant>.kernel.cu`` are the
          historical snapshots.
        - ``<out-dir>/<operator>_<model>_speedup<X>.kernel.cu`` is always
          the current best kernel, overwritten whenever the leaderboard
          changes.
        - ``<out-dir>/artifacts/benchmark_best.json`` is the best benchmark.
        - A Markdown summary table is (re)generated.
    """
    parser = argparse.ArgumentParser(
        description="Record benchmark result and update best kernel artifacts."
    )

    # -- Required arguments --
    parser.add_argument("--benchmark-json", required=True, type=Path,
                        help="Path to benchmark_detailed_results.json from a Modal run")
    parser.add_argument("--kernel-path", required=True, type=Path,
                        help="Path to the kernel.cu source file that was evaluated")
    parser.add_argument("--operator", required=True,
                        help="Operator / definition name (e.g. gdn_decode_qk4_v8_d128_k_last)")
    parser.add_argument("--model", required=True,
                        help="Model or GPU identifier (e.g. B200)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Root output directory for artifacts and summaries")
    parser.add_argument("--variant", required=True,
                        help="Candidate tag / label, e.g. extreme_v5")
    parser.add_argument("--modal-run-url", default="",
                        help="Optional Modal dashboard URL for this run")
    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.benchmark_json.exists():
        raise FileNotFoundError(args.benchmark_json)
    if not args.kernel_path.exists():
        raise FileNotFoundError(args.kernel_path)

    # --- Prepare output directories ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # --- Extract aggregate metrics from the benchmark JSON ---
    metrics = load_metrics(args.benchmark_json)

    # --- Snapshot the benchmark JSON and kernel source with a timestamp ---
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = sanitize_token(args.variant)

    bench_snap = artifacts / f"{stamp}_{tag}_benchmark.json"
    kernel_snap = artifacts / f"{stamp}_{tag}.kernel.cu"
    shutil.copy2(args.benchmark_json, bench_snap)
    shutil.copy2(args.kernel_path, kernel_snap)

    # --- Build the registry entry for this variant ---
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "modal_run_url": args.modal_run_url,
        "benchmark_snapshot": str(bench_snap),
        "kernel_snapshot": str(kernel_snap),
        **metrics,  # merge in all computed metrics (avg_latency_ms, etc.)
    }

    # --- Append to the registry and re-sort ---
    # The registry is sorted by (avg_latency_ms ASC, p95_latency_ms ASC,
    # avg_speedup DESC) so that the first entry is always the overall best.
    registry_path = artifacts / "registry.json"
    entries = load_registry(registry_path)
    entries.append(entry)
    entries.sort(key=lambda e: (e["avg_latency_ms"], e["p95_latency_ms"], -e["avg_speedup"]))
    save_registry(registry_path, entries)

    # --- Promote the best variant's kernel to a well-known path ---
    best = entries[0]
    best_speed = best["avg_speedup"]
    # The filename encodes the operator, model, and speedup for quick identification
    best_kernel_name = f"{args.operator}_{sanitize_token(args.model)}_speedup{best_speed:.3f}.kernel.cu"
    best_kernel_path = args.out_dir / best_kernel_name

    shutil.copy2(best["kernel_snapshot"], best_kernel_path)
    shutil.copy2(best["benchmark_snapshot"], artifacts / "benchmark_best.json")

    # --- (Re)generate the Markdown summary comparing all variants ---
    summary_md = write_summary_md(args.out_dir, args.operator, args.model, entries)

    # --- Write a machine-readable comparison summary JSON ---
    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant": best["variant"],
        "best_kernel": str(best_kernel_path),
        "best_benchmark": str(artifacts / "benchmark_best.json"),
        "summary_markdown": str(summary_md),
    }
    (artifacts / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Print confirmation ---
    print("[OK] Recorded variant:", args.variant)
    print("[OK] Best variant:", best["variant"])
    print("[OK] Best kernel:", best_kernel_path)
    print("[OK] Summary markdown:", summary_md)


if __name__ == "__main__":
    main()
