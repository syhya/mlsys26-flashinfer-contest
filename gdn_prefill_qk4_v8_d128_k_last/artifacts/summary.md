# GDN Prefill Submission Summary

- Date: 2026-04-23
- Definition: `gdn_prefill_qk4_v8_d128_k_last`
- Candidate: `pr3001_blackwell_narrow_short_fallback`
- Solution name: `gdn_prefill_qk4_v8_d128_k_last_pr3001_blackwell_narrow_short_fallback_submission_v26`
- Entry point: `kernel.py::kernel_prefill_hybrid`
- Prepared submission tag: `submission-v26`
- Previous submission: `submission-v25`
- Reporting basis: official-aligned Modal environment (`flashinfer/flashinfer-ci-cu132:20260401-2c675fb`, `flashinfer-bench` GitHub `main`, `cupti-python`, `use_isolated_runner=True`, `warmup=1`, `iter=5`, `trials=3`)

## Official-Aligned 100-Workload Result

| Candidate | Validation | Avg Latency (ms) | Median (ms) | P95 (ms) | Avg Ref (ms) | Avg Speedup | Passed |
|-----------|------------|-----------------:|------------:|---------:|-------------:|------------:|-------:|
| **draft `submission-v26` `pr3001_blackwell_narrow_short_fallback`** | full sweep + retry | **0.05199** | **0.02105** | **0.15769** | **1095.884** | **9621.1x** | 100/100 |
| current tagged `submission-v25` `v56_regcap` | full sweep | 0.18518 | 0.06645 | 0.69512 | 1160.659 | 2714.7x | 100/100 |
| retained `submission-v24` / `submission-v23` `v54a` mean | 2-run mean | 0.16670 | 0.06578 | 0.58858 | n/a | 3069.0x | 100/100 |

Key deltas vs current tagged `submission-v25`:

- **-71.92% avg latency** (`0.1851828 -> 0.0519917 ms`)
- **-68.32% median latency** (`0.06645 -> 0.02105 ms`)
- **-77.31% p95 latency** (`0.69512 -> 0.15769 ms`)
- **90 wins / 10 losses / 0 ties** at the workload level
- **+254.40% avg speedup factor** (`2714.7x -> 9621.1x`)

## Candidate Change

This draft replaces the old wide pair-dispatch prefill surface with the upstream PR #3001 Blackwell chunk kernel as the default main path:

- the old 40-pair override table is gone
- the Blackwell chunk kernel now handles every non-fallback workload
- only a narrow recovered short CUDA fallback remains for the 15 tiny measured-regression shape pairs from the retained comparison
- gate precompute is one minimal scalar CUDA helper (`compute_gates`) for every non-fallback shape

The vendored upstream code carries only the contest-specific patches needed for correctness and cu132 portability:

- `gate` is consumed in natural-log space rather than exponentiated gate space
- `is_persistent=False` is forced in the contest wrapper
- `cute.ceil_div(...)` results are explicitly cast to `cutlass.Int32`
- TMEM copy loops use `range_constexpr` to avoid cu132 lowering failures
- long-sequence state rescaling uses the chunk-tail `cumprod_total` fix

## Validation Notes

- Candidate benchmark mirror: `gdn_prefill_qk4_v8_d128_k_last/retained_run.json` and `gdn_prefill_qk4_v8_d128_k_last/retained_run.log`
- Local scratch archive: `optimize_ops/gdn_prefill_qk4_v8_d128_k_last/full_eval_20260423_narrow_short_fallback`
- Candidate result: **100/100 PASSED**
- Execution detail: the first sweep stalled after writing `42/100 PASSED`; one `--retry` run skipped those `42` and filled the remaining `58` workloads without introducing any failures
- Root `retained_run.json` / `retained_run.log` now mirror that retained run exactly
- Historical `submission-v25` evidence remains archived under:
  - `optimize_ops/benchmarks/round_20260421_gdn_prefill_v56_regcap_modal_gate/full_sweep_candidate`

## Decision

Prepare `submission-v26` as a draft around `pr3001_blackwell_narrow_short_fallback`.

Reason:

- the latest official-aligned 100-workload sweep is materially faster than the current tagged `submission-v25`
- the narrow short fallback recovers the tiny-shape regressions without reintroducing the old 40-pair dispatch surface
- commit, tag, and push should still wait for explicit user confirmation
