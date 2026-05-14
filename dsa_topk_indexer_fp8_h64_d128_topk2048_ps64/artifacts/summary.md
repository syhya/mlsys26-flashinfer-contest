# DSA Top-k Indexer Submission Summary

- Date: `2026-04-25`
- Definition: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- Prepared submission record: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64_submission_v50`
- Prepared submission tag: `submission-v50`
- Status: promoted candidate v49c (CuTe tile N=16)
- Previous tagged top-k submission: `submission-v48`
- Entry point: `kernel.cu::kernel_cuda`
- Candidate: `v49c_cute_tile_n16`

## Dataset

- Source JSONL: `mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl`
- Workload count: `128`
- `num_pages`: always `11923`
- `batch_size` range: `1..31`
- `max_num_pages` bands: `1-32 x69`, `33-36 x18`, `37-45 x21`, `82-91 x20`

## Performance Comparison

| Metric | `submission-v48` | `submission-v50` | Delta |
| --- | ---: | ---: | ---: |
| Avg latency | `0.007604 ms` | `0.006893 ms` | **-9.35%** |
| Passed | `128/128` | `128/128` | - |

## Per-Kernel Profile (medium band, bs=14, mnp=35)

| Kernel | v48 | v50 | Delta |
| --- | ---: | ---: | ---: |
| CuTe Scorer | `6.783 us` | `4.896 us` | **-27.8%** |
| Filtered Selector | `5.728 us` | `5.312 us` | **-7.3%** |
| **Total** | **`12.511 us`** | **`10.208 us`** | **-18.4%** |

## What Changed

1. **CuTe scorer tile N=16**: Double token tile from N=8 to N=16. TiledMMA layout changed from `Shape<_4,_1,_1>` to `Shape<_4,_2,_1>`, giving M=64 (heads) x N=16 (tokens) x K=16 with 8 warps / 256 threads. Halves scorer CTAs (~280→~140 for medium band) with better Q reuse per CTA.

2. **Parallel epilogue**: Each warp independently reduces 2 tokens (32 lanes x 2 heads/lane = 64 heads). Eliminates inter-warp merge and shared-memory barrier, removing `warp_sums` shared array.

3. **launch_bounds(256, 2)**: Targets 2 CTAs/SM so the reduced grid fits in 1 wave on B200 (148 SMs).

4. **Unchanged components**: Filtered selector (vec4 loads), hist2048 fallback, short-only pass-through, PDL pre-init.

## Rejected Variants

| Variant | Result | Reason |
| --- | --- | --- |
| v49d (tile N=32) | Regressed | Per-CTA work 4x but CTAs only halve |
| launch_bounds(256, 3) | No effect | ~0% change on both medium and long bands |

## Submission Decision

v49c is promoted as `submission-v50`. The -9.35% avg latency improvement is stable across all 128 workloads with zero correctness failures. The scorer speedup (-27.8%) is the primary driver. Selector now dominates at 52% of total time, making it the target for future optimization.
