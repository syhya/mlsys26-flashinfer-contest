# DSA Sparse Attention v47 Summary

- Date: `2026-04-25`
- Definition: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- Solution name: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_submission_v47`
- Status: promoted locally; no git tag created in this turn
- Benchmark environment: `flashinfer-bench` GitHub `main` (`ecafa9c2c62007667838226068b21ca9ff1c9183` observed), `force_build=True`, official default `warmup=10`, `iterations=50`, `trials=3`, isolated runner enabled

## Retained Result

| Candidate | Avg latency | Median | P95 | Passed | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| v46c hybrid splits16 run1 | `0.011212434782608693 ms` | `0.01353 ms` | `0.015392 ms` | `23/23` | previous promoted run |
| v46c hybrid splits16 run2 | `0.011322217391304349 ms` | `0.0133278333333333 ms` | `0.0154028333333333 ms` | `23/23` | previous repeat |
| v46c repeat mean | `0.011267326086956521 ms` | n/a | `0.01539741666666665 ms` | `23/23` | promotion threshold |
| v47 half128 run1 | `0.011127804347826085 ms` | `0.0130185 ms` | `0.0152746666666667 ms` | `23/23` | promote candidate |
| v47 half128 run2 | `0.011221753623188413 ms` | `0.0127146666666667 ms` | `0.0152053333333333 ms` | `23/23` | repeat validation |
| v47 repeat mean | `0.011174778985507249 ms` | n/a | `0.01524 ms` | `23/23` | promoted |

v47 repeat mean avg latency is `-0.821376%` versus the v46c repeat mean, and `-8.144%` versus the v45 main-env baseline.

## Kernel Change

The promoted change is intentionally narrow. In `cuda_route_kernel.cu::attn_split_kernel`, the `K_SPLITS=16 / BLOCK_N=128` route now detects tail splits whose upper 64 tokens are all padding via `sIdx[64] == -1`. It keeps the same 128-wide shared-memory/output contract, but skips the second QK slab and the upper-half PV loop for that CTA. The split32 path, upstream MLA path, and short route are unchanged.

## Dispatch Rules

| Condition | Selected path |
| --- | --- |
| `T == 1` | `short_route::fused_dsa_kernel_thr_warpv3` |
| `T == 2 and max_pages < 5` | `short_route::fused_dsa_kernel_thr_warpv3` |
| `T == 2 and max_pages >= 5` | `adapter.py::run_upstream_mla_decode` |
| `T == 6 and max_pages == 1` | `adapter.py::run_upstream_mla_decode` |
| `T == 6 and (max_pages > 1 or non-representable)` | `cuda splits16 / BLOCK_N=128 / base2 / half128 skip` |
| `T == 7 and representable and max_pages < 32` | `cuda splits32 / BLOCK_N=64 / natural exp` |
| `T == 7 and (non-representable or max_pages >= 32)` | `cuda splits16 / BLOCK_N=128 / base2 / half128 skip` |
| `T == 8 and representable and max_pages < 18` | `cuda splits32 / BLOCK_N=64 / natural exp` |
| `T == 8 and (non-representable or max_pages >= 18)` | `cuda splits16 / BLOCK_N=128 / base2 / half128 skip` |

## Gate Results

| Gate | Workload | Baseline | v47 | Delta |
| --- | --- | ---: | ---: | ---: |
| T8 high/nonrepresentable | `564007ac354e4662a62cc4d6352dc494` | `0.013866333333333333 ms` | `0.0138986666666667 ms` | `+0.23%` |
| T6 high span | `d57eb9e19f0642e8af9bd76ad0823303` | `0.015338333333333334 ms` | `0.0135996666666667 ms` | `-11.34%` |
| T7 split32 guard | `3838996164a94d728710f913477feba8` | `0.012469333333333334 ms` | `0.0113593333333333 ms` | `-8.90%` |

The T7 guard does not exercise the new branch; its single-workload win is treated as noise. Promotion is based on the two full 23-workload sweeps.

## NCU Notes

| Band | Workload | Dominant kernel | Duration | Compute | Memory | Regs | Dyn smem | Occ | Waves/SM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v46c T8 high/nonrepresentable | `564007ac354e4662a62cc4d6352dc494` | `attn_split_kernel<16,128,true>` | `12.26 us` | `8.52%` | `12.22%` | `128` | `181.41 KB` | `12.07%` | `0.86` |
| v47 T6 high span | `d57eb9e19f0642e8af9bd76ad0823303` | `attn_split_kernel<16,128,true>` | `12.10 us` | `5.83%` | `8.51%` | `126` | `181.41 KB` | `12.37%` | `0.65` |

## Rejected Probes

| Candidate | Evidence | Decision |
| --- | --- | --- |
| v47 carveout probe | `564007...` baseline `0.013866 ms`, candidate `0.016640 ms` | rejected |
| v47 split16 launch-bounds min-blocks=1 | `564007...` candidate `0.015333 ms` | rejected |

## Artifacts

- Retained v47 run1: `optimize_ops/2026-04-25_v47_half128_full`
- Retained v47 repeat: `optimize_ops/2026-04-25_v47_half128_full_r2`
- Post-change NCU: `optimize_ops/benchmarks/ncu_v47_half128_post_ncu_t6_split16_d57eb9.md`
- Rejected carveout probe: `optimize_ops/benchmarks/benchmark_single_v47_carveout_probe_rejected_t8_high_564007.json`
- Rejected launch-bounds probe: `optimize_ops/benchmarks/benchmark_single_v47_split16_lb1_rejected_t8_high_564007.json`

The retained full-sweep logs were captured under the temporary probe solution name `...submission_v47_half128_probe`. The final `config.toml` uses `...submission_v47`; the benchmarked code is unchanged, and `pack_solution.py` was rerun successfully with the final v47 name.
