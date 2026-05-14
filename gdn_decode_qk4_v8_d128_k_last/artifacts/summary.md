# GDN Decode Submission Summary

- Date: 2026-04-25
- Definition: `gdn_decode_qk4_v8_d128_k_last`
- Candidate: `v36_b8_warp1_stage3_hybrid_fi_vendored`
- Solution name: `gdn_decode_qk4_v8_d128_k_last_v36_b8_warp1_stage3_hybrid_fi_vendored_submission_v27`
- Entry point: `kernel.py::kernel_hybrid_dispatch`
- Latest existing repo tag before this decode update: `submission-v26`
- Prepared submission tag: `submission-v27`
- Reporting basis: official-aligned Modal environment (`flashinfer/flashinfer-ci-cu132:20260401-2c675fb`, `flashinfer-bench` GitHub `main`, `cupti-python`, `use_isolated_runner=True`, `warmup=10`, `iter=50`, `trials=3`)

## Official-Aligned 54-Workload Result

| Candidate | Date | Avg Latency (ms) | Median (ms) | P95 (ms) | Avg Ref (ms) | Avg Speedup | Passed |
|-----------|------|-----------------:|------------:|---------:|-------------:|------------:|-------:|
| current retained `v36_b8_warp1_stage3_hybrid_fi_vendored` | 04-25 | **0.006201** | **0.004910** | **0.012570** | 49.41993 | 5387.7x | 54/54 |
| same-day baseline rerun `v35_hybrid_fla_small_fi_vendored_fastpath_b16_highcta_b48_v16_nbps4` | 04-25 | 0.006332 | **0.004910** | **0.012550** | 48.57900 | 5224.0x | 54/54 |
| 04-21 retained recheck `v35_hybrid_fla_small_fi_vendored_fastpath_b16_highcta_b48_v16_nbps4` | 04-21 | 0.006356 | 0.005025 | 0.012660 | 81.92110 | 8219.3x | 54/54 |
| 04-16 retained recheck `v35_hybrid_fla_small_fi_vendored_fastpath_b16_highcta_b48_v16_nbps4` | 04-16 | 0.006347 | 0.005030 | 0.012710 | 81.99848 | 8354.2x | 54/54 |
| current tagged decode payload in `submission-v20` (`v34_hybrid_fla_small_fi_vendored_fastpath_b16_highcta`, unchanged from `submission-v19`) | 04-15 | 0.006386 | **0.005025** | **0.012660** | 81.94799 | 8310.0x | 54/54 |
| `submission-v16` retained `v31_hybrid_fla_small_fi_vendored` | 04-09 | 0.006490 | 0.005400 | 0.012890 | 90.87250 | **8989.9x** | 54/54 |

Latest retained deltas:

- vs the same-day v35 baseline rerun: **-2.06% avg latency**, median flat, **+0.16% p95 latency**
- vs the 2026-04-21 retained v35 recheck: **-2.43% avg latency**, **-2.29% median latency**, **-0.71% p95 latency**
- vs current tagged decode payload in `submission-v20`: **-2.88% avg latency**, **-2.29% median latency**, **-0.71% p95 latency**
- paired same-day result: `28` wins, `5` ties, `21` losses; mean paired improvement **2.07%**, median paired improvement **0.21%**
- the same-day v35 rerun has one slow `B48` outlier (`0.01442 ms`); excluding that single baseline max outlier, v36 still improves average latency by **0.84%**
- mean across the current v36 run plus the three official-aligned v35 full sweeps: **0.006309 ms** avg latency, **0.004969 ms** median, **0.012623 ms** p95

Latency remains the contest-primary metric. The v36 run is a small average-latency improvement over same-day v35 baseline evidence, with the clearest stable win isolated to `batch_size == 8`.

## Batch Breakdown Vs Same-Day V35 Baseline Rerun

- `batch_size == 1`: improvement (`-9.19%`, `0.002710 -> 0.002461 ms`), with `6/10` wins
- `batch_size == 4`: effectively flat/slight improvement (`-0.37%`, `0.003051 -> 0.003040 ms`), with `5/8` wins
- `batch_size == 8`: stable improvement (`-2.34%`, `0.004147 -> 0.004050 ms`), with `7/7` wins
- `batch_size == 16`: effectively flat (`-0.15%`, `0.004926 -> 0.004919 ms`), with `3/7` wins
- `batch_size == 32`: slight regression (`+0.44%`, `0.007411 -> 0.007444 ms`), with `1/7` wins
- `batch_size == 48`: aggregate improvement (`-5.80%`, `0.010714 -> 0.010093 ms`), but mostly from one same-day baseline outlier; excluding that outlier this bucket is flat
- `batch_size == 64`: slight regression (`+0.37%`, `0.012505 -> 0.012551 ms`), with `2/8` wins

## Architecture

### Dispatch

```python
if batch_size <= 8:
    -> Triton recurrent path
elif batch_size == 16:
    -> vendored FlashInfer CuTe pretranspose path (high CTA)
elif batch_size == 48:
    -> dedicated vendored FlashInfer CuTe `TILE_V=16` path
else:
    -> vendored FlashInfer CuTe pretranspose path (default CTA)
```

### Small-batch path

- Uses the Triton recurrent kernel adapted from the `fla-recurrent` route
- Selected for `batch_size` `{1, 4, 8}`
- `batch_size == 8` now uses `num_warps=1` and `num_stages=3`; B1/B4 keep the previous launch settings

### Transition band: `batch_size == 16`

- Uses the same vendored CuTe pretranspose decode implementation kept in `solution/triton/fi_pretranspose_vendored.py`
- Keeps `B16_NUM_BLOCKS_PER_STATE=16` only for `batch_size == 16`
- The latest recheck still shows this split remains locally positive versus the 04-16 retained sweep

### Dedicated `B48` path

- Uses the widened `TILE_V_LARGE=16` vendored CuTe path with `num_blocks_per_state=4`
- Enabled only for `batch_size == 48`
- The latest recheck stays effectively flat at `B48`, which is consistent with the original full-sweep promotion decision

### Default large-batch path

- `batch_size` `{32, 64}` stay on the default vendored CuTe contest fast path
- `DEFAULT_NUM_BLOCKS_PER_STATE=8` remains the safer baseline outside the `B48` specialization
- `B64` is still slightly better on the updated environment than on the 04-16 retained sweep

## Validation Notes

- Latest full sweep path: `optimize_ops/gdn_decode_qk4_v8_d128_k_last/full_sweep_v36_20260425_syprofile_w20`
- Latest real time: `3m20s` for the initial workers=20 sweep plus `55s` for the targeted retry
- Latest run completed **54/54 PASSED** after one targeted retry
- Same-day v35 baseline rerun path: `optimize_ops/gdn_decode_qk4_v8_d128_k_last/baseline_v35_rerun_20260425_syprofile_w20`
- Same-day v35 baseline rerun completed **54/54 PASSED** in `4m23s` with no retry
- Root `retained_run.json` and `retained_run.log` now mirror the 2026-04-25 v36 full sweep
- Previous retained v35 sweep remains archived at `optimize_ops/gdn_decode_qk4_v8_d128_k_last/re_eval_20260421_env_update`
- The 2026-04-25 sweep needed one targeted retry after a transient `PARSE_ERROR` on workload `9f238670-9a56-4ab9-94f9-555755f32205`

## Decision

Keep v36 as the active decode target for `submission-v27`.

Reason:

- the v36 run improves average latency by **2.06%** vs a same-day v35 baseline rerun, and by **0.84%** after excluding the single slowest same-day baseline outlier
- the B8 launch specialization is stable in the same-day paired check: **7/7** B8 wins and **2.34%** lower B8 mean latency
- the v36 run improves average latency by **2.88%** vs the current tagged decode payload in `submission-v20`
- the updated run shows no compatibility, build, or correctness regression under the pinned `20260401-2c675fb` official-aligned image
