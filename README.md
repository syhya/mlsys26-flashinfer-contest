# MLSys 2026 FlashInfer Contest Submissions

[Technical report](./report.pdf)

This repository is a compact public snapshot of our MLSys 2026 FlashInfer AI
Kernel Generation Contest submissions for the NVIDIA Track. It keeps the final
kernel submission surface small while preserving the scripts and skills needed
to reproduce, evaluate, and continue optimization.

## Repository Layout

```text
.
|-- report.pdf
|-- scripts/
|-- skills/
|-- moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/
|-- gdn_decode_qk4_v8_d128_k_last/
|-- gdn_prefill_qk4_v8_d128_k_last/
|-- dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/
`-- dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/
```

Each kernel directory contains:

- `config.toml`: FlashInfer submission metadata.
- `solution/`: the actual Triton or CUDA source loaded by the evaluator.
- `artifacts/`: retained benchmark evidence from the selected submission.
- `README.md`: per-kernel notes, entry point, candidate name, and retained result.

## Kernels

| Kernel | Entry point | Retained result |
| --- | --- | --- |
| [moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048](./moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/) | `kernel.py::run` | 19/19 passed, 0.289740 ms three-repeat mean latency |
| [gdn_decode_qk4_v8_d128_k_last](./gdn_decode_qk4_v8_d128_k_last/) | `kernel.py::kernel_hybrid_dispatch` | 54/54 passed, 0.006201 ms average latency |
| [gdn_prefill_qk4_v8_d128_k_last](./gdn_prefill_qk4_v8_d128_k_last/) | `kernel.py::kernel_prefill_hybrid` | 100/100 passed, 0.051992 ms average latency |
| [dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64](./dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/) | `kernel.py::run` | 23/23 passed, 0.011128 ms average latency |
| [dsa_topk_indexer_fp8_h64_d128_topk2048_ps64](./dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/) | `kernel.cu::kernel_cuda` | 128/128 passed, 0.006893 ms average latency |

## Full Optimization Records

This combined repository is intentionally curated. The original project
repositories keep the broader optimization traces, scratch work, references, and
historical context:

| Track | Source repository | Snapshot commit used here |
| --- | --- | --- |
| MoE FP8 | [mlsys26-flashinfer-solution-fused-moe](https://github.com/syhya/mlsys26-flashinfer-solution-fused-moe) | `a880866956216b80d1f4a9704b9eb449812f935a` |
| Gated DeltaNet | [mlsys26-flashinfer-solution-gated-delta-net](https://github.com/syhya/mlsys26-flashinfer-solution-gated-delta-net) | `025db70cfdb1247421c2dfaf5ac8f4ab595255d3` |
| DeepSeek Sparse Attention | [mlsys26-flashinfer-solution-sparse-attention](https://github.com/syhya/mlsys26-flashinfer-solution-sparse-attention) | `e2ba69e5a6c26822245db7f2295a37230a6f243e` |

## Skills

The `skills/` directory contains the agent workflow instructions used for
optimization and submission handling:

| Skill | Purpose |
| --- | --- |
| [flashinfer-b200-contest-optimizer](./skills/flashinfer-b200-contest-optimizer/) | FlashInfer B200 contest loop: reference-first recon, shape-aware Modal benchmarking, NCU analysis, and promotion gates. |
| [flashinfer-submission-tagger](./skills/flashinfer-submission-tagger/) | Submission tag and `config.toml` topology validation helper. |

For further optimization, start from `flashinfer-b200-contest-optimizer`, use
the active kernel directory as the project root, and apply the multi sub-agent
prompt template below when deeper debate is useful.

## Setup

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest-trace/
```

The Modal scripts expect the official contest workloads to be available in the
`flashinfer-trace` volume mounted at `/data`.

## Pack a Solution

Pack any kernel directory by pointing at its local `config.toml`:

```bash
python scripts/pack_solution.py \
  --config-path gdn_decode_qk4_v8_d128_k_last/config.toml \
  --output /tmp/gdn_decode.solution.json
```

## Evaluation

Use `--config-path` to select the kernel you want to evaluate. The scripts read
the definition, implementation language, entry point, and solution directory
from that `config.toml`.

For local evaluation on a machine with the dataset and a compatible CUDA GPU:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_local.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml
```

For Modal B200 single-workload evaluation with `run_modal_single.py`, pass either
a workload UUID or a workload index. This is the fastest path for correctness
checks, targeted latency checks, and optional profiler runs:

```bash
python -m modal run scripts/run_modal_single.py \
  --workload-uuid <workload_uuid_or_index> \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --official --no-profile-torch --no-profile-ncu
```

`run_modal_single.py` writes `benchmark_detailed_result_single.json` at the
repository root. To collect profiling data for the same selected workload, use
`--profile-torch` or `--profile-ncu`; NCU profiling also writes
`ncu_profile_report_single.md`.

For Modal B200 full-kernel evaluation with `run_modal_multiple_gpus.py`, set
`FIB_DATASET_PATH` locally so the script can enumerate the workload JSONL, then
run all workloads in parallel:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_modal_multiple_gpus.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --workers 10 \
  --out-dir results/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
```

`run_modal_multiple_gpus.py` launches one Modal run per workload and writes
`benchmark_detailed_results.json` plus `retained_run.log` under `--out-dir`. If
any workload fails because of a transient Modal or container issue, rerun only
the missing or failed workloads:

```bash
python scripts/run_modal_multiple_gpus.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --workers 4 \
  --out-dir results/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
  --retry
```

To evaluate another kernel, replace the `--config-path` and `--out-dir` with
the corresponding top-level kernel directory.
