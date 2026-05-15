# Agent-Assisted FlashInfer Contest Package

This directory is the agent-assisted reproducibility package for our MLSys
2026 FlashInfer AI Kernel Generation Contest submissions. It contains the
retained kernels, submission configs, benchmark artifacts, workflow scripts,
agent skills, and the agent-assisted technical report.

## Layout

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
- `solution/`: Triton or CUDA source loaded by the evaluator.
- `artifacts/`: retained benchmark logs and summaries.
- `README.md`: per-kernel notes, entry point, candidate name, and retained
  result.

## Retained Kernels

| Track | Kernel | Retained result |
| --- | --- | --- |
| MoE FP8 | [Block-scale routing](./moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/)<br>`kernel.py::run` | 19/19 passed<br>0.289740 ms, three-repeat mean |
| Gated DeltaNet | [Decode QK4](./gdn_decode_qk4_v8_d128_k_last/)<br>`kernel.py::kernel_hybrid_dispatch` | 54/54 passed<br>0.006201 ms average |
| Gated DeltaNet | [Prefill QK4](./gdn_prefill_qk4_v8_d128_k_last/)<br>`kernel.py::kernel_prefill_hybrid` | 100/100 passed<br>0.051992 ms average |
| DeepSeek Sparse Attention | [Sparse attention](./dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/)<br>`kernel.py::run` | 23/23 passed<br>0.011128 ms average |
| DeepSeek Sparse Attention | [Top-k indexer](./dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/)<br>`kernel.cu::kernel_cuda` | 128/128 passed<br>0.006893 ms average |

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

Run commands from this `agent-assisted/` directory:

```bash
python scripts/pack_solution.py \
  --config-path gdn_decode_qk4_v8_d128_k_last/config.toml \
  --output /tmp/gdn_decode.solution.json
```

## Evaluation

Use `--config-path` to select the kernel to evaluate. The scripts read the
definition, implementation language, entry point, and solution directory from
that `config.toml`.

For local evaluation on a machine with the dataset and a compatible CUDA GPU:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_local.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml
```

For Modal B200 single-workload evaluation:

```bash
python -m modal run scripts/run_modal_single.py \
  --workload-uuid <workload_uuid_or_index> \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --official --no-profile-torch --no-profile-ncu
```

For Modal B200 full-kernel evaluation:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_modal_multiple_gpus.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --workers 10 \
  --out-dir results/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
```

To retry only missing or failed workloads from an existing output directory:

```bash
python scripts/run_modal_multiple_gpus.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --workers 4 \
  --out-dir results/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
  --retry
```

## Agent Workflow Skills

The [skills](./skills/) directory contains the workflow instructions used for
optimization and submission handling:

| Skill | Purpose |
| --- | --- |
| [flashinfer-b200-contest-optimizer](./skills/flashinfer-b200-contest-optimizer/) | FlashInfer B200 contest loop: reference-first recon, shape-aware Modal benchmarking, NCU analysis, and promotion gates. |
| [flashinfer-submission-tagger](./skills/flashinfer-submission-tagger/) | Submission tag and `config.toml` topology validation helper. |

For broader repository context and the autonomous full-agent package, see the
top-level [README](../README.md).

## License

This package is licensed under the [Apache License 2.0](../LICENSE).

## Citation

If this work is helpful, please cite the technical report:

```bibtex
@misc{shui2026harnessengineering,
  title        = {Harness Engineering for LLM-Driven GPU Kernel Generation},
  author       = {Yue Shui and Chenyu Ma and Hangfei Xu and Shengzhao Wen and Yanpeng Wang},
  year         = {2026},
  howpublished = {\url{https://github.com/syhya/mlsys26-flashinfer-contest}},
  note         = {Technical report for the MLSys 2026 FlashInfer AI Kernel Generation Contest}
}
```
