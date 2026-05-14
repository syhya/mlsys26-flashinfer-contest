# DSA Top-k Indexer Submission

Definition: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`

- Config: `config.toml`
- Source: `solution/cuda/`
- Entry point: `kernel.cu::kernel_cuda`
- Candidate: `v49c_cute_tile_n16`
- Retained result: 128/128 passed, 0.006893 ms average latency

Retained run evidence is in `artifacts/`.
