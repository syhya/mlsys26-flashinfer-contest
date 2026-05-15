# Official Contest Evaluation Environment

Use this file when deciding whether a candidate is merely promising or actually submission-ready.

## Environment

| Field | Value |
|---|---|
| Docker image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| Hardware | Bare-metal NVIDIA B200 (sm_100a) |
| GPU clocks | Locked to max (`nvidia-smi -ac 3996,1965`) |
| CUDA | 13.2 |
| Python | 3.12 |
| PyTorch | 2.12.0+cu132 |
| Triton | 3.6.0 |

Packages inside the container:
- FlashInfer (latest main, built from source)
- FlashInfer-Bench (latest main, built from source)
- `cupti-python` for accurate GPU timing
- `deep-gemm`
- `helion`
- `mlc-ai-tirx-cu130` (TVM)
- `nvidia-cutlass-dsl` (CuTe DSL)
- Contest dataset from https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest

## Evaluation Rules

- For multiple git tags targeting the same definition, only the latest tag is evaluated.
- Tags for different definitions are evaluated independently.
- Each track runs in parallel on B200, but each solution executes in its own isolated subprocess.
- `config.toml` is the source of truth for the submitted definition and build configuration.

## Official Per-Track Commands

### MoE

```bash
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 \
  --atol 1 --rtol 0.3 --required-matched-ratio 0.9
```

### DSA Attention

```bash
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300
```

### DSA Indexer

```bash
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300
```

### GDN Decode

```bash
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300
```

### GDN Prefill

```bash
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions gdn_prefill_qk4_v8_d128_k_last \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 \
  --warmup-runs 1 --iterations 5 --num-trials 3
```

## Official FlashInfer Baselines

| Track | Solution Name | Definition |
|---|---|---|
| MoE | `flashinfer_wrapper_9sdjf3` | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` |
| DSA Attention | `flashinfer_wrapper_5af199` | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| DSA Indexer | `flashinfer_deepgemm_wrapper_2ba145` | `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` |
| GDN Decode | `flashinfer_wrapper_9b7f1e` | `gdn_decode_qk4_v8_d128_k_last` |
| GDN Prefill | `flashinfer_wrapper_123ca6` | `gdn_prefill_qk4_v8_d128_k_last` |

Baseline example:

```bash
flashinfer-bench run \
  --local /path/to/mlsys26-contest \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --solutions flashinfer_wrapper_9b7f1e \
  --use-isolated-runner --timeout 300
```

## Practical Guidance

- Use Modal for broad exploration and NCU-heavy diagnosis, but use official-parity local runs to decide promotion.
- Do not compare a candidate's Modal result directly to an official-parity result without labeling the environment in the notes.
- For `gdn_prefill`, never forget `--warmup-runs 1 --iterations 5 --num-trials 3`; omitting them makes the comparison non-parity.
- Before tagging, verify that the intended commit contains the promoted `config.toml` and kernel for exactly that definition.
- If the repo is private, keep `flashinfer-bot` read access enabled or the submission will not be collected.
