# LLM-CUDA Submissions for the MLSys 2026 FlashInfer Contest

This repository contains two distinct MLSys 2026 FlashInfer submission
packages. They cover overlapping contest kernels, but they document different
levels of agent autonomy and point to different source repositories.

## Technical Reports

- [Agent-Assisted Report](./agent-assisted/report.pdf): human-in-the-loop
  optimization using agent skills, benchmark harnesses, and curated retained
  kernels.
- [Full-Agent Report](./full-agent/FULL_AGENT_WRITEUP.pdf): autonomous
  LoongFlow-derived PES kernel search after task setup, with full planner /
  executor / evaluator / summarizer traces.

## Official Results

LLM-CUDA placed in the top three of three MLSys 2026 FlashInfer NVIDIA Track
categories, according to the official contest results:

| Track | Approach | Result |
| --- | --- | --- |
| Track A — Fused MoE | Agent-Assisted | **3rd place** |
| Track C — Gated Delta Net | Agent-Assisted | **3rd place** |
| Track C — Gated Delta Net | Full-Agent | **2nd place** |

Official leaderboard: <https://mlsys26.flashinfer.ai/>

The two local top-level packages map directly to those report types:

- `agent-assisted/`: the compact public snapshot of our MLSys 2026 FlashInfer
  AI Kernel Generation Contest submissions for the NVIDIA Track.
- `full-agent/`: autonomous Full-Agent submissions produced end-to-end by an
  agent system adapted from [**Baidu Baige LoongFlow**](https://github.com/baidu-baige/LoongFlow). After task setup, no
  human edits intermediate kernels, selects parents, filters failures, or
  steers the next implementation direction. Three operator-group packages
  (`dsa/`, `gdn/`, `moe/`) cover all five contest kernels, and every iteration's planner /
  executor / evaluator / summarizer artifact is preserved on disk for audit.

## Repository Layout

```text
.
|-- README.md
|-- agent-assisted/
|   |-- report.pdf
|   |-- scripts/
|   |-- skills/
|   |-- moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/
|   |-- gdn_decode_qk4_v8_d128_k_last/
|   |-- gdn_prefill_qk4_v8_d128_k_last/
|   |-- dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/
|   `-- dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/
`-- full-agent/
    |-- README.md
    |-- FULL_AGENT_WRITEUP.pdf
    |-- dsa/      DSA — 2 CUDA kernels + LoongFlow agent + full traces
    |-- gdn/      GDN — 2 CUDA kernels + LoongFlow agent + full traces
    `-- moe/      MoE — 1 Triton kernel + LoongFlow agent + full trace
```

Each `agent-assisted/` kernel directory contains:

- `config.toml`: FlashInfer submission metadata.
- `solution/`: the actual Triton or CUDA source loaded by the evaluator.
- `artifacts/`: retained benchmark evidence from the selected submission.
- `README.md`: per-kernel notes, entry point, candidate name, and retained result.

## External Historical Source Repositories

Original repositories for the Agent-Assisted and Full-Agent workflows:

| Track | Agent-Assisted source repository | Full-Agent source repository |
| --- | --- | --- |
| MoE FP8 | [syhya/mlsys26-flashinfer-solution-fused-moe](https://github.com/syhya/mlsys26-flashinfer-solution-fused-moe) | [m-chenyu/mlsys26-flashinfer-solution-fused-moe](https://github.com/m-chenyu/mlsys26-flashinfer-solution-fused-moe) |
| Gated DeltaNet | [syhya/mlsys26-flashinfer-solution-gated-delta-net](https://github.com/syhya/mlsys26-flashinfer-solution-gated-delta-net) | [m-chenyu/flashinfer-bench-loongflow-fullagent-gdn](https://github.com/m-chenyu/flashinfer-bench-loongflow-fullagent-gdn) |
| DeepSeek Sparse Attention | [syhya/mlsys26-flashinfer-solution-sparse-attention](https://github.com/syhya/mlsys26-flashinfer-solution-sparse-attention) | [m-chenyu/flashinfer-bench-loongflow-fullagent-dsa](https://github.com/m-chenyu/flashinfer-bench-loongflow-fullagent-dsa) |

## Agent-Assisted Retained Kernels

These are the curated human-guided submissions under `agent-assisted/`.
For autonomous LoongFlow outputs, use the separate
[Full-Agent Submissions](#full-agent-submissions) section below.

| Track | Kernel | Agent-Assisted retained result |
| --- | --- | --- |
| MoE FP8 | [Block-scale routing](./agent-assisted/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/)<br>`kernel.py::run` | 19/19 passed<br>0.289740 ms, three-repeat mean |
| Gated DeltaNet | [Decode QK4](./agent-assisted/gdn_decode_qk4_v8_d128_k_last/)<br>`kernel.py::kernel_hybrid_dispatch` | 54/54 passed<br>0.006201 ms average |
| Gated DeltaNet | [Prefill QK4](./agent-assisted/gdn_prefill_qk4_v8_d128_k_last/)<br>`kernel.py::kernel_prefill_hybrid` | 100/100 passed<br>0.051992 ms average |
| DeepSeek Sparse Attention | [Sparse attention](./agent-assisted/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/)<br>`kernel.py::run` | 23/23 passed<br>0.011128 ms average |
| DeepSeek Sparse Attention | [Top-k indexer](./agent-assisted/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/)<br>`kernel.cu::kernel_cuda` | 128/128 passed<br>0.006893 ms average |

## Agent-Assisted Skills

For Full-Agent reproduction, use [`full-agent/README.md`](./full-agent/README.md).

The `agent-assisted/skills/` directory contains the agent workflow instructions
used for optimization and submission handling:

| Skill | Purpose |
| --- | --- |
| [flashinfer-b200-contest-optimizer](./agent-assisted/skills/flashinfer-b200-contest-optimizer/) | FlashInfer B200 contest loop: reference-first recon, shape-aware Modal benchmarking, NCU analysis, and promotion gates. |
| [flashinfer-submission-tagger](./agent-assisted/skills/flashinfer-submission-tagger/) | Submission tag and `config.toml` topology validation helper. |

For further optimization, start from `flashinfer-b200-contest-optimizer`, use
the active kernel directory under `agent-assisted/` as the project root.

## Agent-Assisted Setup

For Full-Agent setup, use [`full-agent/README.md`](./full-agent/README.md).

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

## Pack an Agent-Assisted Solution

Pack any kernel directory by pointing at its local `config.toml`:

```bash
cd agent-assisted
python scripts/pack_solution.py \
  --config-path gdn_decode_qk4_v8_d128_k_last/config.toml \
  --output /tmp/gdn_decode.solution.json
```

## Agent-Assisted Evaluation

For Full-Agent evaluation, use [`full-agent/README.md`](./full-agent/README.md).

Use `--config-path` to select the kernel you want to evaluate. The scripts read
the definition, implementation language, entry point, and solution directory
from that `config.toml`.

For local evaluation on a machine with the dataset and a compatible CUDA GPU:

```bash
cd agent-assisted
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_local.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml
```

For Modal B200 single-workload evaluation with `run_modal_single.py`, pass either
a workload UUID or a workload index. This is the fastest path for correctness
checks, targeted latency checks, and optional profiler runs:

```bash
cd agent-assisted
python -m modal run scripts/run_modal_single.py \
  --workload-uuid <workload_uuid_or_index> \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --official --no-profile-torch --no-profile-ncu
```

`run_modal_single.py` writes `benchmark_detailed_result_single.json` at the
`agent-assisted/` root. To collect profiling data for the same selected
workload, use `--profile-torch` or `--profile-ncu`; NCU profiling also writes
`ncu_profile_report_single.md`.

For Modal B200 full-kernel evaluation with `run_modal_multiple_gpus.py`, set
`FIB_DATASET_PATH` locally so the script can enumerate the workload JSONL, then
run all workloads in parallel:

```bash
cd agent-assisted
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
cd agent-assisted
python scripts/run_modal_multiple_gpus.py \
  --config-path dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/config.toml \
  --workers 4 \
  --out-dir results/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
  --retry
```

To evaluate another kernel, replace the `--config-path` and `--out-dir` with
the corresponding kernel directory under `agent-assisted/`.

## Full-Agent Submissions

The `full-agent/` directory contains five autonomous LoongFlow-derived
submissions covering the same three contest tracks. Every kernel was
produced end-to-end by the agent — task setup is the only human step, and
each iteration's plan, candidate code, evaluator log, and summary is
preserved on disk under `Full-agent-trace_*/` (or `Full_agent_trace/` for
MoE).

### Final submitted kernels

All five final kernels pass correctness on the official evaluator. Kernel
latency is the primary metric (lower is better); reference latency is the
supplied FlashInfer baseline included as context only. See the
[Full-Agent report](./full-agent/FULL_AGENT_WRITEUP.pdf) Table 1 for trajectories.

| Track | Task | Impl. | Kernel latency | Reference latency | Best iter. | Solution ID |
| --- | --- | --- | --- | --- | --- | --- |
| GDN | [Decode](./full-agent/gdn/gdn_decode_qk4_v8_d128_k_last/) | CUDA | **0.0084 ms** | 116.783 ms | 10 | `01bea42b` |
| GDN | [Prefill](./full-agent/gdn/gdn_prefill_qk4_v8_d128_k_last/) | CUDA | **0.7020 ms** | 1236.742 ms | 10 | `ed639eed` |
| DSA | [Sparse attention](./full-agent/dsa/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/) | CUDA | **0.0272 ms** | 3.421 ms | 20 | `94736692` |
| DSA | [Top-k indexer](./full-agent/dsa/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/) | CUDA | **0.0356 ms** | 3.256 ms | 14 | `dc51d5fc` |
| MoE | [FP8 block-scale routing](./full-agent/moe/) | Triton | **1.448 ms** | 16.542 ms | 40 | `6da80c61` |

### System design (summary)

The Full-Agent package adapts [Baidu Baige LoongFlow](https://github.com/baidu-baige/LoongFlow) to FlashInfer kernel generation. It combines autonomous planning, code generation, official-style evaluation, checkpointing, and trace export for CUDA / Triton submissions. See the [Full-Agent report](./full-agent/FULL_AGENT_WRITEUP.pdf) for the full system design and per-task analysis.

### Full-Agent Operator Groups

| Operator group | Kernels | Iterations, same order as kernels | Trace mirror |
| --- | --- | --- | --- |
| [`full-agent/dsa/`](./full-agent/dsa/) | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`<br>`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` | 29 / 20 | `Full-agent-trace_sparse_attn/`, `Full-agent-trace_topk_indexer/` |
| [`full-agent/gdn/`](./full-agent/gdn/) | `gdn_decode_qk4_v8_d128_k_last`<br>`gdn_prefill_qk4_v8_d128_k_last` | 10 / 10 | `Full-agent-trace_gdn_decode/`, `Full-agent-trace_gdn_prefill/` |
| [`full-agent/moe/`](./full-agent/moe/) | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` | 40 | `Full_agent_trace/` |

### Trace schema

Every `Full-agent-trace_*/` mirror has the same three subtrees:

- `database/checkpoints/checkpoint-checkpoint-iter-{K}-{N}/` — per-iteration
  best solution (`best_solution.json` with full code, score, `parent_id`),
  population state (`metadata.json`), and admitted candidates (`solutions/`).
- `iteration/{K}/` — per-round working dir: `planner/` prompt + LLM
  response, `executor/{M_N}/` per-child code and history, round-K winner
  in `executor/best_solution.py`, distilled feedback in `summarizer/`.
- `evaluator/eval_<hash>/` — per-evaluation record: kernel under test
  (`llm_code_*.py`), `evaluation_process.log`, `result.json`.

The `solution` field of each selected `best_solution.json` is byte-identical
to the corresponding submission file under `solution/`.

### Selected Checkpoints

| Task | Selected checkpoint |
| --- | --- |
| GDN decode | `full-agent/gdn/Full-agent-trace_gdn_decode/database/checkpoints/checkpoint-checkpoint-iter-10-10/best_solution.json` |
| GDN prefill | `full-agent/gdn/Full-agent-trace_gdn_prefill/database/checkpoints/checkpoint-checkpoint-iter-10-10/best_solution.json` |
| DSA sparse attention | `full-agent/dsa/Full-agent-trace_sparse_attn/database/checkpoints/checkpoint-checkpoint-iter-20-20/best_solution.json` |
| DSA top-k indexer | `full-agent/dsa/Full-agent-trace_topk_indexer/database/checkpoints/checkpoint-checkpoint-iter-14-13/best_solution.json` |
| MoE FP8 routing | `full-agent/moe/Full_agent_trace/database/checkpoints/checkpoint-checkpoint-iter-40-37/best_solution.json` |

### Reproducing a Full-Agent Run

Each Full-Agent operator group is a self-contained reproducibility package using Python
3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
# Select operator group: <group> = gdn or dsa.
cd full-agent/<group>/agent
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

cd ..                                    # back to full-agent/<group>/
export LLM_API_KEY="..."
export LLM_BASE_URL="..."   # dsa only; gdn hard-codes url
./run_<task>.sh                          # outputs land in ./<task>_run/
```

For MoE, create the environment from `full-agent/moe/agent/loongflow/`, then
return to `full-agent/moe/` before launching `./run_moe.sh`.

The launcher renders `task_config.yaml` via `envsubst` and invokes
`agents/math_agent/math_evolve_agent.py` with the task prompt, initial
seed JSON, and Modal-side evaluator. Once launched, the same PES loop
handles planning, code generation, evaluation, summarization,
checkpointing, and parent selection without manual intervention. See
[`full-agent/README.md`](./full-agent/README.md) for Full-Agent package details
and the per-task analysis in the technical report.

## License

This repository is licensed under the [Apache License 2.0](./LICENSE).
