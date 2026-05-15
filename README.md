# MLSys 2026 FlashInfer Contest Submissions

Two technical reports accompany this repository:

- [Agent-assisted report](./agent-assisted/report.pdf) — human-in-the-loop
  harness/controller workflow with curated retained kernels.
- [Full-agent report](./full-agent/report.pdf) — *Full-Agent Kernel Generation
  for FlashInfer @ MLSys 2026*  — autonomous LoongFlow-derived PES system that
  closes the loop after task setup.

Agent-assisted and full-agent reproducibility package for MLSys 2026 FlashInfer
AI Kernel Generation Contest submissions: kernels, agent workflows, skills,
configs, writeup, benchmark artifacts, and full optimization records.

This repository is organized into two top-level tracks:

- `agent-assisted/`: the compact public snapshot of our MLSys 2026 FlashInfer
  AI Kernel Generation Contest submissions for the NVIDIA Track.
- `full-agent/`: autonomous Full-Agent submissions produced end-to-end by an
  agent system adapted from **Baidu Baige LoongFlow**. After task setup, no
  human edits intermediate kernels, selects parents, filters failures, or
  steers the next implementation direction. Three sub-tracks (`dsa/`, `gdn/`,
  `moe/`) cover all five contest kernels, and every iteration's planner /
  executor / evaluator / summarizer artifact is preserved on disk for audit.

The `agent-assisted/` package keeps the final kernel submission surface small
while preserving the scripts and skills needed to reproduce, evaluate, and
continue optimization.

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
    |-- report.pdf
    |-- dsa/      DSA — 2 CUDA kernels + LoongFlow agent + full traces
    |-- gdn/      GDN — 2 CUDA kernels + LoongFlow agent + full traces
    `-- moe/      MoE — 1 Triton kernel + LoongFlow agent + full trace
```

Each kernel directory contains:

- `config.toml`: FlashInfer submission metadata.
- `solution/`: the actual Triton or CUDA source loaded by the evaluator.
- `artifacts/`: retained benchmark evidence from the selected submission.
- `README.md`: per-kernel notes, entry point, candidate name, and retained result.

## Kernels

| Track | Kernel | Retained result |
| --- | --- | --- |
| MoE FP8 | [Block-scale routing](./agent-assisted/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/)<br>`kernel.py::run` | 19/19 passed<br>0.289740 ms, three-repeat mean |
| Gated DeltaNet | [Decode QK4](./agent-assisted/gdn_decode_qk4_v8_d128_k_last/)<br>`kernel.py::kernel_hybrid_dispatch` | 54/54 passed<br>0.006201 ms average |
| Gated DeltaNet | [Prefill QK4](./agent-assisted/gdn_prefill_qk4_v8_d128_k_last/)<br>`kernel.py::kernel_prefill_hybrid` | 100/100 passed<br>0.051992 ms average |
| DeepSeek Sparse Attention | [Sparse attention](./agent-assisted/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/)<br>`kernel.py::run` | 23/23 passed<br>0.011128 ms average |
| DeepSeek Sparse Attention | [Top-k indexer](./agent-assisted/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/)<br>`kernel.cu::kernel_cuda` | 128/128 passed<br>0.006893 ms average |

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
[Full-agent report](./full-agent/report.pdf) Table 1 for trajectories.

| Track | Task | Impl. | Kernel latency | Reference latency | Best iter. | Solution ID |
| --- | --- | --- | --- | --- | --- | --- |
| GDN | [Decode](./full-agent/gdn/) | CUDA | **0.0084 ms** | 116.783 ms | 10 | `01bea42b` |
| GDN | [Prefill](./full-agent/gdn/) | CUDA | **0.7020 ms** | 1236.742 ms | 10 | `ed639eed` |
| DSA | [Sparse attention](./full-agent/dsa/) | CUDA | **0.0272 ms** | 3.421 ms | 20 | `94736692` |
| DSA | [Top-k indexer](./full-agent/dsa/) | CUDA | **0.0356 ms** | 3.256 ms | 14 | `dc51d5fc` |
| MoE | [FP8 block-scale routing](./full-agent/moe/) | Triton | **1.448 ms** | 16.542 ms | 40 | `6da80c61` |

### System design (summary)

The Full-Agent system keeps LoongFlow's directed evolutionary substrate and
replaces its generic task interface with FlashInfer operator contracts,
CUDA / Triton executors, official-style evaluator bindings, trace
packaging, and final-solution export. Five design contributions (report §2):

- **LoongFlow-derived autonomous PES runtime** — planner, executor,
  evaluator, summarizer, database, and finalizer roles. The same runtime
  is reused across GDN, DSA, and MoE; only the task prompts, evaluator
  bindings, and final export paths vary by contest definition.
- **Evidence-conditioned prompt and memory policy** — the planner prompt
  is rebuilt at every iteration from the operator contract plus evolving
  evidence (parent solutions, archived evaluator records, compile and
  correctness failures, latency measurements, summarizer lessons). The
  prompt is therefore a state-dependent search policy rather than a static
  instruction.
- **Shape-specialized kernel search without manual intervention** — the
  agent introduces dispatch paths or specialized kernels for distinct
  shape regimes inside the autonomous loop. DSA top-k uses three branches
  for short / medium / long sequences; DSA sparse attention keeps both a
  fused online-softmax path and a split-K fallback; MoE specializes its
  two grouped FP8 GEMMs.
- **Block-level multi-model switching** — because the search state lives
  in the trace database rather than in a single chat context, planner /
  executor models can be swapped at block boundaries (broad exploration →
  code-heavy refinement → conservative review) while preserving parent
  ids, evaluator logs, and handoff summaries.
- **Auditable full-agent trajectories** — every iteration's planner,
  executor, evaluator, and summarizer artifacts are retained on disk so
  reviewers can inspect which candidate failed, which parent was
  preserved, and where the agent shifted from broad exploration to narrow
  exploitation.

### Sub-tracks

| Sub-track | Kernels | Iterations | Trace mirror |
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

The `solution` field of each winning `best_solution.json` is byte-identical
to the corresponding submission file under `solution/`.

### Winning checkpoints

| Task | Winning checkpoint |
| --- | --- |
| GDN decode | `full-agent/gdn/Full-agent-trace_gdn_decode/database/checkpoints/checkpoint-checkpoint-iter-10-10/best_solution.json` |
| GDN prefill | `full-agent/gdn/Full-agent-trace_gdn_prefill/database/checkpoints/checkpoint-checkpoint-iter-10-10/best_solution.json` |
| DSA sparse attention | `full-agent/dsa/Full-agent-trace_sparse_attn/database/checkpoints/checkpoint-checkpoint-iter-20-20/best_solution.json` |
| DSA top-k indexer | `full-agent/dsa/Full-agent-trace_topk_indexer/database/checkpoints/checkpoint-checkpoint-iter-14-13/best_solution.json` |
| MoE FP8 routing | `full-agent/moe/Full_agent_trace/database/checkpoints/checkpoint-checkpoint-iter-40-37/best_solution.json` |

### Reproducing a Full-Agent run

Each sub-track is a self-contained reproducibility package using Python
3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
# Select track: <track> = gdn, dsa, or moe.
cd full-agent/<track>/agent              # or agent/loongflow for moe
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

cd ..                                    # back to the track root
export LLM_API_KEY="..."
export LLM_BASE_URL="..."   # dsa only; gdn/moe hard-code url
./run_<task>.sh                          # outputs land in ./<task>_run/
```

The launcher renders `task_config.yaml` via `envsubst` and invokes
`agents/math_agent/math_evolve_agent.py` with the task prompt, initial
seed JSON, and Modal-side evaluator. Once launched, the same PES loop
handles planning, code generation, evaluation, summarization,
checkpointing, and parent selection without manual intervention. See
[`full-agent/README.md`](./full-agent/README.md) for per-track details
and the per-task analysis in the technical report.

## Full Optimization Records

This combined repository is intentionally curated. The `full-agent/` directory
is the local home for broader full-agent materials, while the original project
repositories keep additional optimization traces, scratch work, references, and
historical context:

| Track | Source repository |
| --- | --- |
| MoE FP8 | [mlsys26-flashinfer-solution-fused-moe](https://github.com/syhya/mlsys26-flashinfer-solution-fused-moe) |
| Gated DeltaNet | [mlsys26-flashinfer-solution-gated-delta-net](https://github.com/syhya/mlsys26-flashinfer-solution-gated-delta-net) |
| DeepSeek Sparse Attention | [mlsys26-flashinfer-solution-sparse-attention](https://github.com/syhya/mlsys26-flashinfer-solution-sparse-attention) |

## Skills

The `agent-assisted/skills/` directory contains the agent workflow instructions
used for optimization and submission handling:

| Skill | Purpose |
| --- | --- |
| [flashinfer-b200-contest-optimizer](./agent-assisted/skills/flashinfer-b200-contest-optimizer/) | FlashInfer B200 contest loop: reference-first recon, shape-aware Modal benchmarking, NCU analysis, and promotion gates. |
| [flashinfer-submission-tagger](./agent-assisted/skills/flashinfer-submission-tagger/) | Submission tag and `config.toml` topology validation helper. |

For further optimization, start from `flashinfer-b200-contest-optimizer`, use
the active kernel directory under `agent-assisted/` as the project root, and
apply the multi sub-agent prompt template below when deeper debate is useful.

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
cd agent-assisted
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
