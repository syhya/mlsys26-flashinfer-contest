# MLSys 2026 FlashInfer Contest · Full-Agent Track

> **Full-Agent Kernel Generation for FlashInfer @ MLSys 2026**
> Chenyu Ma, Yue Shui, Hangfei Xu, Shengzhao Wen, Yanpeng Wang — Baidu Inc.
> [Technical report (PDF)](./report.pdf)

This directory holds the **Full-Agent** submissions for the MLSys 2026
FlashInfer AI Kernel Generation Contest. Every kernel under this tree was
produced end-to-end by an autonomous CUDA / Triton kernel-search system
adapted from **Baidu Baige LoongFlow** — after task setup, no human edits
intermediate kernels, selects parents, filters failures, or steers the next
implementation direction. The complete search history (planner / executor /
evaluator / summarizer I/O, population snapshots, and per-evaluation logs)
is preserved on disk for reproducibility and audit.

For the compact, manually-curated submission package see `../agent-assisted/`.
For an overview of the whole repository see `../README.md`.

## Final Submitted Kernels

All five final kernels pass correctness on the official evaluator. Kernel
latency is the primary metric (lower is better); reference latency is the
supplied FlashInfer baseline included as contextual context only.

| Track | Task | Impl. | Kernel latency | Reference latency | Best iter. | Solution ID |
| --- | --- | --- | --- | --- | --- | --- |
| GDN | Decode | CUDA | **0.0084 ms** | 116.783 ms | 10 | `01bea42b` |
| GDN | Prefill | CUDA | **0.7020 ms** | 1236.742 ms | 10 | `ed639eed` |
| DSA | Sparse attention | CUDA | **0.0272 ms** | 3.421 ms | 20 | `94736692` |
| DSA | Top-k indexer | CUDA | **0.0356 ms** | 3.256 ms | 14 | `dc51d5fc` |
| MoE | FP8 block-scale routing | Triton | **1.448 ms** | 16.542 ms | 40 | `6da80c61` |

See the technical report for per-iteration trajectories, NCU evidence,
and the multi-model switching ablation. Per-task reproducibility details
live in each sub-track README.

## Sub-Tracks

Three independent agent runs cover the five contest kernels:

| Sub-track | Kernels | Iterations | Trace mirror |
| --- | --- | --- | --- |
| [`dsa/`](./dsa/) | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`<br>`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` | 29 / 20 | `Full-agent-trace_sparse_attn/`, `Full-agent-trace_topk_indexer/` |
| [`gdn/`](./gdn/) | `gdn_decode_qk4_v8_d128_k_last`<br>`gdn_prefill_qk4_v8_d128_k_last` | 10 / 10 | `Full-agent-trace_gdn_decode/`, `Full-agent-trace_gdn_prefill/` |
| [`moe/`](./moe/) | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` | 40 | `Full_agent_trace/` |

## Layout

```text
full-agent/
├── README.md            this file
├── report.pdf           technical report (Full-Agent Kernel Generation for FlashInfer)
├── dsa/                 DeepSeek Sparse Attention — 2 CUDA kernels + LoongFlow
├── gdn/                 Gated DeltaNet — 2 CUDA kernels + LoongFlow
└── moe/                 MoE FP8 block-scale — 1 Triton kernel + LoongFlow
```

Each sub-track is a self-contained reproducibility package containing:

- `solution/` — the final kernel(s) submitted to the evaluator (CUDA or Triton).
- `config.toml` — FlashInfer submission metadata (`entry_point`, `binding`,
  `language`, DPS flag where applicable).
- `scripts/` — official `pack_solution.py`, `run_local.py`, `run_modal.py`.
- `agent/` — the **LoongFlow** framework plus the per-task directories
  (`task_config.yaml`, `task_prompt.txt`, `eval_program_modal.py`, initial
  seed JSON, in-place launcher).
- `Full-agent-trace_*/` (or `Full_agent_trace/`) — top-level mirror of the
  task's `output/`, so reviewers can browse the full evolutionary history
  without descending into `agent/...`.
- `run_*.sh` — top-level launcher that renders `task_config.yaml` via
  `envsubst` and starts the agent in an isolated working directory.
- `README.md` — per-track notes covering best result, layout, and how to
  reproduce.

### Trace directory schema

Every `Full-agent-trace_*/` (and the in-task `output/` it mirrors) has the
same three subtrees:

- `database/checkpoints/checkpoint-checkpoint-iter-{K}-{N}/`
  - `best_solution.json` — best kernel of iteration K (full code, score, `parent_id`).
  - `metadata.json` — population / island state used to seed the next round.
  - `solutions/` — every candidate admitted to the evolutionary database.
- `iteration/{K}/`
  - `planner/` — plan prompt + LLM response.
  - `executor/{M_N}/` — each child's generated code, `history.json`, `history.log`.
  - `executor/best_solution.py` + `best_evaluation.json` — round-K winner.
  - `summarizer/` — distilled feedback fed into the next planner.
- `evaluator/eval_<hash>/`
  - `llm_code_*.py` — kernel under test.
  - `evaluation_process.log` — compile / run / scoring log.
  - `result.json` — final score payload.

The `solution` field of each winning `best_solution.json` is byte-identical
to the corresponding submission file under `solution/`.

### Winning checkpoints

| Task | Winning checkpoint |
| --- | --- |
| GDN decode | `gdn/Full-agent-trace_gdn_decode/database/checkpoints/checkpoint-checkpoint-iter-10-10/best_solution.json` |
| GDN prefill | `gdn/Full-agent-trace_gdn_prefill/database/checkpoints/checkpoint-checkpoint-iter-10-10/best_solution.json` |
| DSA sparse attention | `dsa/Full-agent-trace_sparse_attn/database/checkpoints/checkpoint-checkpoint-iter-20-20/best_solution.json` |
| DSA top-k indexer | `dsa/Full-agent-trace_topk_indexer/database/checkpoints/checkpoint-checkpoint-iter-14-13/best_solution.json` |
| MoE FP8 routing | `moe/Full_agent_trace/database/checkpoints/checkpoint-checkpoint-iter-40-37/best_solution.json` |

## System Design (summary)

The system keeps LoongFlow's directed evolutionary substrate and replaces
its generic task interface with FlashInfer operator contracts, CUDA / Triton
executors, official-style evaluator bindings, trace packaging, and
final-solution export. Key design contributions (see report §2):

- **LoongFlow-derived autonomous PES runtime** — planner, executor,
  evaluator, summarizer, database, and finalizer roles. The same runtime
  is reused across GDN, DSA, and MoE; only the task prompts, evaluator
  bindings, and final export paths vary by contest definition.
- **Evidence-conditioned prompt and memory policy** — the planner prompt
  is rebuilt at every iteration from the fixed operator contract plus
  evolving evidence (parent solutions, archived evaluator records, compile
  failures, correctness failures, latency measurements, summarizer
  lessons). The prompt is therefore a state-dependent search policy rather
  than a static instruction.
- **Shape-specialized kernel search without manual intervention** — the
  agent is allowed to introduce dispatch paths or specialized kernels for
  distinct shape regimes; the routes are generated and validated inside
  the autonomous loop rather than inserted by a human.
- **Block-level multi-model switching** — because the search state lives in
  the trace database rather than in a single chat context, the system can
  switch planner or executor models at block boundaries (e.g. broad
  exploration → code-heavy refinement → conservative review) while
  preserving parent ids, evaluator logs, and handoff summaries.
- **Auditable full-agent trajectories** — the output is not only a final
  kernel: every iteration's planner, executor, evaluator, and summarizer
  artifacts are retained on disk so reviewers can inspect which candidate
  failed, which parent was preserved, and where the agent shifted from
  broad exploration to narrow exploitation.

## Reproducing a Run

Each sub-track is self-contained. Pick one and follow its README. The
common shape is:

```bash
# Select track: <track> = gdn, dsa, or moe.
cd full-agent/<track>/agent              # or agent/loongflow for moe

# Create and activate the Python environment.
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Configure the LLM endpoint.
cd ..                                    # back to the track root
export LLM_API_KEY="..."
export LLM_BASE_URL="..."   # dsa only; gdn/moe hard-code url

# Run one task-specific launcher in the selected repository.
./run_<task>.sh                           # outputs land in ./<task>_run/
```

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required. The
launchers validate credentials, render `task_config.yaml` via `envsubst`,
and invoke `agents/math_agent/math_evolve_agent.py` with the task prompt,
initial seed, and Modal-side evaluator. Once launched, the same PES loop
handles planning, code generation, evaluation, summarization,
checkpointing, and parent selection without manual intervention.

## Packing & Evaluation

Every track ships the standard FlashInfer scripts under `scripts/`:

```bash
# Pack the final kernel into solution.json
python3 scripts/pack_solution.py --config-path <kernel_dir>/config.toml \
  --output /tmp/<kernel>.solution.json

# Local evaluation (needs FIB_DATASET_PATH and a compatible CUDA GPU)
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python3 scripts/run_local.py --config-path <kernel_dir>/config.toml

# Modal B200 evaluation
python3 scripts/run_modal.py --config-path <kernel_dir>/config.toml
```

For DSA and GDN, replace `<kernel_dir>` with the kernel folder under the
track root. For MoE, the track itself is the kernel directory (single
submission), so omit `<kernel_dir>` and use the track's own `config.toml`.

## About LoongFlow

**LoongFlow** is an evolutionary agent framework built around a
**Plan–Execute–Summary (PES)** loop rather than the standard ReAct inner
loop. It targets long-horizon code-search problems — CUDA / Triton kernel
optimization, AutoML, algorithm discovery — where iterative refinement
backed by strong memory of past attempts matters more than reactive tool
use.

- **Planner** — consumes the evolutionary database (scored solutions,
  island state, previous summaries) and produces a strategic improvement
  plan (e.g. "fuse the split-K path", "move query into shared memory
  transposed") rather than token-level edits.
- **Executor** — realizes the plan as a code change. Spawns one or more
  children (one subdir per child under `iteration/{K}/executor/{M_N}/`),
  each producing a fresh candidate plus its full tool/history trace.
- **Evaluator** — compiles and benchmarks each candidate on the remote
  (Modal B200) GPU, returning `(score, correctness, latency)` to the
  database. The evaluator is the sole promotion gate; failed candidates
  are not discarded as noise — they become negative evidence that shapes
  later prompts.
- **Summary** — distills what worked and what failed this round; the
  lesson is fed back into the next planner's context.

LoongFlow maintains an **island-model population** in
`database/checkpoints/.../metadata.json`. New solutions are admitted based
on score and novelty; parent selection draws from those islands, which is
why every winning kernel carries a `parent_id` field and the full lineage
of a submission can be walked backwards through the snapshots.

The framework is LLM-agnostic (OpenAI / DeepSeek / Gemini / any
OpenAI-compatible endpoint), Python 3.12, async runtime, dependencies
managed by `uv`. See each track's `agent/README.md` and `agent/AGENTS.md`
for deeper framework documentation, and `report.pdf` for the full system
design and per-task analysis.

## Citation

If you reference this work, please cite the technical report:

```
Chenyu Ma, Yue Shui, Hangfei Xu, Shengzhao Wen, Yanpeng Wang.
"Full-Agent Kernel Generation for FlashInfer @ MLSys 2026."
FlashInfer AI Kernel Generation Contest @ MLSys 2026, Baidu Inc.
```
