# FlashInfer Bench · DSA Sparse Attention + Top-K Indexer (LoongFlow Full-Agent)

This repository packages two **DeepSeek Sparse Attention (DSA)** submissions for the FlashInfer / MLSys'26 contest, together with the **full-agent** search trajectories that produced the winning kernels.

- `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/` — Sparse Attention forward
- `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/` — FP8 Top-K indexer
- `agent/` — LoongFlow framework + the two CUDA task directories with every iteration preserved

---

## 1. Final Answer (Best Solutions)

The best kernels produced by the agent have already been copied into the corresponding `solution/cuda/kernel.cu`. `config.toml` and `binding.py` are aligned with the exported `dsa_forward` symbol, so they need no edits.

| Task | Submission file | Score | Iteration | solution_id |
| --- | --- | --- | --- | --- |
| dsa_sparse_attn | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/solution/cuda/kernel.cu` | **12.6350** (≈126.35× speedup) | 20 | `94736692` |
| dsa_topk_indexer | `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/solution/cuda/kernel.cu` | **9.5640** (≈95.64× speedup) | 14 | `dc51d5fc` |

Both have `correctness = 1.0`. Each `config.toml` sets `entry_point = "kernel.cu::dsa_forward"` and `binding = "torch"`, matching the `PYBIND11_MODULE` export inside the kernel.

Pack into the evaluator's expected `solution.json`:

```bash
cd dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
python3 ../scripts/pack_solution.py
```

### Full agent trace (top-level copy)

For reviewer convenience, the **complete** LoongFlow search histories that produced the winning kernels are also mirrored at the repo root, so they can be browsed without descending into `agent/...`:

- `Full-agent-trace_sparse_attn/` — 29 iterations (mirror of `agent/agents/math_agent/cuda_task/mlsys26/dsa_sparse_attn/output/`)
- `Full-agent-trace_topk_indexer/` — 20 iterations (mirror of `agent/agents/math_agent/cuda_task/mlsys26/dsa_topk_indexer/output/`)

Each top-level trace has the same three subtrees as the in-task `output/` (see §2):

```
Full-agent-trace_<task>/
├── database/     island-model checkpoints — `checkpoint-checkpoint-iter-{K}-{N}/best_solution.json` (+ metadata / solutions)
├── iteration/    per-round planner / executor / summarizer I/O
└── evaluator/    per-evaluation logs (`eval_<hash>/` with `llm_code_*.py` + `result.json`)
```

Winning checkpoints:

- `Full-agent-trace_sparse_attn/database/checkpoints/checkpoint-checkpoint-iter-20-20/best_solution.json` (solution_id `94736692`, parent `c06e99d2`)
- `Full-agent-trace_topk_indexer/database/checkpoints/checkpoint-checkpoint-iter-14-13/best_solution.json` (solution_id `dc51d5fc`, parent `2fb50fd2`)

---

## 2. Repository Layout

```
flashinfer-bench-loongflow-fullagent-dsa/
├── README.md                               this file
├── EVALUATION.md / FAQ.md                  contest-provided docs
├── pyproject.toml                          submission metadata
├── run_dsa_sparse_attn.sh                  top-level launcher (envsubst + venv)
├── run_dsa_topk_indexer.sh
├── scripts/                                official pack / local / modal runners
│
├── dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/
│   ├── config.toml                         entry_point / binding config
│   └── solution/cuda/{kernel.cu, binding.py}            ★ final submission
│
├── dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/
│   ├── config.toml
│   └── solution/cuda/{kernel.cu, binding.py}            ★ final submission
│
└── agent/                                  LoongFlow framework + task dirs
    ├── src/loongflow/                      framework (agentsdk + pes/react)
    ├── agents/math_agent/                  PES agent implementation
    │   └── cuda_task/mlsys26/
    │       ├── dsa_sparse_attn/
    │       │   ├── task_config.yaml        LLM config (env-var driven)
    │       │   ├── task_prompt.txt         task description for the agent
    │       │   ├── eval_program_modal.py   Modal-side evaluator
    │       │   ├── dsa_sparse_attention_*.json   initial kernel seed
    │       │   ├── run_dsa_sparse_attn.sh  in-place launcher
    │       │   └── output/                 see the "Full iteration trace" block below
    │       └── dsa_topk_indexer/           same structure (20 iterations)
    ├── run_math.sh                         framework-generic launcher
    ├── README.md / README_zh.md / AGENTS.md
    └── pyproject.toml / uv.lock
```

### Full iteration trace

Every Planner / Executor / Summarizer I/O, every population snapshot, and every evaluation log is kept on disk. The two locations below hold the **complete** agent history that produced the final kernels:

- **`agent/agents/math_agent/cuda_task/mlsys26/dsa_sparse_attn/output/`** — 29 iterations
- **`agent/agents/math_agent/cuda_task/mlsys26/dsa_topk_indexer/output/`** — 20 iterations

Each `output/` directory contains three subtrees:

- **`database/checkpoints/checkpoint-checkpoint-iter-{K}-{N}/`** — one per iteration
  - `best_solution.json` — best kernel of that iteration (full code + score + `parent_id`)
  - `metadata.json` — population / island state used to pick parents for the next round
  - `solutions/` — all candidates admitted to the evolutionary database that round
- **`iteration/{K}/`** — per-round working directory
  - `planner/` — plan prompt + LLM response (strategy for round K)
  - `executor/{M_N}/` — each child's generated code, `history.json`, `history.log`
  - `executor/best_solution.py` + `best_evaluation.json` — round-K winner
  - `summarizer/` — distilled feedback fed into the next planner
- **`evaluator/eval_<hash>/`** — one directory per evaluation call
  - `llm_code_*.py` — kernel under test
  - `evaluation_process.log` — compile / run / scoring log
  - `result.json` — final score payload

Traceability shortcuts:

| Task | Initial seed | Winning checkpoint | Parent |
| --- | --- | --- | --- |
| sparse_attn | `.../dsa_sparse_attn/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json` | `.../output/database/checkpoints/checkpoint-checkpoint-iter-20-20/best_solution.json` | `c06e99d2` |
| topk_indexer | `.../dsa_topk_indexer/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json` | `.../output/database/checkpoints/checkpoint-checkpoint-iter-14-13/best_solution.json` | `2fb50fd2` |

The `solution` field of each winning `best_solution.json` is byte-identical to the corresponding `solution/cuda/kernel.cu` at the repo root.

---

## 3. Running the Agent

### 3.1 Environment

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
cd agent
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```


### 3.2 LLM credentials (env-var driven)

Both `task_config.yaml` files expose `url` and `api_key` as `${LLM_BASE_URL}` and `${LLM_API_KEY}`; the launcher uses `envsubst` to render the template at runtime. The `model` field is hard-coded per task and is **not** overridden by an environment variable.

```bash
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.chatanywhere.tech/v1"
```

### 3.3 Launch

Top-level launchers (recommended — each task writes into an isolated run directory at the repo root, so parallel tasks cannot clobber each other):

```bash
./run_dsa_sparse_attn.sh      # outputs land in ./dsa_sparse_attn_run/
./run_dsa_topk_indexer.sh     # outputs land in ./dsa_topk_indexer_run/
```

In-place launchers (outputs land in the task's own `output/`, i.e. the same path shown in §2):

```bash
cd agent/agents/math_agent/cuda_task/mlsys26/dsa_sparse_attn
./run_dsa_sparse_attn.sh
```

What each script does: validate `LLM_API_KEY` / `LLM_BASE_URL` → render the yaml to a temp file via `envsubst` → invoke `agents/math_agent/math_evolve_agent.py` with `--task-file task_prompt.txt`, `--initial-file <seed>.json`, `--eval-file eval_program_modal.py`, and the rendered config.

---

## 4. About LoongFlow

**LoongFlow** is an evolutionary agent framework built around a **Plan–Execute–Summary (PES)** loop rather than the standard ReAct inner loop. It targets long-horizon code-search problems — CUDA kernel optimization, AutoML, algorithm discovery — where iterative refinement backed by strong memory of past attempts matters more than reactive tool use.

### PES loop

- **Planner** — consumes the evolutionary database (scored solutions, population / island state, previous summaries) and produces a concrete improvement plan for the next kernel. Plans are strategic (e.g. "fuse the split-K path", "move query into shared memory transposed") rather than token-level edits.
- **Executor** — realizes the plan as a code change. Spawns one or more children (one subdir per child under `iteration/{K}/executor/{M_N}/`), each producing a fresh candidate kernel plus full tool/history trace.
- **Evaluator** — compiles and benchmarks each candidate on the remote (Modal) GPU, returning `(score, correctness, speedup)` to the database.
- **Summary** — distills what worked and what failed this round; the distilled lesson is fed back into the next planner's context.

### Evolutionary database

LoongFlow maintains an **island-model population** inside `database/checkpoints/.../metadata.json`. New solutions are admitted based on score and novelty; parent selection for the next generation draws from those islands. This is why every winning kernel carries a `parent_id` field, and why the full lineage of a submission can be walked backwards through `metadata.json` + `best_solution.json` snapshots.

### Stack

- Python 3.12, dependencies managed by `uv` (`agent/pyproject.toml`, `agent/uv.lock`)
- Pydantic for typed config / messages; async agent runtime
- LLM-agnostic via a minimal switch layer (OpenAI / DeepSeek / Gemini / any OpenAI-compatible endpoint); configured per task in `task_config.yaml`

### Layout inside `agent/src/loongflow/`

- `framework/base/` — abstract agent / tool / memory interfaces
- `framework/pes/` — the PES implementation (`planner`, `executor`, `evaluator`, `database`, `context`)
- `framework/react/` — a plain ReAct agent (not used by these tasks)
- `agentsdk/` — reusable building blocks: `tools/`, `memory/` (grade + evolution memory), `models/`, `message/`, `logger/`, `token/`

The two CUDA tasks in this repo live under `agent/agents/math_agent/cuda_task/mlsys26/` and plug into PES via `agent/agents/math_agent/math_evolve_agent.py`. See `agent/README.md` and `agent/AGENTS.md` for deeper framework docs.

---
