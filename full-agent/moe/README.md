# FlashInfer Bench · MoE FP8 Triton (LoongFlow Full-Agent)

This repository packages the **MoE FP8 block-scale** submission for the FlashInfer / MLSys'26 contest, together with the **full-agent** search trajectory that produced the winning Triton kernel.

- `solution/triton/kernel.py` — the submitted MoE kernel
- `agent/` — LoongFlow agent framework + the MoE CUDA task directory with every iteration preserved

---

## 1. Final Answer (Best Solution)

The best kernel produced by the agent has already been copied into `solution/triton/kernel.py` (a thin DPS adapter wraps the agent-returned `run(...) -> Tensor` into the contest's `run(..., output)` entry point). `config.toml` is aligned with the Triton entry point, so it needs no edits.

| Task | Submission file | Score | Iteration | solution_id |
| --- | --- | --- | --- | --- |
| moe_fp8_block_scale_ds_routing | `solution/triton/kernel.py` | **1.5985** (≈15.98× speedup, kernel 1.448 ms vs reference 16.542 ms, 19/19 workloads) | 40 | `6da80c61` |

`correctness = 1.0`. `config.toml` sets `entry_point = "kernel.py::run"`, `build.language = "triton"`, and `destination_passing_style = true`, matching the DPS `run(..., output)` adapter in `kernel.py`.

Pack into the evaluator's expected `solution.json`:

```bash
python3 scripts/pack_solution.py
```

### Full agent trace (top-level copy)

For reviewer convenience, the **complete** LoongFlow search history that produced the winning kernel is also mirrored at the repo root under `Full_agent_trace/`, so it can be browsed without descending into `agent/loongflow/...`:

```
Full_agent_trace/
├── database/     island-model checkpoints — `checkpoint-checkpoint-iter-{K}-{N}/best_solution.json` (+ metadata / solutions)
├── iteration/    per-round planner / executor / summarizer I/O for rounds 1..40
└── evaluator/    per-evaluation logs (`eval_<hash>/` with `llm_code_*.py` + `result.json`)
```

The winning kernel is at:

- `Full_agent_trace/database/checkpoints/checkpoint-checkpoint-iter-40-37/best_solution.json` (solution_id `6da80c61`, parent `ba626bb8`)

Contents are identical to `agent/loongflow/agents/math_agent/cuda_task/mlsys26/moe/output/`; §2 below documents the same directory schema in more detail.

---

## 2. Repository Layout

```
flashinfer-bench-loongflow-fullagent-moe/
├── README.md                               this file
├── pyproject.toml                          submission metadata
├── config.toml                             entry_point / binding / DPS flag
├── scripts/                                official pack / local / modal runners
│
├── solution/
│   └── triton/kernel.py                    ★ final submission
│
└── agent/                                  LoongFlow framework + task dir
    └── loongflow/                          (framework + agent live here)
        ├── src/loongflow/                  framework (agentsdk + pes/react)
        ├── agents/math_agent/              PES agent implementation
        │   └── cuda_task/mlsys26/moe/
        │       ├── task_config.yaml        LLM config (env-var driven)
        │       ├── task_prompt.txt         task description for the agent
        │       ├── eval_program_modal.py   Modal-side evaluator
        │       ├── moe_fp8_block_scale_*.json   initial kernel seed
        │       ├── run_contest.sh          in-place launcher
        │       └── output/                 see the "Full iteration trace" block below
        ├── run_math.sh                     framework-generic launcher
        ├── README.md / README_zh.md / AGENTS.md
        └── pyproject.toml / uv.lock
```

### Full iteration trace

Every Planner / Executor / Summarizer I/O, every population snapshot, and every evaluation log is kept on disk. The location below holds the **complete** agent history that produced the final kernel:

- **`agent/loongflow/agents/math_agent/cuda_task/mlsys26/moe/output/`** — 40 iterations

The `output/` directory contains three subtrees:

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

Traceability shortcut:

| Task | Initial seed | Winning checkpoint | Parent |
| --- | --- | --- | --- |
| moe | `.../moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json` | `.../output/database/checkpoints/checkpoint-checkpoint-iter-40-37/best_solution.json` | `ba626bb8` |

The `solution` field of the winning `best_solution.json` is identical to `_run_impl_agent` in `solution/triton/kernel.py` (only a DPS `run(..., output)` wrapper was appended; see the module docstring in `kernel.py`).

---

## 3. Running the Agent

### 3.1 Environment

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
cd agent/loongflow
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### 3.2 LLM credentials (env-var driven)

In `task_config.yaml` only the `api_key` field is env-var driven (`${LLM_API_KEY}`); `url` and `model` are hard-coded for this task:

```yaml
llm_config:
  url: "https://api.chatanywhere.tech/v1"
  api_key: "${LLM_API_KEY}"
  model: "openai/claude-opus-4-7"
```

Set the key before launching:

```bash
export LLM_API_KEY="sk-..."
```

### 3.3 Launch

Two equivalent launchers are provided.

**Option A — top-level launcher (recommended)** · `run_moe.sh` at the repo root:

```bash
export LLM_API_KEY="sk-..."
./run_moe.sh
```

It sets `PROJECT_ROOT=agent/loongflow`, renders `task_config.yaml` through `envsubst` into a temp file, creates a fresh working directory `./moe/` (so a new run does **not** overwrite the archived `agent/loongflow/agents/math_agent/cuda_task/mlsys26/moe/output/`), then invokes `agents/math_agent/math_evolve_agent.py`. Requires `gettext`-flavored `envsubst` in `PATH`.

**Option B — in-place launcher** · `agent/loongflow/agents/math_agent/cuda_task/mlsys26/moe/run_contest.sh`:

```bash
export LLM_API_KEY="sk-..."
cd agent/loongflow/agents/math_agent/cuda_task/mlsys26/moe
./run_contest.sh
```

Same agent entry point; outputs land in the task's own `output/` (the path shown in §2). Use this when reproducing **on top of** the archived trace.

Both scripts call `agents/math_agent/math_evolve_agent.py` with `--config task_config.yaml --task-file task_prompt.txt --initial-file moe_fp8_block_scale_*.json --eval-file eval_program_modal.py`.

---

## 4. About LoongFlow

**LoongFlow** is an evolutionary agent framework built around a **Plan–Execute–Summary (PES)** loop rather than the standard ReAct inner loop. It targets long-horizon code-search problems — CUDA / Triton kernel optimization, AutoML, algorithm discovery — where iterative refinement backed by strong memory of past attempts matters more than reactive tool use.

### PES loop

- **Planner** — consumes the evolutionary database (scored solutions, population / island state, previous summaries) and produces a concrete improvement plan for the next kernel. Plans are strategic (e.g. "fuse FP8 dequant into GEMM", "K-split GEMM1 for small T", "M-outer tile ordering for GEMM2") rather than token-level edits.
- **Executor** — realizes the plan as a code change. Spawns one or more children (one subdir per child under `iteration/{K}/executor/{M_N}/`), each producing a fresh candidate kernel plus full tool/history trace.
- **Evaluator** — compiles and benchmarks each candidate on the remote (Modal) GPU, returning `(score, correctness, speedup)` to the database.
- **Summary** — distills what worked and what failed this round; the distilled lesson is fed back into the next planner's context.

### Evolutionary database

LoongFlow maintains an **island-model population** inside `database/checkpoints/.../metadata.json`. New solutions are admitted based on score and novelty; parent selection for the next generation draws from those islands. This is why every winning kernel carries a `parent_id` field, and why the full lineage of a submission can be walked backwards through `metadata.json` + `best_solution.json` snapshots (e.g. `iter-40-37` ← `ba626bb8` → … → initial seed).

### Stack

- Python 3.12, dependencies managed by `uv` (`agent/loongflow/pyproject.toml`, `agent/loongflow/uv.lock`)
- Pydantic for typed config / messages; async agent runtime
- LLM-agnostic via a minimal switch layer (OpenAI / DeepSeek / Gemini / any OpenAI-compatible endpoint); configured per task in `task_config.yaml`

### Layout inside `agent/loongflow/src/loongflow/`

- `framework/base/` — abstract agent / tool / memory interfaces
- `framework/pes/` — the PES implementation (`planner`, `executor`, `evaluator`, `database`, `context`)
- `framework/react/` — a plain ReAct agent (not used by this task)
- `agentsdk/` — reusable building blocks: `tools/`, `memory/` (grade + evolution memory), `models/`, `message/`, `logger/`, `token/`

The MoE task in this repo lives under `agent/loongflow/agents/math_agent/cuda_task/mlsys26/moe/` and plugs into PES via `agent/loongflow/agents/math_agent/math_evolve_agent.py`. See `agent/loongflow/README.md` and `agent/loongflow/AGENTS.md` for deeper framework docs.

---

