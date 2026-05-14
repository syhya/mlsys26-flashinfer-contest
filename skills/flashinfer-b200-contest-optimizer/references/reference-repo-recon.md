# Reference Repo Recon

Use this file before any optimization round. The goal is to avoid reinventing an operator or missing a proven optimization path that already exists in a local reference repo.

## Discover

Search for reference repos in this order:
- `<project-root>/reference/*`
- embedded support repos inside the same project such as `<project-root>/flashinfer-bench` and `<project-root>/mlsys26-contest`

Treat each immediate child directory containing `.git` as a repo that must be inspected.
Do not inspect sibling local repos under `../` unless the user explicitly asks for them.

## Refresh

For every discovered repo:
1. Run `git -C <repo> status --short`.
2. If the repo is clean, run `git -C <repo> pull --ff-only`.
3. If the repo is dirty, do not reset or stash it. Run `git -C <repo> fetch --prune origin`, note that the local checkout was not fast-forwarded, and inspect the latest remote state separately if needed.
4. Record the inspected commit with `git -C <repo> rev-parse HEAD`.

Do not claim the recon is current unless the refresh step has been completed for every relevant repo.

## Online Search

If local `reference/` repos do not yield a clear reusable path, search official or otherwise high-quality GitHub repos next. Prefer primary sources over blog posts or reposts.

Seed repos:
- `https://github.com/flashinfer-ai/flashinfer`
- `https://github.com/flashinfer-ai/flashinfer-bench`
- `https://github.com/Dao-AILab/flash-attention`
- `https://github.com/NVIDIA/cutlass`
- `https://github.com/deepseek-ai/DeepGEMM`
- `https://github.com/thu-ml/SageAttention`
- `https://github.com/sgl-project/sglang`
- `https://github.com/vllm-project/vllm`

Use online search to answer questions like:
- Is there already a Blackwell or B200 code path for this primitive?
- Has a nearby project already solved the same scheduling, tiling, sparse-routing, or paged-memory problem?
- Are there benchmark or test cases that expose the same pathological shape regime?

## Inspect

Read the top-level `README` first, then inspect the implementation-heavy directories:
- `csrc/`
- `include/`
- `flashinfer/`
- `flash_attn/`
- `fla/`
- `sgl-kernel/`
- `vllm/csrc/`
- `benchmarks/`
- `tests/`

Look for:
- exact operator matches
- near-match primitives that can be transplanted
- benchmark harnesses or profiling scripts worth reusing
- issue threads, tests, or docs that explain why a kernel is shaped the way it is

## Search

Start from the active definition name and expand to operator-family keywords.

Examples:
- sparse attention / top-k tasks: `sparse`, `attention`, `topk`, `indexer`, `gather`, `scatter`, `paged`, `ragged`, `mask`, `decode`, `prefill`, `split-k`, `segment`, `select`
- MoE tasks: `moe`, `routing`, `gating`, `expert`, `grouped gemm`, `splitk`, `DeepGEMM`
- linear attention or recurrence tasks: `gated delta`, `delta rule`, `scan`, `prefix`, `state`, `chunk`, `recurrent`

Search both kernel names and data-layout clues such as `topk2048`, `ps64`, `ckv`, `kpe`, `fp8`, `bf16`, `paged_kv`, or the relevant head dimensions.

## Decide

Before writing a new candidate, classify each promising hit:
- direct reuse: same or nearly same operator, mostly wiring work
- partial reuse: kernel primitive or schedule worth transplanting
- conceptual lead: not reusable directly, but explains a promising tactic

Bias toward adapting an existing kernel or schedule before starting from scratch.

## Recon Note

Write a compact note with one row per promising hit:

| repo | commit | path | relevance | reusable piece | next action |
| --- | --- | --- | --- | --- | --- |
| `flashinfer` | `<sha>` | `csrc/...` | direct / partial / concept | kernel, schedule, memory trick, launch idea | adapt / benchmark / discard |

For online-only hits, replace `commit` with the inspected default-branch commit or page date, and include the URL in the note.

Do not begin candidate implementation until this note exists, even if the conclusion is "no close reusable operator found".
