---
name: flashinfer-b200-contest-optimizer
description: Optimize MLSYS26 FlashInfer contest operators for NVIDIA B200/Blackwell with a reference-first workflow. Use when tasks require refreshing only the current project's local `reference/` git repos before optimization, searching high-quality official repos when local references are insufficient, deriving workload regimes from contest JSONL, running paired-baseline local gates, exploring 2-3 representative regimes before costly full sweeps, asking the user before expensive multi-GPU evaluation, extracting compact NCU metrics from Modal artifacts, archiving scratch outputs, applying CUDA 13.2 Blackwell tactics, and promoting only on repeatable mean-latency wins.
---

# FlashInfer B200 Contest Optimizer

## Overview

Use this skill to optimize MLSYS26 FlashInfer contest operators for B200, benchmark on Modal, pair each local gate with a same-round baseline run, extract compact NCU metrics from scratch artifacts, and keep reproducible artifacts for submission.

Do not start profiling or candidate writing until the reference repositories for the active operator have been refreshed and inspected. Reuse or adapt an existing operator, kernel primitive, or optimization idea when the reference scan shows one is already close to the active problem.

Assume CUDA 13.2 is the active optimization baseline. When Triton reaches a control-surface ceiling (cluster attributes, DSMEM, launch-policy control, or launch-latency overlap), move the hotspot to `solution/cuda/kernel.cu` instead of forcing a Triton-only path.

Treat the official contest evaluator as the final ground truth:
- Docker image: `flashinfer/flashinfer-ci-cu132:latest`
- hardware: bare-metal B200
- GPU clocks: locked to max with `nvidia-smi -ac 3996,1965`
- timing path: `flashinfer-bench` with `cupti-python`
- process model: `--use-isolated-runner`

Deliverables per optimization round:
- A measurable **latency** improvement with repeat validation (latency is the primary metric — speedup is secondary due to unstable reference baseline).
- A shape-aware NCU matrix showing `band -> workload_uuid -> dominant kernel -> bottleneck -> limiter`.
- A Markdown benchmark summary showing `avg_latency_ms` and `p95_latency_ms` as primary comparison columns, with `avg_speedup` as context only.
- A scratch-safe archive: every full run, single-workload run, NCU report, and candidate kernel copied out of the project root before the next experiment.
- A round decision that explicitly says whether a candidate was archived only, promoted globally, or rejected with the repo restored to the previous best.

## Reference-First Reconnaissance

Treat reference-repo refresh and code reconnaissance as a mandatory stage `0`, not optional background reading.

Before the first optimization step of a round:
1. Discover the active project root and search for local reference repos in this order:
   - `<project-root>/reference/*`
   - embedded support repos inside the same project such as `<project-root>/flashinfer-bench`, `<project-root>/mlsys26-contest`, or operator-specific starter-kit mirrors
2. Do **not** scan sibling local repos under `../` just because they exist. Ignore `../mlsys26-flashinfer-solution*` unless the user explicitly asks for them.
3. If the local `reference/` scan does not produce a clear reusable implementation, search official or otherwise high-quality online repos next, preferably GitHub primary sources.
4. Start the online search from these seed repos when they match the operator family:
   - `flashinfer-ai/flashinfer`
   - `flashinfer-ai/flashinfer-bench`
   - `Dao-AILab/flash-attention`
   - `NVIDIA/cutlass`
   - `deepseek-ai/DeepGEMM`
   - `sgl-project/sglang`
   - `vllm-project/vllm`
5. Treat every immediate child directory containing `.git` as a reference repo that must be checked.
6. Refresh each clean local repo with `git pull --ff-only` before reading code. If a repo is dirty, do not stash or reset it; record the blocker, run `git fetch --prune origin`, and inspect the latest remote state without clobbering local work.
7. Read the repo's top-level `README`, then inspect the implementation directories most likely to contain reusable kernels or operator building blocks, usually `csrc/`, `include/`, `flashinfer/`, `flash_attn/`, `fla/`, `sgl-kernel/`, `vllm/csrc/`, `benchmarks/`, and `tests/`.
8. Search each refreshed repo for both exact operator matches and adjacent primitives. For sparse attention or top-k style tasks, search terms should include the active definition name plus primitive keywords such as `sparse attention`, `topk`, `indexer`, `gather`, `scatter`, `paged`, `ragged`, `decode`, `prefill`, `split-k`, `segment`, `select`, `mask`, `routing`, and the active tensor/layout abbreviations.
9. Write a short recon note before candidate work that records:
   - repo and commit inspected
   - candidate file or kernel path
   - what primitive or optimization idea is reusable
   - whether it is a direct drop-in, a partial borrowing target, or only a conceptual lead
10. If the scan finds an existing implementation or near-match, start from adaptation or transplantation first. Only start greenfield kernel design after ruling out the reusable paths.

Use `references/reference-repo-recon.md` as the detailed playbook for repo discovery, refresh rules, and recon-note structure.

## Shape-Aware Rule

Do not optimize against one representative workload only, and do not treat a local win as a global promotion.

Maintain at least three representative workload bands in the full shape matrix that cover the natural shape regimes of the operator. For early gating on very large definitions, it is acceptable to start from 2-3 representative workloads drawn from that matrix, for example:
- short / medium / long sequence or token counts
- low / medium / high tokens per expert
- decode-like / transition / prefill-like shapes

For each band:
- run NCU independently
- record the dominant kernel, bottleneck class, and limiting resource
- decide whether the hotspot cause is truly shape-dependent
- if one workload or tiny regime contributes roughly one third or more of total latency, keep it as its own explicit outlier regime even if only 1-2 UUIDs sit there

Introduce shape-specialized routes only when profiler evidence shows the bottleneck or the limiting resource differs by band.

When a shape split is justified:
- first push each justified band-specific or outlier-specific path to its local limit in isolation
- archive any real local regime win, even if it is not yet safe to promote globally
- only then integrate those paths into one submission kernel or one entrypoint with cheap dispatch / internal branching
- promote only if the representative triplet stays healthy and the full-benchmark mean latency improves versus the baseline mean

When the workload count is greater than 40:
- start with 2-3 representative regimes/workloads drawn from the dominant varying axes plus the latency-heavy or boundary regimes
- do not open the round with a full multi-GPU sweep across the whole definition
- if one representative regime shows a clear, repeatable latency win above the recent noise floor, keep the route and archive the code even before global proof; later integrate it behind a cheap `if` / threshold dispatch keyed on the real workload variables
- ask the user to confirm the spend before the first full-workload multi-GPU sweep or repeat set, because those runs are expensive in both elapsed time and compute budget

## Contest-Actual Shape Discovery

Do not hardcode axis names, thresholds, or representative shapes into the skill.

Before profiling or writing candidates:
1. Read the active definition from `config.toml` (`solution.definition`).
2. Resolve the matching workload file under `${FIB_DATASET_PATH}/workloads/**/<definition>.jsonl` or local `mlsys26-contest/workloads/**/<definition>.jsonl`.
3. Extract the real `axes` and workload UUIDs from that JSONL.
4. Join them with per-workload latency from `benchmark_detailed_results.json` or `solution/triton/retained_run.json`.
5. Identify which axes actually vary and which of them correlate with latency share or bottleneck changes.
6. Build 3-4 representative regimes from the dominant axis or axis combinations, not from a fixed template.

Guidelines:
- If only one axis varies materially, build bands along that axis.
- If multiple axes vary materially, build coupled regimes (for example small-batch+short-context vs large-batch+long-context) instead of pretending one axis explains everything.
- Prioritize deep NCU and kernel-body work by latency share, not by how many workloads fall into a regime. A regime with only 1-2 workloads can still dominate total latency.
- If the dataset contains a latency-significant transition outlier, keep it as its own regime.
- When the active definition changes, recompute the regimes from the workload file. Do not reuse thresholds from a previous operator.

## CUDA 13.2 Ceiling Features

Treat CUDA 13.2 / Blackwell-only features as explicit branches in the search tree, not as generic cleanup.

Reach for these only when NCU or the benchmark shape matrix points at the matching limiter:
- **Cluster Launch Control (CLC)**: use for persistent or block-stealing kernels where tail effects or irregular work assignment leave SMs idle near the end of the launch.
- **Thread-Block Clusters + DSMEM**: use when cross-CTA communication or reuse is hot enough that replacing global-memory exchange with cluster-local exchange can remove L2 / DRAM pressure. Always compare portable cluster size `8` against B200-only nonportable cluster size `16`.
- **L2 persistence / access-policy window**: use for read-mostly tensors or metadata reused across many CTAs within one regime, such as routing maps, expert metadata, quant scales, or repeatedly touched K/V fragments.
- **Shared-memory carveout + >48 KB dynamic shared memory**: use when reuse is real and occupancy loss is still acceptable. On B200, explicitly test carveout choices instead of assuming the default is near-optimal.
- **Programmatic Dependent Launch (PDL)**: use only when a producer/consumer split still makes sense, and the secondary kernel has independent preamble work that can overlap the primary kernel tail.
- **Memory synchronization domains**: use when peer / NVLink / communication traffic causes fence interference. Do not introduce domains into a purely local kernel path without evidence.

Hard gates:
- Do not introduce any CUDA 13.2 feature unless the bottleneck table says what resource it is expected to relieve.
- Do not keep a 13.2-specific path if it improves one outlier regime but loses on the representative triplet mean.
- If a needed launch or memory feature is unavailable in Triton, stop iterating in Triton and branch a CUDA implementation.

## Official Evaluation Parity

Do not treat a Modal win as submission-ready evidence by itself.

Before promoting a candidate as the new best:
- run at least one parity benchmark in the official container/tooling shape
- use `flashinfer-bench run` with `--use-isolated-runner`
- match the official definition-specific flags, including the special `gdn_prefill` warmup/iteration/trial settings
- prefer FlashInfer and FlashInfer-Bench built from latest `main` inside the contest image over ad hoc local installs
- keep `config.toml` aligned with exactly one promoted solution per definition

Submission rules:
- for multiple git tags targeting the same definition, only the latest tag is evaluated
- tags for different definitions are evaluated independently
- before each evaluation checkpoint, ensure the intended tag is pushed and points at the exact promoted kernel
- if using a private repo, keep `flashinfer-bot` read access intact

## Workflow

0. Reference repos, existing operators, and idea harvest.
- Complete the mandatory recon flow in `references/reference-repo-recon.md` before running new NCU or writing a new candidate.
- Refresh every discovered clean local `reference/` repo with `git pull --ff-only`.
- If local references are insufficient, search the online seed repos in `references/reference-repo-recon.md` before writing a greenfield kernel.
- Search for existing kernels, operator-adjacent implementations, or benchmark scripts that already solve all or part of the active problem.
- Record which repo/path/idea you will borrow from, or explicitly note that no reusable implementation was found.

1. Baseline truth, reproducibility, and shape matrix.
- Keep `config.toml` and `solution/triton/kernel.py` (or `solution/cuda/kernel.cu`) reproducible.
- Keep one official-parity command for the active definition in the round notes. The final promotion gate must be reproducible inside `flashinfer/flashinfer-ci-cu132:latest`.
- Before writing candidates, identify the previous best from round notes or retained summaries and write down two numbers:
  - baseline mean latency
  - best retained single run latency
- Use the baseline **mean** as the true promotion threshold. Treat the retained single run as a favorable sample only.
- Derive the representative matrix from the real workload JSONL for the active definition. Do **not** invent synthetic bands.
- Identify the 1-2 dominant varying axes first. Do not assume `seq_len` is the right abstraction for every operator.
- At minimum keep 3 representative regimes in the full matrix that cover the important workload space. For >40-workload early exploration, it is acceptable to start from 2-3 representative workloads/regimes drawn from that matrix.
- If multiple axes vary materially, choose representative workloads by coupled regimes rather than by one axis only.
- If the real dataset contains a latency-significant transition outlier, keep it as its own regime instead of forcing it into a neighboring band.
- If one workload or tiny regime contributes roughly one third or more of total latency, promote it to an explicit outlier regime even if neighboring workloads stay on the default path.
- Record the chosen workload UUIDs, exact axes, and latency share in the round summary.
- Run the current baseline full sweep for this round when the workload count is modest. For large definitions, postpone the first full sweep until the representative gate shows a promising route.
- When possible, anchor the round with the contest baseline solution for the active definition so you can tell whether the current branch is beating the official FlashInfer reference, not just last round's candidate.
- **When the workload count is >= 40**, do **not** start with the multi-GPU full sweep. First:
  - choose 2-3 representative workloads/regimes from the dominant axes plus the latency-heavy or boundary regimes
  - run and archive paired baseline measurements on those representative workloads
  - use only that small set for the first exploration / NCU / A/B loop
- If at least one representative regime shows a clear, repeatable latency win above the recent noise floor, keep the route alive and archive the candidate even if it is not yet proven globally. Assume a later cheap `if` / threshold dispatch may route that regime to a specialized path.
- Before launching the first full-workload sweep for that route, ask the user to confirm the spend. Full multi-GPU sweeps at this scale are expensive in elapsed time and Modal/GPU budget.
- After the user confirms, use the multi-GPU parallel script. Each workload runs in its own Modal container, up to `--workers` concurrently:
```bash
conda run -n fi-bench python scripts/run_modal_multiple_gpus.py --workers 10 \
  --out-dir optimize_ops/<operator>
```
- The script auto-discovers the definition from `config.toml`, packs the solution, and writes `benchmark_detailed_results.json` + `retained_run.log` to `--out-dir`. Use `--definition` to override, `--solution-dir` for per-problem solution dirs, and `--retry` to skip already-PASSED workloads.
- **Mandatory retry until all pass**: After every full sweep, check `benchmark_detailed_results.json` for any non-PASSED workloads (PARSE_ERROR, TIMEOUT, etc.). If any exist, immediately re-run with `--retry` to fill in only the failed cases:
```bash
conda run -n fi-bench python scripts/run_modal_multiple_gpus.py --workers 4 \
  --retry --out-dir optimize_ops/<operator>
```
  Repeat until the JSON and log both show **N/N PASSED, 0 FAILED**. Do NOT report a run as complete or use it for promotion decisions until every workload has a PASSED result. Transient Modal failures (PARSE_ERROR, TIMEOUT) are common and must be retried — they are not kernel bugs.
- **When the workload count is < 40**, the single-container serial script is acceptable:
```bash
conda run -n fi-bench python scripts/pack_solution.py
conda run -n fi-bench modal run scripts/run_modal.py
```
- Archive immediately:
```bash
python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/archive_modal_artifacts.py \
  --project-root $PWD \
  --candidate-id v25baseline \
  --kernel-path solution/triton/kernel.py
```
- For every representative workload used in local gating, run the same workload once on the current baseline in the same round and archive it as the paired baseline for that regime.

2. **Per-shape NCU bottleneck analysis** (run before and after each optimization round).
- Profile each representative band independently to identify whether the dominant bottleneck is shape-dependent:
```bash
# Short-band workload
conda run -n fi-bench modal run scripts/run_modal_single.py   --workload-uuid <uuid_short> --profile-ncu --ncu-set detailed --ncu-timeout 600

# Medium-band workload
conda run -n fi-bench modal run scripts/run_modal_single.py   --workload-uuid <uuid_medium> --profile-ncu --ncu-set detailed --ncu-timeout 600

# Long-band workload
conda run -n fi-bench modal run scripts/run_modal_single.py   --workload-uuid <uuid_long> --profile-ncu --ncu-set detailed --ncu-timeout 600

# Optional full sweep with NCU on selected workloads
conda run -n fi-bench modal run scripts/run_modal.py   --profile-ncu --ncu-max-workloads 3 --ncu-set detailed --ncu-timeout 600

# Filter to specific kernel for focused analysis
conda run -n fi-bench modal run scripts/run_modal_single.py   --workload-uuid <uuid_short> --profile-ncu --ncu-kernel-name fused_moe_gemm1_kernel
```
- NCU output is saved to:
  - `ncu_profile_report.md` / `ncu_profile_report_single.md` — full formatted markdown report
  - `benchmark_detailed_results.json` / `benchmark_detailed_result_single.json` — NCU data embedded in `ncu_profile` key
- Immediately extract compact metrics from the scratch JSON instead of hand-parsing the Markdown report:
```bash
python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/extract_ncu_metrics.py \
  benchmark_detailed_result_single.json \
  --top 5

python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/extract_ncu_metrics.py \
  benchmark_detailed_result_single.json \
  --kernel-substring fused_gemm2_ksplit_kernel
```
- For every band, record a compact diagnosis table in the round notes with:
  - `band`
  - `workload_uuid`
  - `dominant_kernel`
  - `duration`
  - `compute_throughput`
  - `memory_throughput`
  - `registers_per_thread`
  - `dynamic_shared_memory_per_block`
  - `theoretical_occupancy`
  - `achieved_occupancy`
  - `waves_per_sm`
  - `headroom_hypothesis`
- On CUDA 13.2 / Nsight Compute 2025.3, also inspect:
  - `launch__persisting_l2_cache_size` when testing L2-persistence hypotheses
  - instruction mix and scoreboard dependency tables when the kernel is compute- or latency-bound
  - cluster-aware occupancy output for cluster kernels, not just standard CTA launches
  - DSMEM / cluster-memory tables when evaluating thread-block clusters
- **Important**: The library's `flashinfer_bench_run_ncu` uses NVTX filtering which is broken on Modal containers. The scripts use direct subprocess NCU invocation instead, which profiles all kernel launches (warmup + profiling pass).

3. **Shape-dependence decision gate**.
- If all bands show the same dominant kernel and the same bottleneck class, do **not** split yet. Optimize the kernel body globally first.
- If the same kernel dominates but the limiter differs by band, a shape-specific internal path or dispatch threshold may be justified.
- If the hottest kernel itself differs by band, shape specialization is a strong candidate.
- If one workload or tiny regime contributes roughly one third or more of total latency, allow an explicit outlier-only candidate even if the rest of the band stays on the default path.
- Reject single-regime wins as promotion evidence when the full benchmark does not improve. Keep them as local evidence instead.

4. **Shape-aware NCU-guided optimization loop**.
- Read NCU sections to determine the bottleneck category:
  - **Compute-bound**: SM throughput high, memory throughput low → optimize ALU, reduce instruction count, improve ILP
  - **Memory-bound**: DRAM/L2 throughput high, compute throughput low → optimize data reuse, coalescing, cache hit rates
  - **Latency-bound**: Low throughput on both → improve occupancy, reduce warp stalls, increase parallelism
  - **Launch-bound**: CPU overhead dominates (visible in torch.profiler) → reduce kernel launches, fuse operations, minimize Python overhead
- Key NCU metrics to check per kernel:
  - `Duration` — absolute time in microseconds
  - `SM Active Cycles` / `Elapsed Cycles` — compute utilization
  - `Memory Throughput %` / `Compute (SM) Throughput %` — which is the bottleneck
  - Compare the same metric across real contest bands before adding any new bucket; band-to-band differences are the evidence for specialization.
  - `Achieved Occupancy` vs `Theoretical Occupancy` — room for occupancy improvement
  - `Block Limit Registers` / `Block Limit Shared Mem` — what limits occupancy
  - `Grid Size` vs `# SMs` (148 on B200) — grid saturation
  - `Waves Per SM` — how well the grid fills the GPU
- Treat the compact extractor fields as the default comparison table. Only fall back to the raw markdown report when the extractor output is insufficient.
- Calculate theoretical maximum performance:
  - For compute-bound: `theoretical_time = flops / (peak_flops_per_sm * num_sms * occupancy)`
  - For memory-bound: `theoretical_time = bytes_accessed / peak_bandwidth`
  - Gap between actual and theoretical = optimization headroom
- Use NCU recommendations (OPT lines) as direct optimization hints.
- Map bottlenecks to CUDA 13.2 tactic families:
  - **Tail underfill / irregular work distribution** → try persistent work queues or Cluster Launch Control.
  - **Cross-CTA reuse or atomics pressure** → try thread-block clusters + DSMEM before adding more global-memory traffic.
  - **Read-mostly metadata reused across many CTAs** → try L2 persistence / access-policy window and verify with same-round repeats.
  - **Launch-bound multi-kernel pipeline** → try fusion first, then Programmatic Dependent Launch if fusion is not viable.
  - **Barrier / fence interference from mixed traffic** → evaluate memory synchronization domains.

5. Candidate naming and hypothesis discipline.
- Name every candidate from the hypothesis, not from the result.
- Recommended pattern: `<version>_<band>_<change>`.
- Good examples:
  - `v25a_long_gemm2_ns2`
  - `v25b_long_gemm2_ns3`
  - `v25c_t14107_gemm2_ns2`
- Keep one shape-aware hypothesis per round as the default rule.

6. Two-stage optimization (informed by NCU and the shape matrix).
- Stage 1 (explore): 3-8 architecture-diverse candidates targeting the NCU-identified bottleneck, covering at least 3 technique families.
- For shape-dependent operators, it is acceptable to create separate band-specific candidates (`short`, `medium`, `long`, or the task-specific equivalent) plus one stable unified fallback.
- For CUDA 13.2-specific rounds, ensure at least one candidate attacks the bottleneck with a Blackwell control-surface change, not just another tile-size sweep.
- Stage 2 (exploit): refine top 1-3 routes with launch/tiling/register/occupancy tuning and band-specific resource control. Spend the most effort on bands with the highest latency share in the real contest dataset, not simply the largest number of workloads.
- For every promoted shape-specialized route, first push each band path to its local best in isolation. Only after that, integrate the specialized paths into one submission kernel or one entrypoint with cheap dispatch / internal branches.
- Do not integrate early. One shape-aware hypothesis per round is the default rule.
- After each candidate, re-run NCU on each affected band to verify the bottleneck shifted or reduced.
- Always re-run top candidates at least once for anti-noise validation.
- After each candidate, run the paired local gate against the same-round baseline on the relevant representative workload before deciding whether to keep exploring that route.
- When a candidate uses clusters, CLC, large dynamic shared memory, or PDL, record the exact launch attributes and resource settings in the round summary so the path is reproducible.
- **For each candidate**: save the kernel backup and benchmark JSON to `optimize_ops/` per the [Artifact Archival](#artifact-archival) checklist BEFORE modifying kernel.py for the next candidate.
- **At the end of each round**: write a round summary to `optimize_ops/docs/`.

7. Candidate evaluation mode.
- Representative gate before full benchmark:
  - for normal-size definitions, use at least 3 regimes; use 4 when the real dataset contains a transition outlier or a second dominant axis
  - make sure the gate covers both latency-heavy regimes and boundary regimes
  - for definitions with more than 40 workloads, begin with only 2-3 representative regimes/workloads rather than a full sweep, then expand back to the full matrix before any promotion decision
  - for every candidate/regime measurement, compare against the same-round paired baseline on the same workload
  - if one regime improves but another regime collapses, reject or keep it as local evidence only
  - if the candidate changes only one regime, treat first-sweep movement on untouched regimes as environmental noise until paired repeats confirm it
  - if one regime shows a clear, repeatable win above the recent noise floor while the others stay non-disastrous, archive the route and keep it as a dispatch candidate even before global proof
  - ask the user before upgrading from the representative gate to a full multi-GPU sweep on a >40-workload definition
  - if the representative gate stays healthy, the full benchmark decides promotion
- Final promotion gate:
  - run the active definition with `flashinfer-bench run --use-isolated-runner` in the official CUDA 13.2 container shape
  - match official flags for that definition before calling the candidate submission-ready
  - if Modal and official-parity results disagree, trust the official-parity run and re-diagnose
- Small round (<=3 candidates): serial evaluation is acceptable.
- Large round (>3 candidates): use parallel evaluation with isolated workspaces by default for representative-gate work. Spend a full-workload parallel round only after the route earns that budget and the user confirms the spend on >40-workload definitions:
```bash
python3 /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/run_modal_candidates_parallel.py \
  --project-root /path/to/mlsys26-flashinfer-solution \
  --candidates /path/to/candidates.json \
  --max-workers 3 \
  --profile-max-workloads 0 \
  --profile-ncu \
  --ncu-max-workloads 1 \
  --ncu-set basic \
  --out-dir /path/to/optimize_ops/<operator>/parallel_runs
```
- Treat the current workspace's `scripts/run_modal.py` as the canonical profiling interface. When new profiling flags land there, sync the parallel helper to that script instead of keeping stale hardcoded assumptions.
- Preserve per-candidate profiler artifacts when running in parallel. Expect:
  - `<candidate>.benchmark.json` with the embedded `torch_profile` and/or `ncu_profile` payload from `scripts/run_modal.py`
  - `<candidate>.torch_profile.json` with the extracted profiler payload
  - `<candidate>.torch_profile.txt` with the human-readable profiler table
  - `<candidate>.ncu_profile.json` with the extracted NCU payload
  - `<candidate>.ncu_profile.md` with the per-candidate NCU markdown report
  - `parallel_torch_profiles.json` as the combined index across all parallel runs
  - `parallel_ncu_profiles.json` as the combined NCU index across all parallel runs
- Use `--profile-workload-ids`, `--profile-sort-by`, `--profile-row-limit`, `--profile-warmup-runs`, `--profile-ncu`, `--ncu-workload-ids`, `--ncu-max-workloads`, `--ncu-set`, `--ncu-page`, `--ncu-timeout`, `--ncu-kernel-name`, or `--no-profile-torch` when the sweep needs to narrow or disable profiler capture. Prefer passing the representative regime IDs when validating a shape-aware route.
- Never run multiple candidates in the same mutable workspace.

8. Record results and maintain best-of-history.
- After benchmarking, **always run the archival helper or the archival checklist** from [Artifact Archival](#artifact-archival).
- Record whether each result came from Modal exploration, official-parity local evaluation, or both.
- Example:
```bash
python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/archive_modal_artifacts.py \
  --project-root $PWD \
  --candidate-id v25c_t14107_gemm2_ns2 \
  --kernel-path solution/triton/kernel.py \
  --workload-label long_14107
```
- In the round summary, always record:
  - baseline mean latency
  - best retained single run latency
  - recent noise floor
  - whether the candidate was archived only, promoted globally, or rejected and restored
- Optionally use `scripts/record_best_result.py` in this skill folder for structured recording:
```bash
python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/record_best_result.py   --benchmark-json benchmark_detailed_results.json   --kernel-path solution/cuda/kernel.cu   --operator gdn_decode_qk4_v8_d128_k_last   --model gpt5.3-codex   --out-dir optimize_ops/gdn_decode_qk4_v8_d128_k_last   --variant extreme_v23_rpw4
```

9. Keep packaging submission-ready.
- Keep `config.toml` aligned with the currently promoted candidate only.
- If a candidate is archived but not promoted, restore the checked-in main kernel to the previous best before ending the round.
- Keep the best snapshot in `best_solution_cuda/` or `best_solution_triton/`.


## Artifact Archival

**Every optimization round MUST archive its artifacts into `optimize_ops/`.** The project root `benchmark_detailed_results.json`, `benchmark_detailed_result_single.json`, `ncu_profile_report.md`, and `ncu_profile_report_single.md` are scratch files that will be overwritten.

### Directory Layout

```text
optimize_ops/
├── benchmarks/          # All benchmark result JSONs
├── docs/                # Per-round summaries, shape matrices, integration notes
├── kernels/             # Isolated candidate kernels for parallel eval
├── v<major>/            # Archived candidate kernel snapshots
└── <operator>/parallel_runs/  # Optional parallel-eval artifacts
```


### Preferred Path: Use the Archival Helper

Use the helper after every full benchmark or single-workload run:

```bash
python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/archive_modal_artifacts.py \
  --project-root $PWD \
  --candidate-id <candidate_id> \
  --kernel-path solution/triton/kernel.py
```

For single-workload or NCU runs, pass a workload label:

```bash
python /Users/yue/.codex/skills/flashinfer-b200-contest-optimizer/scripts/archive_modal_artifacts.py \
  --project-root $PWD \
  --candidate-id <candidate_id> \
  --kernel-path solution/triton/kernel.py \
  --workload-label long_14107
```

The helper copies whichever scratch artifacts exist, snapshots the current kernel, and avoids collisions by appending run suffixes when needed.

### Fallback: Manual Copy

If the helper is unavailable, copy the scratch files immediately with descriptive names:

```bash
cp benchmark_detailed_results.json \
  optimize_ops/benchmarks/benchmark_<candidate_id>_run<N>.json

cp benchmark_detailed_result_single.json \
  optimize_ops/benchmarks/benchmark_single_<candidate_id>_<workload>.json

cp ncu_profile_report_single.md \
  optimize_ops/benchmarks/ncu_<candidate_id>_<workload>.md

cp solution/triton/kernel.py \
  optimize_ops/v<major>/kernel_<candidate_id>.py
```

**Always save the backup BEFORE modifying `kernel.py` for the next experiment.** This ensures every tested variant is recoverable.

### After Each Optimization Round

Write a summary log capturing what was tried and the results:

```bash
# Create a per-round summary in docs/
cat > optimize_ops/docs/round_<date>_<topic>.md << 'EOF'
# Optimization Round: <topic>
- Date: <YYYY-MM-DD>
- Baseline mean: <version> at <avg_latency_ms>ms over <N> runs
- Best retained single run: <latency_ms>ms (context only)
- Recent noise floor: <abs or pct>
- Shape bands: <short/medium/long or task-specific bands>
- Triplet UUIDs: <uuid_short>, <uuid_medium>, <uuid_long>
- Candidates tested: <N>

## Shape Matrix
| Band | Workload UUID | Dominant kernel | Bottleneck | Limiter | Candidate/path |
|------|---------------|-----------------|------------|---------|----------------|
| <band> | <uuid> | <kernel> | <class> | <resource> | <path> |

## Results
| Candidate | avg_latency_ms | passed | delta vs baseline mean | decision |
|-----------|---------------|--------|------------------------|----------|
| <name>    | <value>       | <N>/19 | +/-<pct>%              | archive / promote / restore |

## Key Findings
- <finding 1>
- <finding 2>

## Integration Decision
<single global path, or how band-specialized paths were integrated into one kernel>

## Decision
<which candidate was archived only, which candidate was promoted, or why the repo was restored to the previous best>
EOF
```


### Archival Checklist (per candidate)

Before moving to the next candidate, verify these are saved:
1. Candidate kernel file → `optimize_ops/v<major>/kernel_<id>.py`
2. Benchmark JSON → `optimize_ops/benchmarks/benchmark_<id>_run<N>.json`
3. Paired baseline and candidate single-workload artifacts for any local gate
4. Per-shape NCU report(s) (if profiled) → `optimize_ops/benchmarks/ncu_<id>_<band>.md`
5. Shape matrix, integration note, and restore or promotion decision → `optimize_ops/docs/round_<date>_<topic>.md`

**Do NOT leave benchmark results only in the project root.** The root benchmark and NCU files are scratch files that get overwritten.


## NCU Profiling Reference

### B200 Hardware Specs (for theoretical peak calculations)
- **SMs**: 148
- **Compute Capability**: 10.0 (Blackwell)
- **DRAM Bandwidth**: ~8 TB/s (HBM3e)
- **L2 Cache**: 128 MB
- **Max warps per SM**: 64
- **Max threads per SM**: 2048
- **Max registers per SM**: 65536
- **Max shared memory per SM**: 228 KB (configurable)

### NCU Set Options
- `basic` — fast (~3 min for 30 kernels): Duration, Grid/Block size, Registers, Occupancy, Throughput summary
- `detailed` — thorough (~10+ min): All basic metrics + memory workload distribution, warp stall reasons, instruction mix
- `full` — exhaustive (very slow): All detailed + source-level metrics

### Interpreting NCU for Optimization Decisions

| NCU Finding | Bottleneck | Optimization Action |
|---|---|---|
| Low `Waves Per SM` (<1.0) | Grid too small | Increase grid size, split work into more blocks |
| `Achieved Occupancy` << `Theoretical Occupancy` | Warp scheduling | Reduce registers (smaller tiles), reduce shared memory |
| `Block Limit Registers` is the limiter | Register pressure | Use smaller BLOCK_M/N/K, add `num_warps` tuning |
| `Memory Throughput` >> `Compute Throughput` | Memory-bound | Improve data reuse, tile for L2, use shared memory |
| `Compute Throughput` >> `Memory Throughput` | Compute-bound | Reduce redundant computation, improve ILP |
| Both throughputs low | Latency-bound | Increase occupancy, reduce stalls, overlap compute/memory |
| High CPU time gap (torch.profiler) | Launch-bound | Fuse kernels, reduce allocations, minimize Python overhead |

### CUDA 13.2 / Blackwell Readouts Worth Using

- `launch__persisting_l2_cache_size`: verify that an L2-persistence experiment actually changed launch policy rather than only changing kernel code.
- Instruction mix + scoreboard dependencies: use when SASS is dense but throughput is still low; this separates dependency chains from true bandwidth limits.
- Cluster-kernel occupancy: for cluster launches, use cluster-aware occupancy numbers instead of plain block occupancy before blaming register pressure.
- DSMEM tables: when testing thread-block clusters, confirm that cross-block shared-memory traffic replaced global-memory traffic instead of adding synchronization without a bandwidth win.

### NCU CLI Quick Reference
```bash
# Quick single-workload profile (basic set, ~3 min)
modal run scripts/run_modal_single.py --workload-uuid 1 --profile-ncu --ncu-set basic

# Detailed single-workload profile (~10 min)
modal run scripts/run_modal_single.py --workload-uuid 1 --profile-ncu --ncu-set detailed --ncu-timeout 600

# Filter to one kernel
modal run scripts/run_modal_single.py --workload-uuid 1 --profile-ncu --ncu-kernel-name fused_moe_gemm1_kernel

# Full sweep with NCU on top 3 workloads
modal run scripts/run_modal.py --profile-ncu --ncu-max-workloads 3 --ncu-set basic
```

## Anti-Noise Benchmark Policy

For microsecond kernels on B200, treat single-run gains as provisional.

**Why latency is the primary metric**: The reference torch baseline is unstable across Modal runs (different container placements, thermal states, background load). `avg_speedup` = `reference_latency / solution_latency`, so speedup inherits the reference noise. Two runs of the same kernel can show 43x and 46x speedup purely from reference variance, while `avg_latency_ms` stays within 1-2%. Always use absolute latency for A/B comparisons.

Required policy:
1. For each finalist candidate, run at least 2 repeats. When the workload count is >= 40, do not schedule those full-workload repeats until a 2-3 regime gate has already shown a clear local win and the user has confirmed the spend; after confirmation, use `scripts/run_modal_multiple_gpus.py --workers 10`.
2. **Every run must reach 100% pass rate before it counts.** After each sweep, check for non-PASSED workloads and immediately `--retry` until the JSON shows N/N PASSED. A run with any failures is incomplete — do not use it for latency averages or promotion decisions.
3. **Primary ranking** (use these to decide which solution is better):
  - lower `avg_latency_ms_mean_of_repeats` — the single most reliable metric
  - lower `p95_latency_ms_mean_of_repeats` — catches tail-latency regressions
  - lower run-to-run latency variance — stability matters
4. **Secondary ranking** (report but do not use for decisions):
  - higher `avg_speedup` — useful for leaderboard context only, unreliable for A/B comparisons due to reference instability
5. Reject candidates with correctness or runtime instability even if they have one exceptional run.
6. Compare global promotion against the best retained **mean latency**, not the best retained single run.
7. When reporting results, always show both `avg_latency_ms` and `avg_speedup` side by side, but explicitly note that latency is the decision metric.
8. For finalists, include at least one official-parity run in `flashinfer/flashinfer-ci-cu132:latest` with `--use-isolated-runner`. Locked clocks reduce variance versus Modal, so treat this run as the stronger evidence for promotion.

## Promotion Checklist

Treat these as hard rules:
1. A local regime win is enough to archive a candidate and continue exploring. If only one representative shape wins clearly, keep it as a future dispatch candidate instead of discarding it for lacking immediate global proof.
2. Global promotion requires a healthy representative gate and a better full-benchmark **mean latency** than the current best mean.
3. For definitions with more than 40 workloads, do not start the first full-workload repeat set until a 2-3 regime gate has shown a clear win and the user has confirmed the spend.
4. Run at least 2 full repeats unless the gain is clearly larger than the recent run noise floor.
5. **Every repeat must have 100% pass rate (N/N PASSED, 0 FAILED) before it counts.** After each sweep, check for failures and `--retry` until all workloads pass. A run with any non-PASSED workloads is incomplete and must not be used for promotion decisions or retained as submission evidence.
6. Before marking a candidate submission-ready, run the official definition command in the official container shape with `--use-isolated-runner`.
7. If a candidate is not promoted, restore the repo's checked-in main kernel to the previous best before ending the round.
8. If a candidate touched only one regime, confirm any claimed untouched-path regression with paired repeats before treating it as real.

## Naming Policy (Latency First)

Primary artifact naming should be latency-first:
- `<operator>_<model>_lat<xxpxxx>us.kernel.cu`

Use hypothesis-driven candidate ids during exploration:
- `<version>_<band>_<change>`
- `v25a_long_gemm2_ns2`
- `v25b_long_gemm2_ns3`
- `v25c_t14107_gemm2_ns2`

Use speedup as secondary metadata (summary/registry) only. Speedup depends on the reference torch baseline which is unstable across runs — the same kernel can show 43x one run and 47x the next purely from reference noise. Latency is reproducible within ~1-2%.

## B200-Specific Optimization Priorities

1. Increase parallelism for low-batch decode.
- If `B` is small, avoid few-block kernels.
- Explore row/head tiling and rows-per-warp/block strategies.

2. Tune occupancy with resource control.
- Sweep `rows_per_warp`, threads/block, and `__launch_bounds__` min blocks/SM.
- Prefer stable median/p95 improvements over single-point bests.
- For cluster kernels, compute occupancy with cluster-aware APIs and test whether cluster size `16` hurts active clusters more than it helps locality.

3. Use shape-regime dispatch, not one tile for all.
- Split decode-like vs prefill-like or low/high token-per-expert regimes only when per-shape NCU shows different bottlenecks or different limiting resources.
- When the workload count is large, a clear win on one representative regime is enough to keep a specialized path alive if the dispatch predicate is cheap and the fallback path stays healthy.
- Keep routing cheaper than the kernel body; if dispatch complexity starts growing faster than the gain, stop and re-diagnose.

4. Benchmark non-power-of-two CTA sizes when a band is underoccupied.
- Do not assume only 128/256/512 are worth testing.
- Re-test larger shapes after a local win to make sure the full matrix stays healthy.

5. Use memory hierarchy intentionally.
- Prefer coalesced global access and vectorized load/store (`float4`, packed bf16 loads).
- Use shared memory only when reuse exceeds synchronization overhead.
- Evaluate cached vs streaming state stores (`st.global` vs `__stcs`) under real workloads.
- Test L2 persistence only for windows with real inter-CTA reuse, and reset or disable it when the hot window changes.
- Use DSMEM only when it replaces a more expensive global-memory exchange. If it only adds `cluster.sync()` overhead, back it out.

6. Treat persistent/shared-memory caching as a hypothesis, not a conclusion.
- A persistent or shared-memory-heavy path must beat a simpler fallback on the representative triplet and the full benchmark, not just on one band.
- If occupancy collapses or synchronization cost dominates, keep or restore a simpler fallback path.
- When a persistent-kernel path is close but tails poorly, try Cluster Launch Control before rewriting the algorithm again.

7. Keep numerics stable while using fast math selectively.
- Use `__expf`, `__logf` only where tolerated by correctness checks.
- Validate across all workloads before promoting a candidate.

8. Use CUDA 13.2 launch-control features only when they reduce a measured bottleneck.
- Prefer fusion over PDL if fusion is possible.
- Prefer simpler non-cluster launches over clusters when locality gains do not offset lower active-cluster count.
- Prefer a plain global-memory path when the shared-memory / DSMEM route inflates registers or synchronization too much.


## Candidate Scoring Rule

**Primary (decision-making)**:
1. Healthy representative gate — no unacceptable regression in any required regime. Single-regime wins are local evidence, not promotion evidence.
2. Lower mean `avg_latency_ms` across repeats — most stable metric, unaffected by reference noise.
3. Lower mean `p95_latency_ms` across repeats — catches tail-latency regressions.
4. Lower latency variance across repeats — prefer stable candidates.

**Secondary (reporting only, do NOT use for A/B decisions)**:
5. Higher mean `avg_speedup` — report for leaderboard context. Reference baseline is unstable so speedup differences of <5% between candidates are noise, not signal.


## Required References

Load these files when running this skill:
- `references/b200-architecture-notes.md`
- `references/cuda-13-2-notes.md`
- `references/contest-evaluation-environment.md`
- `references/cuda-book-checklist.md`
- `references/parallel-eval-and-naming.md`
- `references/reference-repo-recon.md`

Use release-note and programming-guide links in `references/cuda-13-2-notes.md` as the canonical CUDA 13.2 source.
