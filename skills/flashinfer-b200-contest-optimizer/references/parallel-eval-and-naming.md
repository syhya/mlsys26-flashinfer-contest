# Parallel Evaluation and Naming Playbook

## Why parallel evaluation

For B200 micro-kernel tuning, signal is often close to noise.
Running many candidates sequentially can overfit to transient cluster or thermal variance.
Parallel isolated runs reduce wall-clock and help identify robust winners faster.

## Isolation rule

Never evaluate multiple candidates by mutating one shared workspace.
Use per-candidate temporary workspaces copied from project root so each run has deterministic source and config state.

## Paired baseline local gate

Before trusting a local regime win:
- run the current baseline on the same representative workload in the same round
- archive the baseline and candidate pair together
- use that paired delta to decide whether the local hypothesis is real enough to keep exploring

Do not compare a local candidate only against an older retained sample from a different run.

## Baseline truth

For global promotion, compare against the best retained **mean latency**, not the best retained single run.
Treat a retained single run as a favorable sample for context only.
When round notes say the current best is `0.727 ms` mean but the retained run in the repo is `0.706 ms`, use `0.727 ms` as the promotion threshold.

## Repeat policy

- Exploratory sweep: 1 run per candidate.
- Local regime gate: 1 paired baseline + 1 candidate run on the same workload in the same round.
- Finalists: at least 2 full runs per candidate unless the mean gain is clearly larger than the recent noise floor.
- Promotion threshold: winner must beat runner-up on mean latency across repeats, not just best single run.

## Untouched-path noise

If a candidate only changes one regime, apparent regressions on untouched regimes in the first full sweep can be environmental noise.
Do not treat those movements as real until paired repeats confirm them.

## Naming policy

Use latency-first naming for best artifacts:
- `<operator>_<model>_lat<xxpxxx>us.kernel.cu`

Name candidate ids from the hypothesis, not from the result:
- `v25a_long_gemm2_ns2`
- `v25b_long_gemm2_ns3`
- `v25c_t14107_gemm2_ns2`

The id should tell you which regime changed and what changed there.
Keep speedup in summary JSON or Markdown as secondary fields.

## Suggested summary fields

At minimum record:
- `baseline_mean_latency_ms`
- `best_retained_single_latency_ms`
- `avg_latency_ms`
- `median_latency_ms`
- `p95_latency_ms`
- `avg_speedup_factor`
- repeat count and run URLs
- variance or stddev across repeats
- recent noise floor estimate
