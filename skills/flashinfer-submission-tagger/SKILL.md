---
name: flashinfer-submission-tagger
description: Create and push FlashInfer contest submission tags such as `submission-v3` in a git repository. Use when the user asks to "tag your submission", "create a submission tag", "push the latest submission tag", or prepare a FlashInfer contest repo so the evaluator will use the latest tagged commit. Always validate `config.toml` topology before committing or tagging, especially for multi-definition repos where extra or misplaced configs can make the evaluator pack an empty solution.
---

# Flashinfer Submission Tagger

## Overview

Create a new `submission-vN` tag for the current commit, push that tag to the remote repository, and verify the exact tag and commit that FlashInfer contest evaluation should consume.

Use the bundled script to compute the next tag deterministically instead of re-implementing version selection by hand.
Use the bundled topology checker to block invalid `config.toml` layouts before commit, tag, or push.

## Workflow

1. Check repository state before tagging.
- Read the current branch, commit SHA, existing `submission-*` tags, and remotes.
- Assume `origin` is the push remote unless the user explicitly asks for another remote or `origin` is missing.

2. Choose the tag name.
- If the user provides a concrete tag, use it exactly.
- Otherwise, select the next numeric version after the highest existing `submission-vN`.
- Never reuse or retarget an existing tag unless the user explicitly asks for that destructive action.

3. **Validate `config.toml` topology before touching metadata.** This is a hard gate.
- Run:
```bash
python3 /Users/yue/.codex/skills/flashinfer-submission-tagger/scripts/check_config_topology.py --repo /path/to/repo
```
- If the checker exits non-zero, STOP. Do not commit, tag, or push until the layout is fixed.
- The checker only evaluates the submission surface: the repo root, immediate definition subdirectories, and the `solution/` trees attached to them. It intentionally ignores archived artifacts under directories such as `optimize_ops/` or `reference/`.
- Accepted layouts:
  - Single-definition root layout: root `config.toml` + root `solution/<language>/...`
  - Definition-subdir layout: one or more `<definition>/config.toml` files, each paired with `<definition>/solution/<language>/...`
- Forbidden layouts:
  - root `config.toml` in a multi-definition repo
  - both root `config.toml` and definition-subdir configs in the same repo
  - any `config.toml` nested under `solution/`
  - a config whose `entry_point` file does not exist under the resolved source dir
  - a resolved source dir with zero real source files

4. **Discover the solution layout from the validated config topology.** Do NOT hardcode paths like `solution/triton/` or `solution/cuda/`.
- For a single-definition root layout, read root `config.toml` and derive `solution/<language>/`.
- For a definition-subdir layout, read the target `<definition>/config.toml` and derive `<definition>/solution/<language>/`.
- When the repo contains multiple definition subdirectories, operate only on the target definition the user asked to submit. Do not infer metadata from unrelated definitions.

5. **Update all submission metadata files before committing.** Every tagged commit must have consistent, up-to-date metadata. Paths are relative to the solution directory discovered in step 4.
- **`config.toml`** (root for a single-definition root layout, or `<definition>/config.toml` for a definition-subdir layout): update `name` field to match the new submission version.
- **`<solution-dir>/summary.json`**: update with the best benchmark run data — date, candidate name, previous_submission, per-run results (avg_latency_ms, median, p95, min, speedup, passed), mean across repeats, and improvement vs previous. Use `avg_latency_ms` as the primary metric.
- **`<solution-dir>/summary.md`**: update with a human-readable comparison table (latency-primary), description of changes, decision rationale, and selected retained run.
- **`<solution-dir>/retained_run.json`**: if a new benchmark run was performed, update with the retained run's full JSON data.
- **`README.md`**: update the Performance table with the best single-run result, update Implementation Notes if kernel behavior changed, update the solution name.
- **Kernel file** (e.g. `kernel.py` or `kernel.cu` in the solution directory): ensure the docstring/header comment reflects the new version number and lists key changes.
- Move any backup/exploration kernels and benchmark artifacts out of the solution directory into `optimize_ops/` so the submission directory stays clean.
- Do NOT modify `.gitignore`.

6. Commit all updated files.
- Stage only the specific files that changed (avoid `git add -A`).
- Write a descriptive commit message with the key metric (avg_latency_ms), changes summary, and `Co-Authored-By` if applicable.

7. Create the tag on `HEAD`.
- Tag the current commit, not the working tree.
- If the repo has uncommitted changes after step 5, call that out — something was missed.

8. Push the branch and tag.
- Push both `origin <branch>` and `origin <tag>`.
- After push, verify the tag exists and report the commit SHA it points to.

## Scripts

Use `scripts/tag_submission.py` when you need deterministic tag selection.
Use `scripts/check_config_topology.py` as a mandatory submission gate.

Common commands:

```bash
python3 scripts/tag_submission.py --repo /path/to/repo --print-next
python3 scripts/tag_submission.py --repo /path/to/repo --create
python3 scripts/tag_submission.py --repo /path/to/repo --create --push
python3 scripts/tag_submission.py --repo /path/to/repo --tag submission-v7 --create --push
```

Behavior:
- `--print-next`: print the next available `submission-vN` without changing git state.
- `--create`: create the tag on the current `HEAD`.
- `--push`: push the chosen tag to the configured remote. Use with `--create` for the common contest workflow.
- `--tag`: override auto-increment and use an explicit tag name.

Topology gate:

```bash
python3 /Users/yue/.codex/skills/flashinfer-submission-tagger/scripts/check_config_topology.py --repo /path/to/repo
```

Behavior:
- exit `0`: topology is valid for submission work
- exit non-zero: invalid layout; fix the reported `config.toml` issue before commit/tag/push
- prints the detected layout, active config(s), resolved source dir(s), and the first blocking error when invalid

## Output Expectations

When using this skill, report:
- the final tag name
- the commit SHA it points to
- whether the tag was pushed successfully

If anything blocks the workflow, report the exact git condition:
- no git repository
- tag already exists
- remote missing
- push failed

If the topology gate blocks the workflow, report the exact config issue:
- multi-definition repo has a root `config.toml`
- root and definition-subdir configs coexist
- `config.toml` is nested under `solution/`
- resolved source dir is missing or empty
- `entry_point` file is missing relative to the resolved source dir
