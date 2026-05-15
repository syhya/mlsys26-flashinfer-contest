#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for invoking Codex as an external executor during a math_agent iteration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CodexRunnerConfig:
    """Configuration for invoking Codex as an external executor."""
    case_dir: str = "."
    script_name: str = "codex_agent_seed_gen.py"
    codex_bin: str = "codex"
    codex_model: Optional[str] = None
    mode: str = "aggressive"
    validation_level: str = "lightweight"
    reasoning_effort: str = "high"
    required_symbol: str = "ModelNew"
    max_attempts: int = 1
    min_speedup: float = 1.10


@dataclass
class CodexRunnerResult:
    """Container for the result of a single Codex iteration run."""
    solution_path: str
    evaluation_path: str
    report_path: str
    log_path: str
    evaluation_result: dict[str, Any]


def _resolve_case_dir(case_dir: str) -> Path:
    candidate = Path(case_dir)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return candidate.resolve()


def _build_augmented_prompt(
    original_prompt: str,
    plan_text: str,
    trigger_reason: str,
    parent_score: float,
    current_eval: Optional[dict[str, Any]],
) -> str:
    sections = [original_prompt.rstrip()]
    sections.append("\n# Iteration Context\n")
    sections.append(
        "You are being called as an external Codex executor inside an ongoing "
        "LoongFlow PES iteration.\n"
    )
    sections.append(
        "Improve the provided `initial_program.py` instead of restarting from scratch.\n"
    )
    sections.append(f"Trigger reason: {trigger_reason}\n")
    sections.append(f"Parent score: {parent_score:.6f}\n")
    if plan_text.strip():
        sections.append("\n## Current planner plan\n")
        sections.append(plan_text.rstrip())
        sections.append("\n")
    if current_eval:
        sections.append("\n## Current executor evaluation\n")
        sections.append(json.dumps(current_eval, ensure_ascii=False, indent=2))
        sections.append("\n")
    sections.append(
        "\n## Additional instructions\n"
        "- Focus on correcting the current candidate's evaluation failures or weak speedup.\n"
        "- Preserve the required public API and helper entrypoints.\n"
        "- If the current candidate already has the right structure, refine it instead of rewriting everything.\n"
        "- Use the evaluator feedback and profiling signals to guide the optimization.\n"
    )
    return "".join(sections)


def run_codex_iteration(
    output_dir: Path,
    input_code: str,
    plan_text: str,
    trigger_reason: str,
    parent_score: float,
    current_eval: Optional[dict[str, Any]],
    config: CodexRunnerConfig,
) -> CodexRunnerResult:
    """Run a single Codex iteration, returning the solution and evaluation result."""
    case_dir = _resolve_case_dir(config.case_dir)
    script_path = case_dir / config.script_name
    prompt_path = case_dir / "task_prompt.txt"
    eval_path = case_dir / "eval_program_with_profile.py"

    if not script_path.is_file():
        raise FileNotFoundError(f"Codex script not found: {script_path}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Task prompt not found: {prompt_path}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    initial_program_path = output_dir / "initial_program.py"
    generated_prompt_path = output_dir / "task_prompt.txt"
    local_eval_path = output_dir / "eval_program_with_profile.py"
    solution_path = output_dir / "codex_seed.py"
    evaluation_path = output_dir / "codex_seed_eval.json"
    report_path = output_dir / "codex_seed_report.json"
    log_path = output_dir / "codex_seed_run.log"

    initial_program_path.write_text(input_code, encoding="utf-8")
    generated_prompt_path.write_text(
        _build_augmented_prompt(
            original_prompt=prompt_path.read_text(encoding="utf-8"),
            plan_text=plan_text,
            trigger_reason=trigger_reason,
            parent_score=parent_score,
            current_eval=current_eval,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(eval_path, local_eval_path)

    command = [
        sys.executable,
        str(script_path),
        "--work-dir",
        str(output_dir),
        "--input",
        initial_program_path.name,
        "--prompt",
        generated_prompt_path.name,
        "--eval-file",
        local_eval_path.name,
        "--output",
        solution_path.name,
        "--eval-result-file",
        evaluation_path.name,
        "--report-file",
        report_path.name,
        "--log-file",
        log_path.name,
        "--required-symbol",
        config.required_symbol,
        "--mode",
        config.mode,
        "--validation-level",
        config.validation_level,
        "--reasoning-effort",
        config.reasoning_effort,
        "--codex-bin",
        config.codex_bin,
        "--max-attempts",
        str(config.max_attempts),
        "--min-speedup",
        str(config.min_speedup),
    ]
    if config.codex_model:
        command.extend(["--model", config.codex_model])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(output_dir),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Codex iteration fallback failed: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )

    if not solution_path.is_file():
        raise FileNotFoundError(f"Codex did not create solution file: {solution_path}")
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f"Codex did not create evaluation file: {evaluation_path}"
        )

    evaluation_result = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return CodexRunnerResult(
        solution_path=str(solution_path),
        evaluation_path=str(evaluation_path),
        report_path=str(report_path),
        log_path=str(log_path),
        evaluation_result=evaluation_result,
    )

