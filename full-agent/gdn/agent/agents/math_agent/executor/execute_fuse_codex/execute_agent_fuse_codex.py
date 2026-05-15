#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse executor with Codex fallback for CUDA task iterations.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents.math_agent.executor.codex_runner import (
    CodexRunnerConfig,
    run_codex_iteration,
)
from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
    EvolveExecuteAgentFuse,
    ExecuteAgentFuseConfig,
)
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import ContentElement, Message, MimeType, Role
from loongflow.framework.pes.context import Context, LLMConfig, Workspace
from loongflow.framework.pes.evaluator.evaluator import LoongFlowEvaluator
from loongflow.framework.pes.register import Worker

logger = get_logger(__name__)


@dataclass
class ExecuteAgentFuseCodexConfig:
    """Configuration for the fuse executor with Codex fallback."""
    react_system_prompt: Optional[str] = None
    chat_system_prompt: Optional[str] = None
    llm_config: Optional[LLMConfig] = None
    llm_switch: Optional[dict[str, Any]] = None
    max_rounds: int = 1
    react_max_steps: int = 2
    score_threshold: float = 0.9

    codex_enabled: bool = True
    codex_case_dir: str = "."
    codex_script_name: str = "codex_agent_seed_gen.py"
    codex_bin: str = "codex"
    codex_model: Optional[str] = None
    codex_mode: str = "aggressive"
    codex_validation_level: str = "lightweight"
    codex_reasoning_effort: str = "high"
    codex_required_symbol: str = "ModelNew"
    codex_max_attempts: int = 1
    codex_min_speedup: float = 1.10
    codex_min_gain: float = 0.02

    codex_output_root: str = "output_with_codex"


class EvolveExecuteAgentFuseCodex(Worker):
    """Run fuse executor first, then invoke Codex when the result is weak or failed."""

    def __init__(self, config: Any, evaluator: LoongFlowEvaluator):
        super().__init__()
        self.config = (
            config
            if isinstance(config, ExecuteAgentFuseCodexConfig)
            else ExecuteAgentFuseCodexConfig(**config)
        )
        self.evaluator = evaluator
        inner_config = ExecuteAgentFuseConfig(
            react_system_prompt=self.config.react_system_prompt,
            chat_system_prompt=self.config.chat_system_prompt,
            llm_config=self.config.llm_config,
            llm_switch=self.config.llm_switch,
            max_rounds=self.config.max_rounds,
            react_max_steps=self.config.react_max_steps,
            score_threshold=self.config.score_threshold,
        )
        self.inner = EvolveExecuteAgentFuse(inner_config, evaluator)
        logger.info("Executor: Agent Fuse Codex successfully initialized")

    async def run(self, context: Context, message: Message) -> Message:
        """Execute the fuse workflow, falling back to Codex when the result is weak."""
        parent_info = self._load_parent_info_from_message(message)
        fuse_result = await self.inner.run(context, message)
        result_data = self._message_data(fuse_result)

        trigger_reason = self._should_trigger_codex(parent_info, result_data)
        if not trigger_reason:
            return fuse_result

        logger.info(
            f"Trace ID: {context.trace_id}: Executor Fuse Codex: triggering Codex fallback because {trigger_reason}"
        )

        try:
            codex_result = await asyncio.to_thread(
                run_codex_iteration,
                output_dir=self._codex_output_dir(context),
                input_code=self._select_input_code(parent_info, result_data),
                plan_text=self._read_text(result_data.get("best_plan_file_path")),
                trigger_reason=trigger_reason,
                parent_score=float(parent_info.get("score", 0.0) or 0.0),
                current_eval=self._read_json(result_data.get("best_evaluation_file_path")),
                config=CodexRunnerConfig(
                    case_dir=self.config.codex_case_dir,
                    script_name=self.config.codex_script_name,
                    codex_bin=self.config.codex_bin,
                    codex_model=self.config.codex_model,
                    mode=self.config.codex_mode,
                    validation_level=self.config.codex_validation_level,
                    reasoning_effort=self.config.codex_reasoning_effort,
                    required_symbol=self.config.codex_required_symbol,
                    max_attempts=self.config.codex_max_attempts,
                    min_speedup=self.config.codex_min_speedup,
                ),
            )
        except Exception as exc:
            logger.exception(
                f"Trace ID: {context.trace_id}: Executor Fuse Codex: fallback failed: {exc}"
            )
            return fuse_result

        base_score = self._extract_score(
            self._read_json(result_data.get("best_evaluation_file_path"))
        )
        codex_score = self._extract_score(codex_result.evaluation_result)
        if codex_score <= base_score:
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse Codex: keeping base executor result "
                + f"(base_score={base_score:.6f}, codex_score={codex_score:.6f})"
            )
            return fuse_result

        logger.info(
            f"Trace ID: {context.trace_id}: Executor Fuse Codex: Codex result selected "
            + f"(base_score={base_score:.6f}, codex_score={codex_score:.6f})"
        )
        Workspace.write_executor_best_solution(context, codex_result.solution_path)
        Workspace.write_executor_best_eval(context, codex_result.evaluation_path)

        result_data["best_solution_file_path"] = Workspace.get_executor_best_solution_path(
            context
        )
        result_data["best_evaluation_file_path"] = (
            Workspace.get_executor_best_evaluation_path(context)
        )
        return Message.from_text(
            data=result_data,
            sender="executor",
            role=Role.USER,
            mime_type=MimeType.APPLICATION_JSON,
        )

    def _message_data(self, message: Message) -> dict[str, Any]:
        elements = message.get_elements(ContentElement)
        if not elements:
            raise ValueError("Executor result missing ContentElement data.")
        data = elements[0].data
        if not isinstance(data, dict):
            raise ValueError("Executor result payload must be a dict.")
        return dict(data)

    def _load_parent_info_from_message(self, message: Message) -> dict[str, Any]:
        elements = message.get_elements(ContentElement)
        if not elements:
            raise ValueError("Planner result missing ContentElement data.")
        data = elements[0].data
        parent_info_path = data.get("parent_info_file_path")
        if not parent_info_path or not os.path.exists(parent_info_path):
            raise FileNotFoundError(f"Missing parent_info.json: {parent_info_path}")
        with open(parent_info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_text(self, path: Optional[str]) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_json(self, path: Optional[str]) -> Optional[dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _extract_score(self, evaluation: Optional[dict[str, Any]]) -> float:
        if not evaluation:
            return 0.0
        return float(evaluation.get("score", 0.0) or 0.0)

    def _extract_speedup(self, evaluation: Optional[dict[str, Any]]) -> float:
        if not evaluation:
            return 0.0
        metrics = evaluation.get("metrics", {}) or {}
        return float(metrics.get("speedup", 0.0) or 0.0)

    def _should_trigger_codex(
        self,
        parent_info: dict[str, Any],
        result_data: dict[str, Any],
    ) -> Optional[str]:
        if not self.config.codex_enabled:
            return None

        base_solution_path = result_data.get("best_solution_file_path")
        base_eval = self._read_json(result_data.get("best_evaluation_file_path"))
        parent_score = float(parent_info.get("score", 0.0) or 0.0)

        if not base_solution_path or not os.path.exists(base_solution_path):
            logger.info(
                "Executor Fuse Codex: skip Codex fallback because base executor did not produce a candidate file."
            )
            return None
        if not base_eval:
            logger.info(
                "Executor Fuse Codex: skip Codex fallback because "
                "base executor did not produce a readable evaluation result."
            )
            return None

        base_score = self._extract_score(base_eval)
        if base_score <= parent_score + self.config.codex_min_gain:
            return (
                f"base executor gain is too small: parent_score={parent_score:.6f}, "
                f"child_score={base_score:.6f}"
            )

        speedup = self._extract_speedup(base_eval)
        if speedup < self.config.codex_min_speedup:
            return (
                f"base executor speedup is below threshold: "
                f"speedup={speedup:.4f}x < {self.config.codex_min_speedup:.4f}x"
            )

        return None

    def _codex_output_dir(self, context: Context) -> Path:
        case_dir = Path(self.config.codex_case_dir)
        if not case_dir.is_absolute():
            case_dir = (Path.cwd() / case_dir).resolve()
        trace_id = context.trace_id or "unknown_trace"
        return (
            case_dir
            / self.config.codex_output_root
            / f"iteration_{context.current_iteration}_{trace_id}"
        )

    def _select_input_code(
        self,
        parent_info: dict[str, Any],
        result_data: dict[str, Any],
    ) -> str:
        best_solution_path = result_data.get("best_solution_file_path")
        if best_solution_path and os.path.exists(best_solution_path):
            with open(best_solution_path, "r", encoding="utf-8") as f:
                return f.read()

        solution_text = parent_info.get("solution", "")
        if not isinstance(solution_text, str) or not solution_text.strip():
            raise RuntimeError("No valid parent solution code is available for Codex fallback.")
        return solution_text
