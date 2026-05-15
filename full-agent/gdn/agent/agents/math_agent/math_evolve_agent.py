#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Evolve Agent Runner - Refactored to use BasePESRunner.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from agents.math_agent.executor.execute_chat.execute_agent_chat import (
    EvolveExecuteAgentChat,
)
from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
    EvolveExecuteAgentFuse,
)
from agents.math_agent.executor.execute_fuse_codex.execute_agent_fuse_codex import (
    EvolveExecuteAgentFuseCodex,
)
from agents.math_agent.executor.execute_react.execute_agent_react import (
    EvolveExecuteAgentReact,
)
from agents.math_agent.planner.plan_agent import EvolvePlanAgent
from agents.math_agent.summary.summary_agent import EvolveSummaryAgent
from loongflow.framework.pes import Worker
from loongflow.framework.pes.base_runner import BasePESRunner


class MathPESAgent(BasePESRunner):
    """
    Math Evolve Agent runner for open-ended math and algorithm optimization tasks.

    Extends BasePESRunner with:
    - Support for initial code file (--initial-file)
    - Multiple executor variants (chat, react, fuse)
    """

    def _add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        """Add math-agent specific CLI arguments."""
        parser.add_argument(
            "--initial-file",
            type=str,
            default=None,
            help="Override the 'initial_code' by reading from a file.",
        )

    def _merge_custom_configs(
        self, args: argparse.Namespace, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle math-agent specific config merging."""
        if args.initial_file is not None:
            try:
                config["evolve"]["initial_code"] = Path(args.initial_file).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                print(
                    f"Error: Initial code file not found at '{args.initial_file}'",
                    file=sys.stderr,
                )
                sys.exit(1)
        return config

    def _get_process_name(self) -> str:
        return "Evolve"

    def _get_worker_registrations(
        self,
    ) -> Tuple[
        List[Tuple[str, Type[Worker]]],
        List[Tuple[str, Type[Worker]]],
        List[Tuple[str, Type[Worker]]],
    ]:
        """Register Math agent workers."""
        planners = [("evolve_planner", EvolvePlanAgent)]
        executors = [
            ("evolve_executor_chat", EvolveExecuteAgentChat),
            ("evolve_executor_react", EvolveExecuteAgentReact),
            ("evolve_executor_fuse", EvolveExecuteAgentFuse),
            ("evolve_executor_fuse_codex", EvolveExecuteAgentFuseCodex),
        ]
        summarizers = [("evolve_summary", EvolveSummaryAgent)]
        return planners, executors, summarizers


if __name__ == "__main__":
    runner = MathPESAgent()
    runner.start()
