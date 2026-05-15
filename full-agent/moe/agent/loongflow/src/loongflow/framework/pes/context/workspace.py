# -*- coding: utf-8 -*-
"""
This file defines the Workspace class, which provides unified path management
and file I/O helpers for each stage (planner, executor, summarizer) in the LoongFlow
evolution workflow.
Typical directory layout:
{base_path}/{task_id}/iteration{idx}/
    ├── planner/
    │   ├── parent_info.json
    │   ├── plan{idx}.txt
    │   ├── best_plan.txt
    ├── executor/
    │   ├── history.json
    │   ├── best_solution.py
    │   ├── best_evaluation.json
    │   ├──{round_idx}_{candidate_idx}
    │   │  ├── solution{random_idx}.py
    │   │  ├── evaluation{random_idx}.json
    ├── summarizer/
    │   ├── best_summary.json
"""

import os
from pathlib import Path
from typing import Optional

from loongflow.framework.pes.context import Context

PLANNER_PARENT_FILE = "parent_info.json"
PLANNER_BEST_PLAN_FILE = "best_plan.txt"

EXECUTOR_HISTORY_FILE = "history.json"
EXECUTOR_BEST_PLAN_FILE = "best_plan.txt"
EXECUTOR_BEST_SOLUTION_FILE = "best_solution.py"
EXECUTOR_BEST_EVALUATION_FILE = "best_evaluation.json"
SUMMARIZER_BEST_SUMMARY_FILE = "best_summary.txt"


class Workspace:
    """
    Workspace provides unified directory and file handling utilities
    for planner, executor, summarizer, and evaluator stages.
    """

    @staticmethod
    def _get_iteration_path(context: Context) -> Path:
        """
        Build the base iteration path, optionally including a model_name
        subdirectory for multi-model competitive mode.

        Returns:
            Single-model: {base_path}/{task_id}/{iteration}
            Multi-model:  {base_path}/{task_id}/{iteration}/{model_name}
        """
        base_path = Path(context.base_path)
        path = base_path / str(context.task_id) / str(context.current_iteration)
        if context.model_name:
            path = path / context.model_name
        return path

    # -----------------------------
    # Planner Utilities
    # -----------------------------
    @staticmethod
    def get_planner_path(context: Context, create: bool = True) -> Path:
        """
        get planner path
        """
        path = Workspace._get_iteration_path(context) / "planner"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def write_planner_parent_info(context: Context, parent_json: str) -> None:
        """
        Write planner parent info.
        :param context: task execution context
        :param parent_json: serialized parent JSON string
        """
        planner_path = Workspace.get_planner_path(context)
        parent_path = planner_path / PLANNER_PARENT_FILE
        with open(parent_path, "w") as f:
            f.write(parent_json)

    @staticmethod
    def write_planner_best_plan(
        context: Context,
        best_plan: str,
        best_plan_file_path: Optional[str] = PLANNER_BEST_PLAN_FILE,
    ) -> None:
        """
        Write planner best plan.
        :param context: task execution context
        :param best_plan: serialized parent JSON string
        :param best_plan_file_path: the path to the best plan file
        """
        planner_path = Workspace.get_planner_path(context)
        parent_path = planner_path / best_plan_file_path
        with open(parent_path, "w") as f:
            f.write(best_plan)

    @staticmethod
    def get_planner_parent_info_path(context: Context) -> str:
        """Return the absolute path to planner/parent_info.json."""
        return str(Workspace.get_planner_path(context) / PLANNER_PARENT_FILE)

    @staticmethod
    def get_planner_best_plan_path(
        context: Context, best_plan_file_path: Optional[str] = PLANNER_BEST_PLAN_FILE
    ) -> str:
        """Return the absolute path to planner/best_plan.txt."""
        return str(Workspace.get_planner_path(context) / best_plan_file_path)

    # -----------------------------
    # Executor Utilities
    # -----------------------------
    @staticmethod
    def get_executor_path(context: Context, create: bool = True) -> Path:
        """
        get executor path
        """
        path = Workspace._get_iteration_path(context) / "executor"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_executor_candidate_path(context: Context, candidate_idx: int) -> str:
        """
        get executor candidate path
        """
        executor_path = Workspace.get_executor_path(context)
        candidate_path = executor_path / str(candidate_idx)

        # Create the directory if it does not exist
        if not candidate_path.exists():
            candidate_path.mkdir(parents=True, exist_ok=True)

        return str(candidate_path)

    @staticmethod
    def write_executor_history(context: Context, history_json: str) -> None:
        """
        Write executor evolution history.
        :param context: task execution context
        :param history_json: serialized history JSON string
        """
        executor_path = Workspace.get_executor_path(context)
        history_path = executor_path / EXECUTOR_HISTORY_FILE
        with open(history_path, "w") as f:
            f.write(history_json)

    @staticmethod
    def write_executor_best_solution(
        context: Context,
        src_solution_path: str,
        best_solution_file_path: Optional[str] = EXECUTOR_BEST_SOLUTION_FILE,
    ) -> None:
        """
        Write or copy the best solution file into executor directory.
        :param context: task execution context
        :param src_solution_path: source best_solution.py file path
        :param best_solution_file_path: best solution file path
        """
        executor_path = Workspace.get_executor_path(context)
        dst = executor_path / best_solution_file_path
        if os.path.exists(src_solution_path):
            with open(src_solution_path, "r") as src, open(dst, "w") as dst_f:
                dst_f.write(src.read())

    @staticmethod
    def write_executor_best_eval(
        context: Context,
        src_evaluation_path: str,
        best_evaluation_file_path: Optional[str] = EXECUTOR_BEST_EVALUATION_FILE,
    ) -> None:
        """
        Write the best evaluation JSON into executor directory.
        :param context: task execution context
        :param src_evaluation_path: source evaluation file path
        :param best_evaluation_file_path: best evaluation file path
        """
        executor_path = Workspace.get_executor_path(context)
        dst = executor_path / best_evaluation_file_path
        if os.path.exists(src_evaluation_path):
            with open(src_evaluation_path, "r") as src, open(dst, "w") as dst_f:
                dst_f.write(src.read())

    @staticmethod
    def write_executor_file(context: Context, path: str, file_content: str) -> str:
        """
        Write an executor file to the given absolute or workspace-relative path.

        Args:
            context (Context): Task execution context.
            path (str): Target file path. Can be:
                - A relative path under executor workspace (e.g. "evaluation1_2.json")
                - An absolute path
            file_content (str): Content to write to the file.

        Returns:
            str: Absolute file path written to.
        """
        # Convert to Path object
        target_path = Path(path)

        # Ensure parent directories exist
        os.makedirs(target_path.parent, exist_ok=True)

        # Write file content
        try:
            with open(target_path, "w") as f:
                f.write(file_content)
        except Exception as e:
            raise RuntimeError(f"Failed to write executor file to {target_path}: {e}")

        return str(target_path)

    @staticmethod
    def get_executor_history_path(context: Context) -> str:
        """Return the absolute path to executor/history.json."""
        return str(Workspace.get_executor_path(context) / EXECUTOR_HISTORY_FILE)

    @staticmethod
    def get_executor_best_solution_path(
        context: Context,
        best_solution_file_path: Optional[str] = EXECUTOR_BEST_SOLUTION_FILE,
    ) -> str:
        """Return the absolute path to executor/best_solution.py."""
        return str(Workspace.get_executor_path(context) / best_solution_file_path)

    @staticmethod
    def get_executor_best_evaluation_path(
        context: Context,
        best_evaluation_file_path: Optional[str] = EXECUTOR_BEST_EVALUATION_FILE,
    ) -> str:
        """Return the absolute path to executor/best_evaluation.json."""
        return str(Workspace.get_executor_path(context) / best_evaluation_file_path)

    # -----------------------------
    # Summarizer Utilities
    # -----------------------------
    @staticmethod
    def get_summarizer_path(context: Context, create: bool = True) -> Path:
        """
        get summarizer path
        """
        path = Workspace._get_iteration_path(context) / "summarizer"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_summarizer_best_summary_path(
        context: Context,
        best_summary_file_path: Optional[str] = SUMMARIZER_BEST_SUMMARY_FILE,
    ) -> str:
        """Return the absolute path to summary/best_summary.json."""
        return str(Workspace.get_summarizer_path(context) / best_summary_file_path)

    @staticmethod
    def write_summarizer_best_summary(
        context: Context,
        summary: str,
        best_summary_file_path: Optional[str] = SUMMARIZER_BEST_SUMMARY_FILE,
    ) -> None:
        """
        Write executor evolution history.
        :param context: task execution context
        :param summary: serialized summary JSON string
        :param best_summary_file_path: best summary file path
        """
        summary_path = Workspace.get_summarizer_best_summary_path(
            context, best_summary_file_path
        )
        with open(summary_path, "w") as f:
            f.write(summary)

    # -----------------------------
    # Evaluator Utilities
    # -----------------------------
    @staticmethod
    def get_evaluator_path(context: Context, create: bool = True) -> Path:
        """
        get evaluator path
        """
        path = Workspace._get_iteration_path(context) / "evaluator"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
