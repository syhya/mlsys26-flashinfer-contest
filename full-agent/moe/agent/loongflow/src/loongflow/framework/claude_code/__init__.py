#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is init file
"""

from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent
from loongflow.framework.claude_code.general_prompt import *

__all__ = [
    "ClaudeCodeAgent",
    "GENERAL_PLANNER_SYSTEM",
    "GENERAL_EXECUTOR_SYSTEM",
    "GENERAL_SUMMARY_SYSTEM",
    "GENERAL_PLANNER_USER",
    "GENERAL_EXECUTOR_USER",
    "GENERAL_SUMMARY_USER",
    "GENERAL_EVALUATOR_SIMPLE_SYSTEM",
    "GENERAL_EVALUATOR_SIMPLE_USER",
    "GENERAL_EVALUATOR_TOOL_SYSTEM",
    "GENERAL_EVALUATOR_TOOL_USER",
    "DEFAULT_LOADED_SKILLS",
]
