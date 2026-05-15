"""Framework - Core agent architectures for LoongFlow.

This module provides the core implementations for different agent paradigms:
- PESAgent: Evolutionary algorithms for optimization
- ReActAgent: Standard reasoning loops
- Base classes for building custom agents
"""

from . import base
from . import pes
from . import react
from . import claude_code

__all__ = ["base", "pes", "react", "claude_code"]
