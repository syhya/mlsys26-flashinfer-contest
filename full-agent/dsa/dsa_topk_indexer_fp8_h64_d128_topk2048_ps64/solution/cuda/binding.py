"""Auxiliary metadata for the CUDA torch-binding submission.

The official build config for this repo uses:
- language = "cuda"
- binding = "torch"
- entry_point = "kernel.cu::dsa_forward"

This file is shipped with the submission so the packaged sources include a
plain Python-side reference to the exported symbol.
"""

from __future__ import annotations

from typing import Any, Callable


ENTRY_FILE = "kernel.cu"
ENTRY_SYMBOL = "dsa_forward"
BINDING_BACKEND = "torch"


def resolve(module: Any) -> Callable[..., Any]:
    """Return the exported callable from a compiled torch extension module."""
    return getattr(module, ENTRY_SYMBOL)
