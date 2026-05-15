"""Shape-aware route selection for the merged MoE submission.

Keep the policy centralized so future A/B sweeps only need to edit one module.
The retained Triton fastpath is preserved in the tree for future experiments,
but the current promoted config keeps it disabled until a same-round full sweep
shows a real global gain.
"""

CURRENT_TRITON_FASTPATH_SEQ_LENS = frozenset()


def should_use_current_triton_path(seq_len: int) -> bool:
    """Return whether the retained Triton fastpath should handle this shape.

    The promoted configuration currently disables the Triton override for every
    shape. Keeping the helper in one place makes it trivial to re-run exact-
    shape experiments without rewriting the entrypoint logic.
    """
    return seq_len in CURRENT_TRITON_FASTPATH_SEQ_LENS
