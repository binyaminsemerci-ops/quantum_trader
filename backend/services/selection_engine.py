"""
Selection engine wrapper.

This module re-exports selection engine functions from the execution submodule
to maintain backward compatibility with imports.
"""

from backend.services.execution.selection_engine import (
    blend_liquidity_and_model,
    score_symbols_with_agent,
    summarize_selection,
)

__all__ = [
    "blend_liquidity_and_model",
    "score_symbols_with_agent",
    "summarize_selection",
]
