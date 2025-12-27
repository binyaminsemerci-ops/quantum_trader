"""
AI Trading Engine wrapper.

This module re-exports AI trading engine from the ai submodule
to maintain backward compatibility with imports.
"""

from backend.services.ai.ai_trading_engine import (
    AITradingEngine,
    create_ai_trading_engine,
)

__all__ = [
    "AITradingEngine",
    "create_ai_trading_engine",
]
