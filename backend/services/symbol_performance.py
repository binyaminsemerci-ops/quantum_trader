"""
Symbol Performance wrapper.

This module re-exports symbol performance tracking from the monitoring submodule
to maintain backward compatibility with imports.
"""

from backend.services.monitoring.symbol_performance import (
    SymbolPerformanceManager,
    SymbolPerformanceConfig,
    SymbolStats,
    TradeResult,
)

# Alias for backward compatibility
SymbolPerformanceTracker = SymbolPerformanceManager

__all__ = [
    "SymbolPerformanceManager",
    "SymbolPerformanceTracker",
    "SymbolPerformanceConfig",
    "SymbolStats",
    "TradeResult",
]
