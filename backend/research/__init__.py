"""
Research subsystem for automated strategy generation and evolution.

This module contains the Strategy Generator AI (SG AI) that autonomously
discovers, backtests, and deploys trading strategies.
"""

from .models import (
    StrategyConfig,
    StrategyStats,
    StrategyStatus,
    RegimeFilter,
)

from .backtest import StrategyBacktester
from .search import StrategySearchEngine
from .shadow import StrategyShadowTester
from .deployment import StrategyDeploymentManager

__all__ = [
    "StrategyConfig",
    "StrategyStats",
    "StrategyStatus",
    "RegimeFilter",
    "StrategyBacktester",
    "StrategySearchEngine",
    "StrategyShadowTester",
    "StrategyDeploymentManager",
]
