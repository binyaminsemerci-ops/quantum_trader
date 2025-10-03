"""backend.routes package

Import the submodules at package import time so that router-side-effects are
registered predictably. This avoids import-order issues where tests may import
``backend.main:app`` before submodules are loaded which can cause routes like
``/candles`` to be missing in some CI/test harnesses.
"""

# Import submodules to ensure their top-level router objects and handlers are
# created when the package is imported. Keep the list explicit so linters
# and tooling can see what's exported.
from . import (
    ai_trading,
    binance,
    candles,
    chart,
    enhanced_api,
    health,
    layout,
    portfolio,
    prices,
    settings,
    signals,
    stats,
    stress,
    trade_logs,
    trades,
    trading,
    watchlist,
)

__all__ = [
    "trades",
    "stats",
    "chart",
    "settings",
    "binance",
    "signals",
    "prices",
    "candles",
    "stress",
    "trade_logs",
    "health",
    "watchlist",
    "layout",
    "portfolio",
    "trading",
    "enhanced_api",
    "ai_trading",
]
