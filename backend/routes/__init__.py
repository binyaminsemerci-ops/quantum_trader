"""Backend routes package: expose route submodules for easy imports.

Some tests (and backend.main) import submodules like
`from backend.routes import trades, stats, chart, settings, binance, health`.
To make that work when importing the package, re-export those modules here.
"""

from . import ai, backtest, binance, candles, external_data, health, trade_logs, ws
from . import trades, stats, chart, settings

__all__ = [
	"ai",
	"backtest",
	"binance",
	"candles",
	"external_data",
	"health",
	"trade_logs",
	"ws",
	"trades",
	"stats",
	"chart",
	"settings",
]
