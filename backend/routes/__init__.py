"""backend.routes package

Import the submodules at package import time so that router-side-effects are
registered predictably. This avoids import-order issues where tests may import
``backend.main:app`` before submodules are loaded which can cause routes like
``/candles`` to be missing in some CI/test harnesses.
"""

# Import submodules to ensure their top-level router objects and handlers are
# created when the package is imported. Keep the list explicit so linters
# and tooling can see what's exported.
from . import trades, stats, chart, settings, binance, signals, prices, candles

__all__ = [
	"trades",
	"stats",
	"chart",
	"settings",
	"binance",
	"signals",
	"prices",
	"candles",
]
