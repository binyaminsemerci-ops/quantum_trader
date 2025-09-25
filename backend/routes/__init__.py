"""Backend routes package: expose route submodules for easy imports.

Some tests (and backend.main) import submodules like
`from backend.routes import trades, stats, chart, settings, binance, health`.
To make that work when importing the package, re-export those modules here.
"""

from . import external_data, health
from . import trades, stats, chart, settings

__all__ = [
	"external_data",
	"health",
	"trades",
	"stats",
	"chart",
	"settings",
]
