"""Backend routes package: expose route submodules for easy imports.

Some tests (and backend.main) import submodules like
`from backend.routes import trades, stats, chart, settings, binance, health`.
To make that work when importing the package, re-export those modules here.
"""

"""Routes package. Avoid importing submodules at package import time.

Tests and application code import submodules like
`backend.routes.external_data` directly; importing submodules here caused
circular import problems in the rebased worktree. Keep this file minimal so
submodules are imported on demand.
"""

__all__: list[str] = []
