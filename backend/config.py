"""Compatibility shim so existing imports that reference ``backend.config``
continue to work while the canonical config lives in ``config.config``.

This module re-exports the minimal symbols tests and runtime code expect:
- ``settings``: a lightweight runtime settings object
- ``load_config``: function that returns the runtime config object

Keeping this shim avoids touching many imports across the codebase and is
safe to include in CI to satisfy mypy/import resolution.
"""

from typing import Any

try:
    # canonical config location
    from config.config import load_config, settings  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback for static analysis environments

    def load_config() -> Any:  # type: ignore[override]
        """Fallback stub used in analysis environments where the real
        config package isn't available. Tests that need real settings should
        import from the real module or set up fixtures.
        """

        class _Stub:
            binance_api_key = None
            binance_api_secret = None

        return _Stub()

    settings = load_config()

__all__ = ["load_config", "settings"]
