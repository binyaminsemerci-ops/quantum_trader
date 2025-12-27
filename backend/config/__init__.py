"""Configuration helpers for Quantum Trader backend."""

from typing import Any
import os

# Import from canonical config if available, otherwise use defaults
try:
    from config.config import load_config, settings, DEFAULT_EXCHANGE, DEFAULT_QUOTE, FUTURES_QUOTE, make_pair  # type: ignore[import-error]
except Exception:  # pragma: no cover - fallback for analysis environments
    # Default values when canonical config is unavailable
    DEFAULT_EXCHANGE = os.environ.get("DEFAULT_EXCHANGE", "binance")
    DEFAULT_QUOTE = os.environ.get("DEFAULT_QUOTE", "USDC")
    FUTURES_QUOTE = os.environ.get("FUTURES_QUOTE", DEFAULT_QUOTE)
    
    def make_pair(base: str, quote: str | None = None) -> str:
        """Build a simple market pair string."""
        return f"{base}{quote or DEFAULT_QUOTE}"

    def load_config() -> Any:  # type: ignore[override]
        """Fallback minimal config loader used when the canonical
        ``config.config`` module is unavailable (static analysis, isolated tests).

        This implementation mirrors the priority logic of the real loader:
        1. Dashboard settings (if import succeeds)
        2. Environment variables
        3. None (attributes remain unset)
        """
        dashboard: dict[str, Any] = {}
        try:
            from backend.routes.settings import SETTINGS  # type: ignore
            if isinstance(SETTINGS, dict):
                dashboard = SETTINGS
        except Exception:  # pragma: no cover - best effort only
            pass

        class _Stub:
            binance_api_key = dashboard.get("api_key") or os.getenv("BINANCE_API_KEY")
            binance_api_secret = dashboard.get("api_secret") or os.getenv("BINANCE_API_SECRET")

        return _Stub()

    settings = load_config()

from .execution import ExecutionConfig, load_execution_config
from .liquidity import LiquidityConfig, load_liquidity_config
from .risk import RiskConfig, load_risk_config

__all__ = [
    "load_config",
    "settings",
    "DEFAULT_EXCHANGE",
    "DEFAULT_QUOTE",
    "FUTURES_QUOTE",
    "make_pair",
    "RiskConfig",
    "load_risk_config",
    "LiquidityConfig",
    "load_liquidity_config",
    "ExecutionConfig",
    "load_execution_config",
]
