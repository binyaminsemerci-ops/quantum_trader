"""Minimal runtime configuration for tests and CI.

This file provides a tiny, safe implementation that other modules import
from. It intentionally keeps behavior simple: values are read from
environment variables with sensible defaults so tests and CI don't need a
complex external config system.
"""

from __future__ import annotations

import os

from typing import Dict, Any
from types import SimpleNamespace


# A simple default exchange name used by the exchange factory when no
# explicit name is provided. Keep this stable across tests.
DEFAULT_EXCHANGE: str = os.environ.get("DEFAULT_EXCHANGE", "binance")


# Common quote symbols used across the codebase. Keep these simple so tests
# and CI which import them from `config.config` don't fail.
DEFAULT_QUOTE: str = os.environ.get("DEFAULT_QUOTE", "USDC")
FUTURES_QUOTE: str = os.environ.get("FUTURES_QUOTE", DEFAULT_QUOTE)


def make_pair(base: str, quote: str | None = None) -> str:
    """Build a simple market pair string used by helpers and tests.

    The implementation is intentionally minimal: it concatenates base+quote
    so callers can import and use it without requiring exchange SDKs.
    """
    q = quote or FUTURES_QUOTE
    return f"{base}{q}"


def _get_dashboard_settings() -> Dict[str, Any]:
    """Try to import dashboard settings; return empty dict if unavailable."""
    try:
        from backend.routes.settings import SETTINGS
        return SETTINGS if isinstance(SETTINGS, dict) else {}
    except (ImportError, AttributeError):
        return {}


def load_config() -> Any:
    """Return a minimal settings object (SimpleNamespace) for tests and CI.

    Many modules in the repository access configuration attributes (e.g.
    `cfg.binance_api_key`) so returning a SimpleNamespace preserves
    attribute-style access while keeping the implementation tiny.
    
    Priority for API keys (first match wins):
    1. Dashboard settings (stored via /settings API)
    2. Environment variables
    """
    dashboard = _get_dashboard_settings()
    
    ns = SimpleNamespace(
        DEFAULT_EXCHANGE=os.environ.get("DEFAULT_EXCHANGE", DEFAULT_EXCHANGE),
        QUANTUM_TRADER_DATABASE_URL=os.environ.get(
            "QUANTUM_TRADER_DATABASE_URL", "sqlite:///./db.sqlite3"
        ),
        DEFAULT_QUOTE=os.environ.get("DEFAULT_QUOTE", DEFAULT_QUOTE),
        FUTURES_QUOTE=os.environ.get("FUTURES_QUOTE", FUTURES_QUOTE),
        # Priority: dashboard settings > environment variables
        binance_api_key=dashboard.get("api_key") or os.environ.get("BINANCE_API_KEY"),
        binance_api_secret=dashboard.get("api_secret") or os.environ.get("BINANCE_API_SECRET"),
        # TESTNET KEYS (get from https://testnet.binancefuture.com)
        binance_testnet_api_key=os.environ.get("BINANCE_TESTNET_API_KEY"),
        binance_testnet_secret_key=os.environ.get("BINANCE_TESTNET_SECRET_KEY"),
        coinbase_api_key=dashboard.get("COINBASE_API_KEY") or os.environ.get("COINBASE_API_KEY"),
        coinbase_api_secret=dashboard.get("COINBASE_API_SECRET") or os.environ.get("COINBASE_API_SECRET"),
    )

    # Attach convenience helpers to the namespace so tests that import
    # these via `config.config` still find them.
    setattr(ns, "make_pair", make_pair)

    return ns


def masked_config_summary(cfg: Any) -> Dict[str, Any]:
    """Return a tiny, masked summary of secrets suitable for health endpoints.

    The real application might omit or redact secrets; for CI/tests we
    provide a deterministic (non-secret) structure.
    """
    return {
        "has_binance_keys": bool(
            getattr(cfg, "binance_api_key", None)
            and getattr(cfg, "binance_api_secret", None)
        ),
        "has_coinbase_keys": bool(
            getattr(cfg, "coinbase_api_key", None)
            and getattr(cfg, "coinbase_api_secret", None)
        ),
    }


# Convenience variable expected by some modules (and by the shim).
settings: Any = load_config()


# ==============================================================================
# UNIVERSE CONFIGURATION
# ==============================================================================

def get_qt_symbols() -> str:
    """Get explicit QT_SYMBOLS environment variable (comma-separated list).
    
    Returns:
        Empty string if not set, otherwise the raw comma-separated string.
    """
    return os.environ.get("QT_SYMBOLS", "").strip()


def get_qt_universe() -> str:
    """Get QT_UNIVERSE profile name.
    
    Returns:
        Universe profile name, defaults to 'megacap' for safety.
    """
    return os.environ.get("QT_UNIVERSE", "megacap").strip().lower()


def get_qt_max_symbols() -> int:
    """Get QT_MAX_SYMBOLS limit with bounds checking.
    
    Returns:
        Integer between 10 and 1000, defaults to 300.
    """
    try:
        max_symbols = int(os.environ.get("QT_MAX_SYMBOLS", "300"))
        # Enforce safety bounds
        if max_symbols < 10:
            return 10
        if max_symbols > 1000:
            return 1000
        return max_symbols
    except (ValueError, TypeError):
        return 300


def get_qt_countertrend_min_conf() -> float:
    """Get QT_COUNTERTREND_MIN_CONF threshold for allowing SHORT trades in uptrend.
    
    When AI confidence >= this threshold, SHORT trades in uptrend are allowed.
    This solves AI short-bias without removing EMA200 safety.
    
    Returns:
        Float between 0.40 and 0.90, defaults to 0.50 (50% confidence).
    """
    try:
        threshold = float(os.environ.get("QT_COUNTERTREND_MIN_CONF", "0.50"))
        # Enforce safety bounds
        if threshold < 0.40:
            return 0.40
        if threshold > 0.90:
            return 0.90
        return threshold
    except (ValueError, TypeError):
        return 0.50


def get_model_supervisor_enabled() -> bool:
    """Get QT_AI_MODEL_SUPERVISOR_ENABLED flag.
    
    Returns:
        True if Model Supervisor should be initialized, False otherwise.
        Defaults to True for observation mode.
    """
    return os.environ.get("QT_AI_MODEL_SUPERVISOR_ENABLED", "true").strip().lower() == "true"


def get_model_supervisor_mode() -> str:
    """Get MODEL_SUPERVISOR_MODE or QT_MODEL_SUPERVISOR_MODE (OBSERVE/ENFORCED/ADVISORY).
    
    Returns:
        String mode, defaults to 'ENFORCED' for bias blocking.
    """
    mode = os.environ.get("QT_MODEL_SUPERVISOR_MODE", 
                         os.environ.get("MODEL_SUPERVISOR_MODE", "ENFORCED")).strip().upper()
    if mode not in ["OBSERVE", "ENFORCED", "ADVISORY"]:
        return "ENFORCED"
    return mode


def get_policy_min_confidence_trending() -> float:
    """Get minimum confidence threshold for TRENDING regime.
    
    This is the base confidence threshold used by the orchestrator policy
    when the market is in a TRENDING state. The value is adjustable via
    environment variable for fine-tuning.
    
    Returns:
        Float value between 0.0 and 1.0, defaults to 0.32.
        
    Environment Variable:
        QT_POLICY_MIN_CONF_TRENDING: Set custom threshold (e.g., "0.35")
    """
    raw = os.environ.get("QT_POLICY_MIN_CONF_TRENDING", "0.32")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.32
    # Clamp to reasonable range [0.20, 0.50]
    return max(0.20, min(0.50, value))


def get_policy_min_confidence_ranging() -> float:
    """Get minimum confidence threshold for RANGING regime.
    
    Returns:
        Float value between 0.0 and 1.0, defaults to 0.40.
        
    Environment Variable:
        QT_POLICY_MIN_CONF_RANGING: Set custom threshold (e.g., "0.43")
    """
    raw = os.environ.get("QT_POLICY_MIN_CONF_RANGING", "0.40")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.40
    # Clamp to reasonable range [0.25, 0.55]
    return max(0.25, min(0.55, value))


def get_policy_min_confidence_normal() -> float:
    """Get minimum confidence threshold for NORMAL regime.
    
    Returns:
        Float value between 0.0 and 1.0, defaults to 0.38.
        
    Environment Variable:
        QT_POLICY_MIN_CONF_NORMAL: Set custom threshold (e.g., "0.40")
    """
    raw = os.environ.get("QT_POLICY_MIN_CONF_NORMAL", "0.38")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.38
    # Clamp to reasonable range [0.22, 0.52]
    return max(0.22, min(0.52, value))


# ==============================================================================
# RISK & SAFETY CONFIGURATION
# ==============================================================================

def get_risk_per_trade() -> float:
    """Get risk percentage per trade.
    
    Returns:
        Float value between 0.1% and 2.0%, defaults to 0.75%.
        
    Environment Variable:
        QT_RISK_PER_TRADE: Set custom risk percentage (e.g., "0.5" for 0.5%)
    """
    raw = os.environ.get("QT_RISK_PER_TRADE", "0.75")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.75
    # Clamp to reasonable range [0.1, 2.0]
    return max(0.1, min(2.0, value))


def get_max_positions_per_symbol() -> int:
    """Get maximum number of open positions allowed per symbol.
    
    Returns:
        Integer between 1 and 5, defaults to 2.
        
    Environment Variable:
        QT_MAX_POSITIONS_PER_SYMBOL: Set max positions per symbol (e.g., "1")
    """
    raw = os.environ.get("QT_MAX_POSITIONS_PER_SYMBOL", "2")
    try:
        value = int(raw)
    except (ValueError, TypeError):
        value = 2
    # Clamp to reasonable range [1, 5]
    return max(1, min(5, value))


def get_trail_callback_rate() -> float:
    """Get trailing stop callback rate for Binance.
    
    Binance requires callbackRate to be between 0.1 and 5.0 (percentage).
    
    Returns:
        Float value between 0.1 and 5.0, defaults to 1.5 (1.5%).
        
    Environment Variable:
        QT_TRAIL_CALLBACK: Set custom callback rate (e.g., "2.0" for 2%)
    """
    raw = os.environ.get("QT_TRAIL_CALLBACK", "1.5")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 1.5
    # Strict validation: Binance requires [0.1, 5.0]
    if value < 0.1:
        return 0.1
    if value > 5.0:
        return 5.0
    return value


def get_uptrend_short_exception_threshold() -> float:
    """Get minimum confidence required to allow SHORT in UPTREND as exception.
    
    Returns:
        Float value between 0.60 and 0.90, defaults to 0.65.
        
    Environment Variable:
        QT_UPTREND_SHORT_EXCEPTION_CONF: Set threshold (e.g., "0.70")
    """
    raw = os.environ.get("QT_UPTREND_SHORT_EXCEPTION_CONF", "0.65")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.65
    # Clamp to reasonable range [0.60, 0.90]
    return max(0.60, min(0.90, value))


__all__ = [
    "DEFAULT_EXCHANGE",
    "load_config",
    "settings",
    "get_qt_symbols",
    "get_qt_universe",
    "get_qt_max_symbols",
    "get_qt_countertrend_min_conf",
    "get_model_supervisor_enabled",
    "get_model_supervisor_mode",
    "get_policy_min_confidence_trending",
    "get_policy_min_confidence_ranging",
    "get_policy_min_confidence_normal",
    "get_risk_per_trade",
    "get_max_positions_per_symbol",
    "get_trail_callback_rate",
    "get_uptrend_short_exception_threshold",
]
