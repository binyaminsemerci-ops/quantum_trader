"""Risk management configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_symbols(value: str | None, *, default: List[str]) -> List[str]:
    if not value:
        return default
    symbols = [item.strip().upper().replace(" ", "") for item in value.split(",") if item.strip()]
    return symbols or default


def _parse_positive_float(value: str | None) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _parse_positive_int(value: str | None) -> Optional[int]:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


@dataclass(frozen=True)
class RiskConfig:
    staging_mode: bool
    kill_switch: bool
    max_notional_per_trade: float
    max_daily_loss: float
    allowed_symbols: List[str]
    failsafe_reset_minutes: int
    risk_state_db_path: Optional[str]
    admin_api_token: Optional[str]
    max_position_per_symbol: Optional[float]
    max_gross_exposure: Optional[float]
    min_unit_price: Optional[float]
    max_unit_price: Optional[float]
    max_price_staleness_seconds: Optional[int]


def load_risk_config() -> RiskConfig:
    """Load risk guard configuration from environment variables."""

    staging_mode = _parse_bool(os.getenv("STAGING_MODE"), default=False)
    kill_switch = _parse_bool(os.getenv("QT_KILL_SWITCH"), default=False)
    max_notional = _parse_float(os.getenv("QT_MAX_NOTIONAL_PER_TRADE"), default=5000.0)
    max_daily_loss = _parse_float(os.getenv("QT_MAX_DAILY_LOSS"), default=800.0)
    
    # Empty default = use dynamic liquidity selection (top 100 by volume)
    # Set QT_ALLOWED_SYMBOLS to restrict if needed
    default_coins = []
    
    allowed_symbols = _parse_symbols(
        os.getenv("QT_ALLOWED_SYMBOLS"), default=default_coins
    )
    failsafe_reset = _parse_int(os.getenv("QT_FAILSAFE_RESET_MINUTES"), default=60)
    risk_state_db_path = os.getenv("QT_RISK_STATE_DB")
    admin_api_token = os.getenv("QT_ADMIN_TOKEN")
    max_position_per_symbol = _parse_positive_float(os.getenv("QT_MAX_POSITION_PER_SYMBOL"))
    max_gross_exposure = _parse_positive_float(os.getenv("QT_MAX_GROSS_EXPOSURE"))
    min_unit_price = _parse_positive_float(os.getenv("QT_MIN_UNIT_PRICE"))
    max_unit_price = _parse_positive_float(os.getenv("QT_MAX_UNIT_PRICE"))
    max_price_staleness_seconds = _parse_positive_int(os.getenv("QT_MAX_PRICE_STALENESS_SECONDS"))

    return RiskConfig(
        staging_mode=staging_mode,
        kill_switch=kill_switch,
        max_notional_per_trade=max_notional,
        max_daily_loss=max_daily_loss,
        allowed_symbols=allowed_symbols,
        failsafe_reset_minutes=failsafe_reset,
        risk_state_db_path=risk_state_db_path,
        admin_api_token=admin_api_token,
        max_position_per_symbol=max_position_per_symbol,
        max_gross_exposure=max_gross_exposure,
        min_unit_price=min_unit_price,
        max_unit_price=max_unit_price,
        max_price_staleness_seconds=max_price_staleness_seconds,
    )


__all__ = ["RiskConfig", "load_risk_config"]
