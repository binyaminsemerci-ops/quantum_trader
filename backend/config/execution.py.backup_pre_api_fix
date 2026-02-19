"""Execution loop configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class ExecutionConfig:
    min_notional: float = 50.0
    max_orders: int = 10
    cash_buffer: float = 0.0
    allow_partial: bool = True
    exchange: str = "paper"
    quote_asset: str = "USDC"
    binance_testnet: bool = False


def load_execution_config() -> ExecutionConfig:
    """Read execution configuration from environment variables."""

    min_notional = _parse_float(os.getenv("QT_EXECUTION_MIN_NOTIONAL"), default=50.0)
    max_orders = max(0, _parse_int(os.getenv("QT_EXECUTION_MAX_ORDERS"), default=10))
    cash_buffer = max(0.0, _parse_float(os.getenv("QT_EXECUTION_CASH_BUFFER"), default=0.0))
    allow_partial = _parse_bool(os.getenv("QT_EXECUTION_ALLOW_PARTIAL"), default=True)
    exchange = (os.getenv("QT_EXECUTION_EXCHANGE") or "paper").strip().lower()
    quote_asset = (os.getenv("QT_EXECUTION_QUOTE_ASSET") or os.getenv("DEFAULT_QUOTE") or "USDC").strip().upper()
    binance_testnet = _parse_bool(os.getenv("QT_EXECUTION_BINANCE_TESTNET"), default=False)
    return ExecutionConfig(
        min_notional=min_notional,
        max_orders=max_orders,
        cash_buffer=cash_buffer,
        allow_partial=allow_partial,
        exchange=exchange or "paper",
        quote_asset=quote_asset or "USDT",
        binance_testnet=binance_testnet,
    )


__all__ = ["ExecutionConfig", "load_execution_config"]
