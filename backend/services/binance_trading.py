# Binance trading service implementation
from __future__ import annotations

from typing import Any


class TradingEngine:
    """Basic trading engine stub."""

    def __init__(self) -> None:
        pass

    def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        return {"USDT": 1000.0, "BTC": 0.0}

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float | None = None,
    ) -> dict[str, Any]:
        """Place a trading order."""
        return {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "FILLED",
        }


def get_trading_engine() -> TradingEngine:
    """Get the trading engine instance."""
    return TradingEngine()
