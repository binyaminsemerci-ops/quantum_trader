# Binance trading service implementation
from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TradingEngineProto(Protocol):
    """Protocol describing the attributes/methods the API layer expects.

    This is intentionally permissive; real implementation can exceed this.
    """

    # Status / lifecycle
    @property
    def is_running(self) -> bool:  # pragma: no cover - simple property contract
        ...

    def get_trading_status(self) -> dict[str, Any]: ...

    async def start_trading(self, interval_minutes: int) -> None: ...

    def stop_trading(self) -> None: ...

    # Market data & cycles
    async def get_market_data(self, symbol: str) -> list[dict[str, Any]]: ...

    async def run_trading_cycle(self) -> list[Any]: ...

    def get_trading_symbols(self) -> list[str]: ...

    # AI agent interface (nested object assumed)
    @property
    def ai_agent(self) -> Any:  # could refine later
        ...

    # Underlying exchange client (optional, used for balances/positions)
    @property
    def client(self) -> Any:  # pragma: no cover - structural only
        ...

    # Risk / sizing helpers
    def get_position_size(
        self, symbol: str, price: float, confidence: float
    ) -> float: ...

    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        confidence: float,
    ) -> dict[str, Any]: ...

    def get_account_balance(self) -> dict[str, float]: ...

    # Configurable parameters (attributes set dynamically)
    max_position_size_usdc: float
    min_confidence_threshold: float
    risk_per_trade: float


class TradingEngine(TradingEngineProto):  # type: ignore[misc]
    """Concrete minimal stub implementing the protocol for development/testing.

    Real logic should replace these placeholders; attributes exist to satisfy the
    FastAPI route expectations and mypy protocol checks.
    """

    def __init__(self) -> None:  # pragma: no cover - trivial
        self._running = False
        self.max_position_size_usdc = 1000.0
        self.min_confidence_threshold = 0.2
        self.risk_per_trade = 0.01
        self._symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
        self._ai_agent = type(
            "_AIAgentStub",
            (),
            {  # simple dynamic stub
                "predict_for_symbol": staticmethod(
                    lambda data: {"action": "HOLD", "score": 0.5}
                ),
                "get_metadata": staticmethod(lambda: {}),
                "model": None,
                "scaler": None,
                "model_path": "ai_engine/models/xgb_model.pkl",
                "scaler_path": "ai_engine/models/scaler.pkl",
            },
        )()
        # lightweight client stub with futures_position_information
        self._client = type(
            "_ClientStub",
            (),
            {
                "futures_position_information": staticmethod(
                    lambda: [
                        {
                            "symbol": "BTCUSDT",
                            "positionAmt": "0.01",
                            "entryPrice": "50000",
                            "markPrice": "50100",
                            "unRealizedProfit": "10",
                        }
                    ],
                )
            },
        )()

    # Properties / status
    @property
    def is_running(self) -> bool:
        return self._running

    def get_trading_status(self) -> dict[str, Any]:
        return {"running": self._running, "symbols": self._symbols}

    async def start_trading(self, interval_minutes: int) -> None:  # pragma: no cover
        self._running = True
        # Fake async loop iteration placeholder
        await asyncio.sleep(0)

    def stop_trading(self) -> None:
        self._running = False

    # Market data & cycles
    async def get_market_data(
        self, symbol: str
    ) -> list[dict[str, Any]]:  # pragma: no cover
        # Return tiny synthetic OHLCV list containing 'close' key
        return [{"close": 50000.0, "open": 49900.0, "high": 50100.0, "low": 49800.0}]

    async def run_trading_cycle(self) -> list[Any]:  # pragma: no cover
        await asyncio.sleep(0)
        return []

    def get_trading_symbols(self) -> list[str]:
        return list(self._symbols)

    # AI agent
    @property
    def ai_agent(self) -> Any:  # pragma: no cover - simple passthrough
        return self._ai_agent

    @property
    def client(self) -> Any:  # pragma: no cover
        return self._client

    # Risk helpers
    def get_position_size(self, symbol: str, price: float, confidence: float) -> float:
        base = self.max_position_size_usdc * min(1.0, max(0.0, confidence))
        return round(base / max(price, 1e-6), 6)

    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        confidence: float,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "confidence": confidence,
            "status": "EXECUTED",
        }

    def get_account_balance(self) -> dict[str, float]:
        return {"USDT": 10_000.0, "BTC": 0.5}


_ENGINE_SINGLETON: TradingEngine | None = None


def get_trading_engine() -> TradingEngineProto:
    """Return a cached TradingEngine instance implementing the protocol."""
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = TradingEngine()
    return _ENGINE_SINGLETON
