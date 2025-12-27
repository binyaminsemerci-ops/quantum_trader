import pytest

from backend.services.execution.execution import BinanceFuturesExecutionAdapter


class _DummySession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:  # pragma: no cover - awaited via asyncio.run
        self.closed = True


def test_normalize_symbol_converts_to_supported_quote():
    adapter = BinanceFuturesExecutionAdapter(api_key=None, api_secret=None, quote_asset="USDC")

    assert adapter.normalize_symbol("BTCUSDC") == "BTCUSDT"
    assert adapter.normalize_symbol("ETHUSDT") == "ETHUSDT"
    assert adapter.normalize_symbol("SOLBUSD") == "SOLUSDT"
    assert adapter._quote == "USDT"


def test_sync_close_session_closes_active_session():
    adapter = BinanceFuturesExecutionAdapter(api_key=None, api_secret=None)
    dummy = _DummySession()
    adapter._aiohttp_session = dummy

    adapter._sync_close_session()

    assert dummy.closed is True