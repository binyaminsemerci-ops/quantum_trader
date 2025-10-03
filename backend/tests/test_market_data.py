import copy
import types

import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.routes import settings as route_settings
from backend.utils import market_data


@pytest.fixture(autouse=True)
def reset_settings():
    original = copy.deepcopy(route_settings.SETTINGS)
    try:
        yield
    finally:
        route_settings.SETTINGS.clear()
        route_settings.SETTINGS.update(original)


def test_fetch_recent_candles_live_uses_ccxt(monkeypatch):
    class FakeExchange:
        def __init__(self, params=None):
            self.params = params
            self.closed = False

        def fetch_ohlcv(self, market, timeframe="1m", limit=2):
            return [
                [1700000000000, 1, 2, 0.5, 1.5, 10],
                [1700000006000, 1.6, 2.1, 1.4, 1.9, 12],
            ]

        def close(self):
            self.closed = True

    fake_ccxt = types.SimpleNamespace(binance=lambda params=None: FakeExchange(params))
    monkeypatch.setattr(market_data, "ccxt", fake_ccxt)

    route_settings.SETTINGS["ENABLE_LIVE_MARKET_DATA"] = "1"

    candles = market_data.fetch_recent_candles("BTCUSDT", limit=2)
    assert len(candles) == 2
    assert candles[0]["close"] == 1.5
    assert candles[1]["close"] == 1.9


@pytest.mark.asyncio
async def test_binance_routes_live(monkeypatch):
    class DummyClient:
        def spot_balance(self):
            return {"asset": "USDC", "free": 42.0}

        def futures_balance(self):
            return {"asset": "USDT", "balance": 11.5}

        def fetch_recent_trades(self, symbol, limit=5):
            return [
                {"symbol": symbol, "qty": "1", "price": "100", "side": "buy"}
                for _ in range(limit)
            ]

    from backend.routes import binance as binance_route

    monkeypatch.setattr(
        binance_route,
        "get_exchange_client",
        lambda name, api_key=None, api_secret=None: DummyClient(),
    )

    route_settings.SETTINGS["ENABLE_LIVE_MARKET_DATA"] = True

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        spot = await ac.get("/binance/spot-balance")
        fut = await ac.get("/binance/futures-balance")
        trades = await ac.get(
            "/binance/recent-trades", params={"symbol": "ETHUSDC", "limit": 2}
        )

    assert spot.status_code == 200
    assert spot.json()["balance"]["free"] == 42.0
    assert fut.json()["balance"]["balance"] == 11.5
    assert len(trades.json()["trades"]) == 2


@pytest.mark.asyncio
async def test_binance_routes_demo_when_live_disabled():
    route_settings.SETTINGS["ENABLE_LIVE_MARKET_DATA"] = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        spot = await ac.get("/binance/spot-balance")
    assert spot.status_code == 200
    assert spot.json()["balance"]["source"] == "demo"
