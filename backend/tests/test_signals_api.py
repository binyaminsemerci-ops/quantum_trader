from fastapi.testclient import TestClient
import pytest

from backend.main import app


client = TestClient(app)


def test_signals_list_pagination_default():
    r = client.get("/signals/?page=1&page_size=5")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data and "items" in data
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert len(data["items"]) <= 5


def test_signals_list_symbol_filter():
    r = client.get("/signals/?page=1&page_size=10&symbol=BTCUSDT")
    assert r.status_code == 200
    data = r.json()
    items = data["items"]
    assert all(item["symbol"] == "BTCUSDT" for item in items)


def test_recent_endpoint_compatible_shape():
    r = client.get("/signals/recent?limit=3")
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list)
    assert len(arr) == 3
    first = arr[0]
    assert "id" in first and "symbol" in first and "timestamp" in first


def test_recent_endpoint_live_timeout_fallback(monkeypatch):
    from backend.routes import live_ai_signals

    async def fake_live(limit, profile):
        raise TimeoutError("simulated slow provider")

    monkeypatch.setattr(live_ai_signals, "get_live_ai_signals", fake_live)

    response = client.get("/signals/recent?limit=4")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 4


@pytest.mark.asyncio
async def test_get_live_ai_signals_prefers_agent(monkeypatch):
    from backend.routes import live_ai_signals

    class _FakeAgent:
        async def scan_top_by_volume_from_api(self, symbols, top_n=10, limit=240):
            return {
                "BTCUSDT": {"action": "BUY", "score": 0.9, "confidence": 0.95, "model": "ensemble"},
                "ETHUSDT": {"action": "SELL", "score": 0.8, "confidence": 0.85},
            }

    async def fake_get_agent():
        return _FakeAgent()

    async def fake_fetch_prices(symbols):
        return {symbol: 100.0 for symbol in symbols}

    async def fail_generate(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("Heuristic fallback should not run when agent succeeds")

    monkeypatch.setattr(live_ai_signals, "_get_agent", fake_get_agent)
    monkeypatch.setattr(live_ai_signals, "_fetch_latest_prices", fake_fetch_prices)
    monkeypatch.setattr(live_ai_signals.ai_trader, "generate_signals", fail_generate)

    signals = await live_ai_signals.get_live_ai_signals(limit=2, profile="mixed")
    assert len(signals) == 2
    assert all(signal["details"]["source"] == "XGBAgent" for signal in signals)


def test_latest_ai_signals_include_source(monkeypatch):
    async def fake_live(limit, profile):
        return [
            {
                "id": "x1",
                "symbol": "BTCUSDT",
                "side": "buy",
                "confidence": 0.75,
                "price": 100.0,
                "details": {"source": "XGBAgent", "note": "ensemble"},
                "source": "XGBAgent",
                "model": "ensemble",
            }
        ]

    monkeypatch.setattr("backend.main.get_live_ai_signals", fake_live)

    response = client.get("/api/ai/signals/latest?limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    signal = payload[0]
    assert signal["source"] == "XGBAgent"
    assert signal["model"] == "ensemble"
