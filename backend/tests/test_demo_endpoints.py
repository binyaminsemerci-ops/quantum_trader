from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_prices_recent_shape():
    r = client.get("/prices/recent")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "time" in first and "open" in first and "close" in first


def test_signals_recent_shape():
    r = client.get("/signals/recent")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "symbol" in first and "score" in first and "timestamp" in first


def test_candles_endpoint():
    r = client.get("/candles/?symbol=BTCUSDT&limit=5")
    assert r.status_code == 200
    data = r.json()
    # Expecting a dict with 'symbol' and 'candles' (list)
    assert isinstance(data, dict)
    assert "symbol" in data and "candles" in data
    candles = data["candles"]
    assert isinstance(candles, list)
    assert len(candles) <= 5
    if len(candles) > 0:
        c = candles[0]
        assert "timestamp" in c and "open" in c and "close" in c
