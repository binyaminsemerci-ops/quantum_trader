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


def test_candles_slash_variants():
    # Ensure both /candles and /candles/ resolve. Some test harnesses or
    # import-order changes can cause one form to become unregistered.
    r1 = client.get("/candles?symbol=BTCUSDT&limit=2")
    r2 = client.get("/candles/?symbol=BTCUSDT&limit=2")
    assert r1.status_code == 200, f"/candles returned {r1.status_code}"
    assert r2.status_code == 200, f"/candles/ returned {r2.status_code}"


def test_trades_recent():
    r = client.get("/trades/recent?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 5
    first = data[0]
    assert "id" in first and "symbol" in first and "side" in first


def test_stats_overview():
    r = client.get("/stats/overview")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert "total_trades" in data and "pnl" in data and "open_positions" in data
