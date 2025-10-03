from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_prices_demo():
    r = client.get("/prices/recent?symbol=BTCUSDT&limit=5")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 5


def test_watchlist_prices_endpoint():
    r = client.get("/watchlist/prices?symbols=BTCUSDT,ETHUSDT&limit=5")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert any(d.get("symbol") == "BTCUSDT" for d in data)
