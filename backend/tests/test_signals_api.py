from fastapi.testclient import TestClient
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
