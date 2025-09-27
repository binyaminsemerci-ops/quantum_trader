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
