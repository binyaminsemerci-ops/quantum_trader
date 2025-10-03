from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_ws_watchlist_accepts_connection():
    with client.websocket_connect(
        "/watchlist/ws/watchlist?symbols=BTCUSDT&limit=3"
    ) as ws:
        # should receive a JSON array quickly
        data = ws.receive_json()
        assert isinstance(data, list)


def test_ws_alerts_accepts_connection():
    with client.websocket_connect("/watchlist/ws/alerts") as ws:
        # keep connection open briefly; server should accept
        # The evaluator may not push anything immediately; ensure send/recv doesn't crash
        ws.close()
