import os


def test_autotrader_order_gate(monkeypatch, tmp_path):
    calls = {}

    class DummyClient:
        def create_order(self, symbol, side, qty, order_type="MARKET"):
            calls['called'] = True
            return {"status": "ok", "order_id": "d-1", "qty": qty}

    def fake_get_client(name=None, api_key=None, api_secret=None):
        return DummyClient()

    monkeypatch.setenv("AUTOTRADER_ALLOW_REAL_ORDERS", "0")
    monkeypatch.setattr("backend.utils.exchanges.get_exchange_client", fake_get_client)

    # import and call send_order with dry_run False -> should simulate and not call create_order
    from scripts.autotrader import send_order

    resp = send_order("BTCUSDT", "BUY", 0.01, 100.0, dry_run=False)
    assert resp.get("status") == "simulated"
    assert 'called' not in calls

    # enable allow flag and call again
    monkeypatch.setenv("AUTOTRADER_ALLOW_REAL_ORDERS", "1")
    resp2 = send_order("BTCUSDT", "BUY", 0.02, 100.0, dry_run=False)
    assert resp2.get("status") == "ok"
    assert calls.get('called') is True
