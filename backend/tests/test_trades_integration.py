

def test_create_and_read_trade(client):
    # Create a trade
    payload = {"symbol": "ETHUSDC", "side": "SELL", "qty": 0.5, "price": 1800.0}
    post_resp = client.post("/trades", json=payload)
    assert post_resp.status_code == 200
    data = post_resp.json()
    assert data["symbol"] == "ETHUSDC"
    assert data["side"] == "SELL"

    # Read trades and ensure the created trade is present
    get_resp = client.get("/trades")
    assert get_resp.status_code == 200
    trades = get_resp.json()
    assert any(t.get("symbol") == "ETHUSDC" and t.get("side") == "SELL" for t in trades)
