ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


def test_risk_snapshot_with_valid_token(client):
    response = client.get("/risk", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert payload["config"]["allowed_symbols"]
    assert payload["state"]["trade_count"] == 0
    assert payload["state"]["kill_switch_override"] is None
    assert "kill_switch_state" in payload["state"]
    assert "positions" in payload


def test_kill_switch_override_blocks_trades(client):
    response = client.post(
        "/risk/kill-switch",
        json={"enabled": True, "reason": "unit_test"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["snapshot"]["state"]["kill_switch_override"] is True
    assert payload["snapshot"]["state"]["kill_switch_state"]["reason"] == "unit_test"

    trade_payload = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "qty": 0.1,
        "price": 1000.0,
    }
    trade_response = client.post("/trades", json=trade_payload, headers=ADMIN_HEADERS)
    assert trade_response.status_code == 403
    trade_body = trade_response.json()
    assert trade_body["detail"]["reason"] == "kill_switch"

    snapshot_response = client.get("/risk", headers=ADMIN_HEADERS)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert snapshot["state"]["kill_switch_override"] is True
    assert snapshot["state"]["kill_switch_state"]["enabled"] is True
    assert snapshot["positions"]["total_notional"] >= 0.0


def test_reset_endpoint_clears_state(client):
    trade_payload = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "qty": 0.5,
        "price": 1000.0,
    }
    trade_response = client.post("/trades", json=trade_payload, headers=ADMIN_HEADERS)
    assert trade_response.status_code == 201

    pre_reset = client.get("/risk", headers=ADMIN_HEADERS)
    assert pre_reset.status_code == 200
    snapshot_before = pre_reset.json()
    assert snapshot_before["state"]["trade_count"] == 1

    toggle_response = client.post(
        "/risk/kill-switch",
        json={"enabled": False},
        headers=ADMIN_HEADERS,
    )
    assert toggle_response.status_code == 200

    reset_response = client.post("/risk/reset", headers=ADMIN_HEADERS)
    assert reset_response.status_code == 200
    snapshot_after = reset_response.json()["snapshot"]
    assert snapshot_after["state"]["trade_count"] == 0
    assert snapshot_after["state"]["kill_switch_override"] is None
    assert snapshot_after["state"]["kill_switch_state"] is None
    assert "positions" in snapshot_after


def test_get_kill_switch_state_endpoint(client):
    response = client.post(
        "/risk/kill-switch",
        json={"enabled": True, "reason": "inspection"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200

    state_response = client.get("/risk/kill-switch", headers=ADMIN_HEADERS)
    assert state_response.status_code == 200
    payload = state_response.json()
    state = payload["snapshot"]["state"]
    assert state["kill_switch_override"] is True
    assert state["kill_switch_state"]["reason"] == "inspection"

    client.post("/risk/reset", headers=ADMIN_HEADERS)
