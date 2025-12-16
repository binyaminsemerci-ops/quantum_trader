import pytest
from httpx import AsyncClient, ASGITransport
from backend.config.risk import load_risk_config  # type: ignore[import-error]
from backend.main import app
from backend.services.risk.risk_guard import RiskGuardService  # type: ignore[import-error]

ADMIN_TOKEN = "test-admin-token"
ADMIN_HEADERS = {"X-Admin-Token": ADMIN_TOKEN}


@pytest.fixture(autouse=True)
async def _ensure_risk_guard():
    guard = RiskGuardService(load_risk_config())
    app.state.risk_guard = guard
    yield
    if hasattr(app.state, "risk_guard"):
        delattr(app.state, "risk_guard")


@pytest.mark.asyncio
async def test_get_trades():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        unauthorized = await ac.get("/trades")
        assert unauthorized.status_code == 401

        response = await ac.get("/trades", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_trade_rejected_when_kill_switch_enabled():
    transport = ASGITransport(app=app)
    payload = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.1, "price": 100.0}

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        guard = app.state.risk_guard
        await guard.reset()
        await guard.set_kill_switch(True)
        response = await ac.post("/trades", json=payload, headers=ADMIN_HEADERS)
        await guard.set_kill_switch(None)

    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["reason"] == "kill_switch"


@pytest.mark.asyncio
async def test_trade_allowed_updates_risk_state():
    transport = ASGITransport(app=app)
    payload = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.1, "price": 100.0}

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        guard = app.state.risk_guard
        await guard.reset()
        response = await ac.post("/trades", json=payload, headers=ADMIN_HEADERS)

        assert response.status_code == 201
        snapshot = await guard.snapshot()
    assert snapshot["state"]["trade_count"] >= 1


@pytest.mark.asyncio
async def test_risk_snapshot_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        unauthorized = await ac.get("/risk")
        assert unauthorized.status_code == 401

        response = await ac.get("/risk", headers={"X-Admin-Token": ADMIN_TOKEN})

    assert response.status_code == 200
    payload = response.json()
    assert "config" in payload and "state" in payload
    assert "kill_switch_override" in payload["state"]
    assert "kill_switch_state" in payload["state"]
    assert "positions" in payload


@pytest.mark.asyncio
async def test_risk_kill_switch_and_reset_routes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        guard = app.state.risk_guard
        await guard.reset()

        forbidden = await ac.post("/risk/kill-switch", json={"enabled": True}, headers={"X-Admin-Token": "wrong"})
        assert forbidden.status_code == 403

        response_override = await ac.post(
            "/risk/kill-switch",
            json={"enabled": True, "reason": "pytest"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
        assert response_override.status_code == 200
        payload_override = response_override.json()
        assert payload_override["snapshot"]["state"]["kill_switch_override"] is True
        assert payload_override["snapshot"]["state"]["kill_switch_state"]["reason"] == "pytest"

        response_reset = await ac.post(
            "/risk/reset",
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
        assert response_reset.status_code == 200
        payload_reset = response_reset.json()
        assert payload_reset["snapshot"]["state"]["kill_switch_override"] is None
        assert payload_reset["snapshot"]["state"]["kill_switch_state"] is None

        # ensure guard state matches API payload
        snapshot = await guard.snapshot()
        assert snapshot["state"]["kill_switch_override"] is None
        assert snapshot["state"]["kill_switch_state"] is None
