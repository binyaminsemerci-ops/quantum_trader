ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


def test_scheduler_status_endpoint(client, monkeypatch):
    sample_snapshot = {
        "enabled": True,
        "running": False,
        "config": {"symbols": ["BTCUSDT"], "refresh_seconds": 180, "liquidity_refresh_seconds": 900, "execution_seconds": 1800},
        "job": None,
        "liquidity_job": None,
        "execution_job": None,
        "last_run": {"status": "ok"},
        "providers": {},
        "liquidity": {"status": "ok", "analytics": {"top_allocations": []}},
        "execution": {"status": "ok"},
        "provider_priority": ["binance", "coingecko"],
    }
    monkeypatch.setattr("backend.routes.scheduler.get_scheduler_snapshot", lambda: sample_snapshot)

    response = client.get("/scheduler/status", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    assert response.json() == sample_snapshot


def test_scheduler_warm_endpoint(client, monkeypatch):
    async def fake_run(symbols):
        return {"symbols": symbols, "last_run": {"status": "ok"}}

    monkeypatch.setattr("backend.routes.scheduler.run_market_cache_refresh_now", fake_run)
    monkeypatch.setattr("backend.routes.scheduler.get_configured_symbols", lambda: ["BTCUSDT", "ETHUSDT"])

    response = client.post("/scheduler/warm", json={}, headers=ADMIN_HEADERS)
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["last_run"]["status"] == "ok"


def test_scheduler_liquidity_endpoint(client, monkeypatch):
    analytics = {"selection_size": 5, "top_allocations": [{"symbol": "BTCUSDT", "allocation_score": 0.42}]}

    async def fake_liquidity():
        return {
            "status": "ok",
            "run_id": "test-run",
            "selection_size": 5,
            "analytics": analytics,
        }

    monkeypatch.setattr("backend.routes.scheduler.run_liquidity_refresh_now", fake_liquidity)

    response = client.post("/scheduler/liquidity", headers=ADMIN_HEADERS)
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["liquidity"]["status"] == "ok"
    assert payload["liquidity"]["run_id"] == "test-run"
    assert payload["liquidity"]["analytics"] == analytics


def test_scheduler_execution_endpoint(client, monkeypatch):
    async def fake_execution():
        return {"status": "ok", "orders_planned": 1, "orders_submitted": 1}

    monkeypatch.setattr("backend.routes.scheduler.run_execution_cycle_now", fake_execution)

    response = client.post("/scheduler/execution", headers=ADMIN_HEADERS)
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["execution"]["status"] == "ok"
    assert payload["execution"]["orders_planned"] == 1
    assert payload["execution"]["orders_submitted"] == 1
