import json
from pathlib import Path

import pytest

from starlette.websockets import WebSocketDisconnect

from backend.utils.admin_events import AdminEvent, record_admin_event  # type: ignore[import-error]
from backend.utils.telemetry import (  # type: ignore[import-error]
    ADMIN_EVENTS_TOTAL,
    reset_admin_event_metrics,
)


ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


@pytest.fixture(autouse=True)
def _reset_admin_metrics():
    reset_admin_event_metrics()
    yield
    reset_admin_event_metrics()


def _read_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def test_missing_token_is_logged(client, audit_log_file):
    response = client.get("/risk")
    assert response.status_code == 401

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.RISK_AUTH_MISSING.value for entry in entries)


def test_kill_switch_action_logged(client, audit_log_file):
    response = client.post(
        "/risk/kill-switch",
        json={"enabled": True},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.RISK_KILL_SWITCH.value for entry in entries)

    kill_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.RISK_KILL_SWITCH.value
    )
    assert kill_entry["success"] is True
    assert kill_entry["details"]["override"] is True


def test_settings_update_action_logged(client, audit_log_file):
    payload = {"risk_percentage": 2.5, "trading_enabled": True}

    response = client.post("/settings", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.SETTINGS_UPDATE.value
    )
    assert audit_entry["success"] is True
    updated = audit_entry["details"]["updated_fields"]
    assert set(updated) == {"risk_percentage", "trading_enabled"}


def test_settings_update_validation_logged(client, audit_log_file):
    payload = {"risk_percentage": 25}

    response = client.post("/settings", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 422

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.SETTINGS_UPDATE.value
    )
    assert audit_entry["success"] is False
    assert audit_entry["details"]["status_code"] == 422


def test_settings_missing_token_logged(client, audit_log_file):
    response = client.post("/settings", json={"trading_enabled": True})
    assert response.status_code == 401

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.ADMIN_AUTH_MISSING.value for entry in entries)


def test_settings_invalid_token_logged(client, audit_log_file):
    response = client.post(
        "/settings",
        json={"trading_enabled": True},
        headers={"X-Admin-Token": "wrong-token"},
    )
    assert response.status_code == 403

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.ADMIN_AUTH_INVALID.value for entry in entries)


def test_settings_get_action_logged(client, audit_log_file):
    response = client.get("/settings", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.SETTINGS_READ.value
    )
    assert audit_entry["success"] is True
    assert audit_entry["details"]["secure"] is False


def test_settings_get_secure_action_logged(client, audit_log_file):
    response = client.get("/settings?secure=true", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.SETTINGS_READ.value
        and entry["details"].get("secure") is True
    )
    assert audit_entry["success"] is True


def test_trades_create_action_logged(client, audit_log_file):
    payload = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.1, "price": 100.0}

    response = client.post("/trades", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 201

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.TRADES_CREATE.value
    )
    assert audit_entry["success"] is True
    details = audit_entry["details"]
    assert details["symbol"] == "BTCUSDT"
    assert details["notional"] == pytest.approx(payload["qty"] * payload["price"])


def test_trades_create_validation_logged(client, audit_log_file):
    payload = {"symbol": "INVALID", "side": "BUY", "qty": 0.1, "price": 100.0}

    response = client.post("/trades", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 403

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.TRADES_CREATE.value
    )
    assert audit_entry["success"] is False
    details = audit_entry["details"]
    assert details["status_code"] == 403
    assert details["detail"] == {"error": "RiskCheckFailed", "reason": "symbol_not_allowed"}


def test_trades_get_action_logged(client, audit_log_file):
    response = client.get("/trades", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.TRADES_READ.value
    )
    assert audit_entry["success"] is True
    assert "result_count" in audit_entry["details"]


def test_trades_recent_action_logged(client, audit_log_file):
    response = client.get("/trades/recent", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.TRADES_RECENT.value
    )
    assert audit_entry["success"] is True
    assert audit_entry["details"]["result_count"] == len(response.json())


def test_trade_logs_action_logged(client, audit_log_file):
    response = client.get("/trade_logs", headers=ADMIN_HEADERS)
    # Route returns 200 even if dataset empty; ensure JSON schema with list
    assert response.status_code == 200
    logs_payload = response.json()
    assert isinstance(logs_payload, dict) and "logs" in logs_payload

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.TRADE_LOGS_READ.value
    )
    assert audit_entry["success"] is True
    assert audit_entry["details"]["limit"] == 50


def test_dashboard_stream_requires_token(client, audit_log_file):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/dashboard"):
            pass

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.ADMIN_AUTH_MISSING.value for entry in entries)


def test_dashboard_stream_logged(client, audit_log_file):
    with client.websocket_connect("/ws/dashboard", headers=ADMIN_HEADERS) as websocket:
        websocket.receive_json()

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.DASHBOARD_STREAM.value
    )
    assert audit_entry["success"] is True
    assert audit_entry["details"].get("status") == "connected"


def test_record_admin_event_updates_prometheus_counter(audit_log_file):
    record_admin_event(AdminEvent.TRADES_CREATE, success=True, details={"symbol": "BTCUSDT"})
    metric = ADMIN_EVENTS_TOTAL.labels(
        event="trades.create", category="trade", severity="high", success="true"
    )
    assert metric._value.get() == 1.0

    record_admin_event(AdminEvent.TRADES_CREATE, success=False, details={"error": "boom"})
    failure_metric = ADMIN_EVENTS_TOTAL.labels(
        event="trades.create", category="trade", severity="high", success="false"
    )
    assert failure_metric._value.get() == 1.0


def test_ai_tasks_require_admin_token(client, audit_log_file):
    response = client.get("/ai/tasks")
    assert response.status_code == 401

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.ADMIN_AUTH_MISSING.value for entry in entries)


def test_ai_tasks_list_logged(client, audit_log_file):
    response = client.get("/ai/tasks", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.AI_TASK_LIST.value
    )
    assert audit_entry["success"] is True
    assert audit_entry["details"]["limit"] == 50


def test_ai_task_detail_requires_admin_token(client, audit_log_file):
    response = client.get("/ai/tasks/1")
    assert response.status_code == 401

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.ADMIN_AUTH_MISSING.value for entry in entries)


def test_ai_task_detail_not_found_logged(client, audit_log_file):
    response = client.get("/ai/tasks/999", headers=ADMIN_HEADERS)
    assert response.status_code == 404

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.AI_TASK_DETAIL.value
    )
    assert audit_entry["success"] is False
    assert audit_entry["details"]["status_code"] == 404


def test_ai_status_requires_admin_token(client, audit_log_file):
    response = client.get("/ai/status")
    assert response.status_code == 401

    entries = _read_entries(audit_log_file)
    assert any(entry["event"] == AdminEvent.ADMIN_AUTH_MISSING.value for entry in entries)


def test_ai_status_logged(client, audit_log_file, monkeypatch):
    class _DummyAgent:
        def __init__(self) -> None:
            self.model = object()
            self.scaler = object()

        def get_metadata(self):
            return {"version": "test"}

    monkeypatch.setattr("backend.routes.ai.make_default_agent", lambda: _DummyAgent())

    response = client.get("/ai/status", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    entries = _read_entries(audit_log_file)
    audit_entry = next(
        entry for entry in entries if entry["event"] == AdminEvent.AI_STATUS.value
    )
    assert audit_entry["success"] is True
    assert audit_entry["details"]["metadata_available"] is True