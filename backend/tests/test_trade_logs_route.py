from datetime import datetime, timezone

from backend.database import SessionLocal, TradeLog
import backend.routes.trade_logs as trade_logs_route

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


def test_trade_logs_requires_admin_token(client):
    response = client.get("/trade_logs")
    assert response.status_code == 401


def test_trade_logs_returns_recent_entries(client):
    session = SessionLocal()
    try:
        session.add(
            TradeLog(
                symbol="BTCUSDT",
                side="BUY",
                qty=0.25,
                price=19500.0,
                status="FILLED",
                reason="rebalance",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        )
        session.add(
            TradeLog(
                symbol="ETHUSDT",
                side="SELL",
                qty=1.5,
                price=1250.0,
                status="FILLED",
                reason="risk_trim",
                timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
            )
        )
        session.commit()
    finally:
        session.close()

    verify_session = SessionLocal()
    try:
        assert verify_session.query(TradeLog).count() == 2
    finally:
        verify_session.close()

    direct_session = SessionLocal()
    try:
        rows = (
            direct_session.query(TradeLog)
            .order_by(TradeLog.id.desc())
            .limit(50)
            .with_entities(
                TradeLog.timestamp,
                TradeLog.symbol,
                TradeLog.side,
                TradeLog.qty,
                TradeLog.price,
                TradeLog.status,
                TradeLog.reason,
            )
            .all()
        )
        assert len(rows) == 2
    finally:
        direct_session.close()

    assert trade_logs_route.TradeLog is TradeLog
    response = client.get("/trade_logs", headers=ADMIN_HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert "logs" in payload
    assert len(payload["logs"]) == 2

    first_entry, second_entry = payload["logs"]
    assert first_entry["symbol"] == "ETHUSDT"
    assert second_entry["symbol"] == "BTCUSDT"
    assert first_entry["timestamp"].startswith("2025-01-02T00:00:00")
    assert second_entry["timestamp"].startswith("2025-01-01T00:00:00")
