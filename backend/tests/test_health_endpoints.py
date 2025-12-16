import pytest

from fastapi.testclient import TestClient

from backend.database import SessionLocal
from backend.services.execution.positions import PortfolioPositionService


@pytest.mark.usefixtures("client")
def test_health_scheduler_exposes_execution_metrics(client: TestClient) -> None:
    response = client.get("/health/scheduler")
    assert response.status_code == 200
    payload = response.json()
    assert "execution" in payload
    execution = payload["execution"]
    assert "gross_exposure" in execution
    assert "positions_synced" in execution
    assert isinstance(execution["positions_synced"], bool)


@pytest.mark.usefixtures("client")
def test_health_includes_risk_positions_snapshot(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    risk_snapshot = payload.get("risk")
    assert isinstance(risk_snapshot, dict)
    positions = risk_snapshot.get("positions")
    assert isinstance(positions, dict)
    assert "positions" in positions
    assert "total_notional" in positions
    assert "as_of" in positions


@pytest.mark.usefixtures("client")
def test_health_reflects_portfolio_positions(client: TestClient) -> None:
    with SessionLocal() as session:
        service = PortfolioPositionService(session)
        service.sync_from_holdings({"BTCUSDT": 0.5}, {"BTCUSDT": 20000.0})

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    risk_snapshot = payload["risk"]
    positions = risk_snapshot["positions"]
    symbols = [entry["symbol"] for entry in positions["positions"]]
    assert "BTCUSDT" in symbols
    assert positions["total_notional"] > 0
