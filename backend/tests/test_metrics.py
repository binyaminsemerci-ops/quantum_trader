import pytest
from httpx import AsyncClient, ASGITransport

from backend.config.risk import load_risk_config  # type: ignore[import-error]
from backend.main import app
from backend.services.risk.risk_guard import RiskGuardService  # type: ignore[import-error]


@pytest.fixture(autouse=True)
async def _setup_risk_guard():
    app.state.risk_guard = RiskGuardService(load_risk_config())
    yield
    if hasattr(app.state, "risk_guard"):
        delattr(app.state, "risk_guard")


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_http_metrics():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.get("/health")
        response = await ac.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "qt_api_request_total" in body
    assert "qt_scheduler_runs_total" in body


@pytest.mark.asyncio
async def test_model_inference_metrics_recorded(monkeypatch):
    class DummyModel:
        telemetry_name = "test_model"

        def predict(self, arr):  # noqa: ARG002 - the array is unused in this stub
            return [1.23]

    monkeypatch.setattr("backend.routes.ai._load_model", lambda: DummyModel(), raising=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/ai/predict", json={"features": [1.0, 2.0]})
        assert response.status_code == 200

        metrics = await ac.get("/metrics")

    assert metrics.status_code == 200
    body = metrics.text
    assert "qt_model_inference_duration_seconds" in body
    assert 'model_name="test_model"' in body
