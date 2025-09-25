import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest.mark.asyncio
async def test_read_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Quantum Trader API is running"}


@pytest.mark.asyncio
async def test_stats_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/stats")
    assert response.status_code == 200
    assert "total_trades" in response.json()


@pytest.mark.asyncio
async def test_trades_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/trades")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_chart_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/chart")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_settings_roundtrip():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # The following values are test-only literals and are intentionally
        # present in the test. Mark them for detect-secrets allowlisting.
        payload = {"api_key": "dummy", "api_secret": "dummy"}  # pragma: allowlist secret
        post_resp = await ac.post("/settings", json=payload)
        assert post_resp.status_code == 200

        get_resp = await ac.get("/settings")
        assert get_resp.status_code == 200
        assert get_resp.json()["api_key"] == "dummy"

