import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest.mark.asyncio
async def test_get_settings():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/settings")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_settings():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"api_key": "dummy_key", "api_secret": "dummy_secret"}
        response = await ac.post("/settings", json=payload)
    assert response.status_code == 200
