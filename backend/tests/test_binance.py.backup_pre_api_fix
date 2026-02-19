import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest.mark.asyncio
async def test_get_server_time():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/binance/server-time")
    assert response.status_code == 200
    assert "serverTime" in response.json()


@pytest.mark.asyncio
async def test_spot_balance():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/binance/spot-balance")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_futures_balance():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/binance/futures-balance")
    assert response.status_code == 200
