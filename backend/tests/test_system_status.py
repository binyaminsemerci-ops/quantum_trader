import importlib
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.parametrize(
    "module_name",
    [
        "backend.main",
        "backend.simple_main",
        "ai_engine.train_and_save",
    ],
)
def test_import_modules(module_name):
    importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_system_status_main():
    from backend.main import app as main_app

    transport = ASGITransport(app=main_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/v1/system/status")
    assert r.status_code == 200
    js = r.json()
    assert "uptime_seconds" in js and js["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_system_status_simple():
    from backend.simple_main import app as simple_app

    transport = ASGITransport(app=simple_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/v1/system/status")
    assert r.status_code == 200
    js = r.json()
    assert js.get("service") in ("quantum_trader_simple", "quantum_trader_core")
