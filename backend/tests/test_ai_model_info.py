import pytest
from httpx import AsyncClient, ASGITransport

from backend.database import (
    Base,
    engine,
    SessionLocal,
    start_model_run,
    complete_model_run,
)
from backend.main import app


@pytest.fixture(autouse=True)
def _prepare_database(test_db_file):
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.mark.asyncio
async def test_model_info_endpoint_returns_latest_run():
    session = SessionLocal()
    try:
        run = start_model_run(
            session,
            version="20250101T000000Z",
            symbol_count=1,
            sample_count=10,
            feature_count=4,
        )
        complete_model_run(
            session,
            run.id,
            model_path="/tmp/old_model.pkl",
            scaler_path="/tmp/old_scaler.pkl",
            metrics={"samples": 10, "features": 4},
        )

        newer = start_model_run(
            session,
            version="20250202T000000Z",
            symbol_count=2,
            sample_count=50,
            feature_count=6,
        )
        complete_model_run(
            session,
            newer.id,
            model_path="/tmp/new_model.pkl",
            scaler_path="/tmp/new_scaler.pkl",
            metrics={"samples": 50, "features": 6},
        )
    finally:
        session.close()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ai/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "20250202T000000Z"
    assert payload["samples"] == 50
    assert payload["features"] == 6
    assert payload["model_path"] == "/tmp/new_model.pkl"
    assert payload["status"] == "completed"