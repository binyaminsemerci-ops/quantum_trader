import json

import pytest

from backend.database import (
    Base,
    engine,
    SessionLocal,
    ModelTrainingRun,
    start_model_run,
    complete_model_run,
    fail_model_run,
)


@pytest.fixture(autouse=True)
def _create_tables():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_model_run_lifecycle(db_session):
    run = start_model_run(
        db_session,
        version="test-run-001",
        symbol_count=2,
        sample_count=100,
        feature_count=8,
        dataset_path="/tmp/raw.json",
    )
    assert run.status == "running"

    metrics = {"samples": 100, "features": 8, "rmse": 0.12}
    completed = complete_model_run(
        db_session,
        run.id,
        model_path="/tmp/model.pkl",
        scaler_path="/tmp/scaler.pkl",
        metrics=metrics,
    )
    assert completed.status == "completed"
    assert completed.model_path == "/tmp/model.pkl"
    assert completed.scaler_path == "/tmp/scaler.pkl"
    stored_metrics = json.loads(completed.metrics)
    assert stored_metrics["rmse"] == pytest.approx(0.12)

    latest = (
        db_session.query(ModelTrainingRun)
        .order_by(ModelTrainingRun.started_at.desc())
        .first()
    )
    assert latest is not None
    assert latest.id == completed.id


def test_fail_model_run(db_session):
    run = start_model_run(db_session, version="test-run-002")
    failed = fail_model_run(db_session, run.id, metrics={"error": "boom"})
    assert failed.status == "failed"
    assert json.loads(failed.metrics)["error"] == "boom"
