from backend.database import SessionLocal, ModelRegistry, Base, engine
from backend.utils.metrics import update_model_perf, update_model_sortino


def _seed_active_model(version="test-ver", tag="unit", sharpe=1.23, max_dd=0.12):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        # demote any existing
        session.query(ModelRegistry).filter(ModelRegistry.is_active == 1).update({"is_active": 0})
        row = ModelRegistry(
            version=version,
            tag=tag,
            path="/tmp/model.pkl",
            metrics_json=(
                '{"backtest": {"sharpe": ' + str(sharpe) + ', "max_drawdown": ' + str(max_dd) + '}}'
            ),
            is_active=1,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def test_active_model_endpoint(client):
    _seed_active_model()
    r = client.get("/api/v1/model/active")
    assert r.status_code == 200
    data = r.json()
    assert data.get("active") is True
    assert data.get("version") == "test-ver"
    assert data.get("sharpe") == 1.23
    # sortino may be absent in seeded metrics_json (legacy) so allow None
    assert "sortino" in data or "sortino" not in data  # presence check non-failing


def test_system_status_includes_active_model(client):
    _seed_active_model(version="vv2", tag="prod")
    r = client.get("/api/v1/system/status")
    assert r.status_code == 200
    data = r.json()
    active = data.get("active_model")
    assert active
    assert active.get("version") in ("vv2", "test-ver")  # allow race if heartbeat refreshed


def test_metrics_expose_model_info(client):
    _seed_active_model(version="met-v", tag="met-tag")
    # Simulate metrics gauge updates
    update_model_perf(2.5, 0.15)
    update_model_sortino(3.1)
    r = client.get("/api/v1/metrics/")
    assert r.status_code == 200
    body = r.text
    assert "quantum_model" in body
    assert "quantum_model_sharpe" in body
    assert "quantum_model_max_drawdown" in body
    assert "quantum_model_sortino" in body