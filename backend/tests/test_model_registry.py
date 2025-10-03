import json
from backend.database import SessionLocal, ModelRegistry, Base, engine
from ai_engine.train_and_save import train_and_save


def test_model_registry_insert(tmp_path, monkeypatch):
    # Use temp model dir
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Train (will insert into registry)
    summary = train_and_save(
        symbols=["BTCUSDT"],
        limit=50,
        model_dir=model_dir,
        backtest=False,
        write_report=False,
    )
    assert "metrics" in summary

    # Verify registry row
    with SessionLocal() as session:
        rows = session.query(ModelRegistry).order_by(ModelRegistry.id.desc()).all()
        assert rows, "Expected at least one model_registry row"
        last = rows[0]
        assert last.path.endswith("xgb_model.pkl") or last.path.endswith(
            "xgb_model.json"
        )
        assert last.version
        assert last.is_active == 0


def test_promote_latest(monkeypatch):
    with SessionLocal() as session:
        # Insert two dummy rows
        m1 = ModelRegistry(version="v1", path="/tmp/a")
        m2 = ModelRegistry(version="v2", path="/tmp/b")
        session.add_all([m1, m2])
        session.commit()

    # Promote second manually
    with SessionLocal() as session:
        # demote existing
        session.query(ModelRegistry).filter(ModelRegistry.is_active == 1).update(
            {"is_active": 0}
        )
        latest = session.query(ModelRegistry).order_by(ModelRegistry.id.desc()).first()
        latest.is_active = 1
        session.add(latest)
        session.commit()

    with SessionLocal() as session:
        active = (
            session.query(ModelRegistry).filter(ModelRegistry.is_active == 1).first()
        )
        assert active.version == "v2"
