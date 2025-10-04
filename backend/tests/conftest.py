import os
import shutil
import pytest

from fastapi.testclient import TestClient


@pytest.fixture
def test_db_file(tmp_path):
    # Create a temporary directory for the test DB per-test
    tmpdir = tmp_path / "test_db"
    tmpdir.mkdir()
    db_path = tmpdir / "test_trades.db"
    db_url = f"sqlite:///{db_path}"

    # Export env var so backend.database picks it up
    os.environ["QUANTUM_TRADER_DATABASE_URL"] = db_url

    yield str(db_path)

    # Cleanup
    try:
        shutil.rmtree(str(tmpdir))
    except Exception as e:
        import logging

        logging.getLogger(__name__).debug("failed to remove tmp test dir: %s", e)


@pytest.fixture
def client(test_db_file):
    # Import here so that backend.database reads the env var we set above
    from backend.main import app
    from backend.database import Base, engine

    # Create all tables for the test
    Base.metadata.create_all(bind=engine)

    with TestClient(app) as c:
        yield c
        # Drop all tables after test for clean DB
        Base.metadata.drop_all(bind=engine)
