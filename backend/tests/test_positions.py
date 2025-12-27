from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models.positions import PortfolioPosition
from backend.services.execution.positions import PortfolioPositionService


def _make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_sync_from_holdings_persists_positions():
    session = _make_session()
    service = PortfolioPositionService(session)

    snapshot = service.sync_from_holdings({"BTCUSDT": 0.5}, {"BTCUSDT": 20000.0})
    assert snapshot["total_notional"] == 10000.0
    positions = session.query(PortfolioPosition).all()
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"
    assert positions[0].quantity == 0.5

    snapshot = service.sync_from_holdings({"ETHUSDT": 2.0}, {"ETHUSDT": 1500.0})
    positions = session.query(PortfolioPosition).all()
    assert len(positions) == 1
    assert positions[0].symbol == "ETHUSDT"
    assert snapshot["total_notional"] == 3000.0

    session.close()
