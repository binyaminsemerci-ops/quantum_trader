from backend.utils.trade_logger import log_trade
from backend.database import get_db, TradeLog, Base, engine


def setup_module(module):
    # Ensure table exists before test
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        db.query(TradeLog).delete()
        db.commit()
    except Exception:
        # If table doesn't exist or error occurs, continue
        db.rollback()


def test_trade_logs_endpoint():
    log_trade(
        {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05, "price": 25000},
        status="accepted",
    )
    db = next(get_db())
    logs = db.query(TradeLog).all()
    assert len(logs) == 1
    assert logs[0].symbol == "BTCUSDT"
