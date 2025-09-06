import pytest
from backend.utils.trade_logger import log_trade
from backend.database import get_db, TradeLog

def setup_module(module):
    db = next(get_db())
    db.query(TradeLog).delete()
    db.commit()

def test_trade_logs_endpoint():
    log_trade({"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05, "price": 25000}, status="accepted")
    db = next(get_db())
    logs = db.query(TradeLog).all()
    assert len(logs) == 1
    assert logs[0].symbol == "BTCUSDT"
