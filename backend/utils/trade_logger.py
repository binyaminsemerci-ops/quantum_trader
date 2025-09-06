from backend.database import get_db, TradeLog
from datetime import datetime


def log_trade(trade: dict, status: str, reason: str = None):
    """
    Logger en trade til databasen.
    trade = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05, "price": 25000}
    """
    db = next(get_db())
    try:
        log = TradeLog(
            symbol=trade.get("symbol"),
            side=trade.get("side"),
            qty=trade.get("qty"),
            price=trade.get("price"),
            status=status,
            reason=reason,
            # 🔑 Bruk datetime-objekt, ikke string
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
