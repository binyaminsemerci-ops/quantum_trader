from backend.database import get_db, TradeLog
import datetime
from typing import Optional




def log_trade(trade: dict, status: str, reason: Optional[str] = None):
    """
    Logger en trade til databasen.
    trade = {"symbol": "BTCUSDC", "side": "BUY", "qty": 0.05, "price": 25000}
    """
    db = next(get_db())
    try:
        log = TradeLog(
            # Use safe defaults/casts so mypy can verify types
            symbol=str(trade.get("symbol") or ""),
            side=str(trade.get("side") or ""),
            qty=float(trade.get("qty") or 0.0),
            price=float(trade.get("price") or 0.0),
            status=status,
            reason=reason,
            # 🔑 Bruk datetime-objekt, ikke string
            timestamp=datetime.datetime.now(datetime.timezone.utc),
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
