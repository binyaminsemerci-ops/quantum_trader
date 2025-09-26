from backend.database import get_db, TradeLog
from datetime import datetime
from typing import Optional, Any


def log_trade(trade: dict[str, Any], status: str, reason: Optional[str] = None):
    """
    Logger en trade til databasen.
    trade = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05, "price": 25000}
    """
    db = next(get_db())
    try:
        # Coerce and validate incoming values so MyPy sees the expected types
        raw_symbol = trade.get("symbol")
        raw_side = trade.get("side")
        raw_qty = trade.get("qty")
        raw_price = trade.get("price")

        symbol: str = str(raw_symbol or "")
        side: str = str(raw_side or "")
        try:
            qty: float = float(raw_qty or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            price: float = float(raw_price or 0.0)
        except (TypeError, ValueError):
            price = 0.0

        log = TradeLog(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
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
