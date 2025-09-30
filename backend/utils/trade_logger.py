from datetime import datetime, timezone
from typing import Optional, Any

from backend.database import session_scope, TradeLog


def log_trade(trade: dict[str, Any], status: str, reason: Optional[str] = None):
    """Persist a trade log entry using the configured database backend."""
    with session_scope() as session:
        symbol = str(trade.get("symbol") or "")
        side = str(trade.get("side") or "")
        try:
            qty = float(trade.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            price = float(trade.get("price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0

        log = TradeLog(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            status=status,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
        )
        session.add(log)
        session.flush()
        session.refresh(log)
        return log
