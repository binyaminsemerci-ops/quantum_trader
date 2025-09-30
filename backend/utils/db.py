from typing import List, Dict
from sqlalchemy import select

from backend.database import session_scope, Trade


def get_all_trades() -> List[Dict[str, object]]:
    with session_scope() as session:
        rows = session.execute(
            select(Trade).order_by(Trade.timestamp.desc())
        ).scalars()
        return [
            {
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.qty,
                "price": trade.price,
                "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
            }
            for trade in rows
        ]
