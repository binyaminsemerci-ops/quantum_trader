from typing import Dict
from sqlalchemy import select

from backend.database import session_scope, Trade


def calculate_pnl() -> float:
    with session_scope() as session:
        rows = session.execute(select(Trade.side, Trade.qty, Trade.price)).all()
    pnl = 0.0
    for side, qty, price in rows:
        if side.upper() == "SELL":
            pnl += qty * price
        else:
            pnl -= qty * price
    return round(pnl, 2)


def calculate_pnl_per_symbol() -> Dict[str, float]:
    with session_scope() as session:
        rows = session.execute(select(Trade.symbol, Trade.side, Trade.qty, Trade.price)).all()
    pnl: Dict[str, float] = {}
    for symbol, side, qty, price in rows:
        delta = qty * price if side.upper() == "SELL" else -qty * price
        pnl[symbol] = round(pnl.get(symbol, 0.0) + delta, 2)
    return pnl
