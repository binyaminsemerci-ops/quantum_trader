from collections import defaultdict
from typing import Dict, Iterable, Tuple

from sqlalchemy import select

from backend.database import TradeLog, get_db


def _fetch_trades(columns: Iterable) -> Iterable[Tuple]:
    """Yield selected trade log columns with automatic session management."""

    db_iter = get_db()
    session = next(db_iter)
    try:
        statement = select(*columns)
        for row in session.execute(statement):
            yield tuple(row)
    finally:
        db_iter.close()


def calculate_pnl() -> float:
    """Compute aggregate PnL based on recorded trade logs."""

    pnl = 0.0
    for side, qty, price in _fetch_trades((TradeLog.side, TradeLog.qty, TradeLog.price)):
        if not side or qty is None or price is None:
            continue
        if side.upper() == "BUY":
            pnl -= qty * price
        elif side.upper() == "SELL":
            pnl += qty * price

    return round(pnl, 2)


def calculate_pnl_per_symbol() -> Dict[str, float]:
    """Return per-symbol realised PnL derived from trade logs."""

    pnl_by_symbol: Dict[str, float] = defaultdict(float)
    for symbol, side, qty, price in _fetch_trades(
        (TradeLog.symbol, TradeLog.side, TradeLog.qty, TradeLog.price)
    ):
        if not symbol or not side or qty is None or price is None:
            continue
        if side.upper() == "BUY":
            pnl_by_symbol[symbol] -= qty * price
        elif side.upper() == "SELL":
            pnl_by_symbol[symbol] += qty * price

    return {sym: round(val, 2) for sym, val in pnl_by_symbol.items()}
