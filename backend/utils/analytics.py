import math
from typing import Iterable, Tuple

from sqlalchemy import select

from backend.database import TradeLog, get_db


def _iter_trade_returns() -> Iterable[Tuple[str, float, float]]:
    """Yield side, quantity, and price for recorded trades."""

    db_iter = get_db()
    session = next(db_iter)
    try:
        statement = select(TradeLog.side, TradeLog.qty, TradeLog.price)
        for row in session.execute(statement):
            yield tuple(row)
    finally:
        db_iter.close()


def calculate_analytics() -> dict:
    trades = list(_iter_trade_returns())

    wins = 0
    losses = 0
    returns: list[float] = []

    for side, qty, price in trades:
        if not side or qty is None or price is None:
            continue

        realised = qty * price if side.upper() == "SELL" else -qty * price
        returns.append(realised)
        if realised > 0:
            wins += 1
        else:
            losses += 1

    win_rate = (wins / max(1, wins + losses)) * 100
    avg_return = sum(returns) / max(1, len(returns))
    variance = sum((r - avg_return) ** 2 for r in returns) / max(1, len(returns))
    std_dev = math.sqrt(variance)
    sharpe = avg_return / std_dev if std_dev else 0.0

    return {
        "win_rate": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 2),
        "trades_count": len(returns),
    }
