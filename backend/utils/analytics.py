import math
from typing import List, Tuple

from sqlalchemy import select

from backend.database import Trade, session_scope


def calculate_analytics():
    with session_scope() as session:
        rows: List[Tuple[str, float, float]] = list(
            session.execute(select(Trade.side, Trade.qty, Trade.price))
        )

    wins = 0
    losses = 0
    returns: List[float] = []

    for side, qty, price in rows:
        ret = qty * price if side.upper() == "SELL" else -qty * price
        returns.append(ret)
        if ret > 0:
            wins += 1
        else:
            losses += 1

    win_rate = (wins / max(1, (wins + losses))) * 100
    avg_return = sum(returns) / max(1, len(returns))
    std_dev = math.sqrt(
        sum((r - avg_return) ** 2 for r in returns) / max(1, len(returns))
    )
    sharpe = avg_return / std_dev if std_dev else 0

    return {
        "win_rate": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 2),
        "trades_count": len(returns),
    }
