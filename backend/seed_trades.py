"""Seed demo trades/stats/equity data using SQLAlchemy."""

from datetime import datetime, timedelta, timezone
from typing import Iterable

from backend.database import session_scope, Trade, TradeLog, EquityPoint


def _clear_tables():
    with session_scope() as session:
        session.query(Trade).delete()
        session.query(TradeLog).delete()
        session.query(EquityPoint).delete()


def seed_trades() -> None:
    demo_trades: Iterable[Trade] = [
        Trade(symbol="BTCUSDT", side="BUY", qty=0.01, price=20_000),
        Trade(symbol="BTCUSDT", side="SELL", qty=0.01, price=20_300),
        Trade(symbol="ETHUSDT", side="BUY", qty=0.5, price=1_500),
        Trade(symbol="ETHUSDT", side="SELL", qty=0.5, price=1_620),
    ]
    with session_scope() as session:
        for trade in demo_trades:
            session.add(trade)
            session.flush()
            session.add(
                TradeLog(
                    symbol=trade.symbol,
                    side=trade.side,
                    qty=trade.qty,
                    price=trade.price,
                    status="demo",
                    reason="seed",
                    timestamp=trade.timestamp,
                )
            )


def seed_equity_curve(days: int = 30) -> None:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    equity = 10_000.0
    with session_scope() as session:
        for i in range(days):
            equity *= 1 + 0.002 * ((-1) ** (i % 5))
            session.add(
                EquityPoint(
                    date=(start + timedelta(days=i)),
                    equity=round(equity, 2),
                )
            )


def main() -> None:
    _clear_tables()
    seed_trades()
    seed_equity_curve()
    print("Seeded demo trades and equity curve")


if __name__ == "__main__":
    main()
