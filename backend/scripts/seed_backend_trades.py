"""
Seed a few TradeLog rows into backend/data/trades.db used by SQLAlchemy.

Usage (PowerShell):
  Set-Location C:\\quantum_trader
  python backend/scripts/seed_backend_trades.py
"""

import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from datetime import datetime, timedelta
from random import randint

try:
    from backend.database import SessionLocal, TradeLog
except Exception:
    from database import SessionLocal, TradeLog  # type: ignore


def main() -> None:
    sess = SessionLocal()
    try:
        # Insert a few recent trades across BTC and ETH
        base = datetime.utcnow()
        rows = []
        for i in range(6):
            ts = base - timedelta(minutes=5 * i)
            rows.append(
                TradeLog(
                    symbol="BTCUSDT",
                    side="BUY" if i % 2 == 0 else "SELL",
                    qty=0.01,
                    price=43000 + i * 25,
                    status="filled",
                    reason="seed",
                    timestamp=ts,
                )
            )
        for i in range(4):
            ts = base - timedelta(minutes=7 * i)
            rows.append(
                TradeLog(
                    symbol="ETHUSDT",
                    side="BUY" if i % 2 == 1 else "SELL",
                    qty=0.05,
                    price=2800 + i * 10,
                    status="filled",
                    reason="seed",
                    timestamp=ts,
                )
            )
        for r in rows:
            sess.add(r)
        sess.commit()
        print(f"Seeded {len(rows)} trade_logs into backend/data/trades.db")
    finally:
        sess.close()


if __name__ == "__main__":
    main()
