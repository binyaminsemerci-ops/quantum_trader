# backend/seed_trades.py
import sqlite3
import os
from backend.utils.trade_utils import select_execution_pair

# Finn riktig database path
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "trades.db")


def seed_trades():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("🗑️ Sletter eksisterende trades...")
    cur.execute("DELETE FROM trades")

    # Prefer futures (USDT-margined) for test trades, spot is rare in live systems
    dummy_trades = [
        (select_execution_pair('BTC', market='futures'), "buy", 20000, 20100, 0.01, 100),
        (select_execution_pair('BTC', market='futures'), "sell", 20100, 20000, 0.01, 100),
        (select_execution_pair('ETH', market='futures'), "buy", 1500, 1550, 0.5, 250),
        (select_execution_pair('ETH', market='futures'), "sell", 1550, 1500, 0.5, 250),
    ]

    cur.executemany(
        """
        INSERT INTO trades (symbol, side, entry_price, exit_price, qty, pnl)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        dummy_trades,
    )

    conn.commit()
    conn.close()
    print("✅ Trades testdata lagt inn!")


def seed_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("🗑️ Sletter eksisterende stats...")
    cur.execute("DELETE FROM stats")

    cur.execute(
        """
        INSERT INTO stats (balance, total_pnl, win_rate)
        VALUES (?, ?, ?)
    """,
        (10000, 450, 0.75),
    )

    conn.commit()
    conn.close()
    print("✅ Stats testdata lagt inn!")


if __name__ == "__main__":
    seed_trades()
    seed_stats()
