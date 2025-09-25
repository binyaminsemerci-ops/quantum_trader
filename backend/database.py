"""Minimal database runtime shim for tests.

This file provides a simple sqlite3-backed session generator and a
lightweight TradeLog class so tests can import `backend.database` and
operate against a real sqlite DB. It's intentionally small and test-friendly.
"""
from typing import Iterator, Any
import sqlite3
import os
from dataclasses import dataclass
from datetime import datetime, timezone


DB_PATH = os.path.join(os.path.dirname(__file__), "test.db")


def _ensure_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            status TEXT,
            reason TEXT
        )
        """
    )
    conn.commit()
    conn.close()


@dataclass
class TradeLog:
    id: int | None
    timestamp: datetime | None
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    reason: str | None


class Session:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, obj: Any) -> None:
        # Expect TradeLog-like object
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO trade_logs (timestamp, symbol, side, qty, price, status, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                obj.timestamp.isoformat() if obj.timestamp else None,
                obj.symbol,
                obj.side,
                obj.qty,
                obj.price,
                obj.status,
                obj.reason,
            ),
        )
        self.conn.commit()
        obj.id = cur.lastrowid

    def commit(self) -> None:
        self.conn.commit()

    def refresh(self, obj: Any) -> None:
        # no-op for sqlite shim
        pass

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()

    # lightweight query helpers used by tests
    def query(self, model: Any):
        class Q:
            def __init__(self, conn):
                self.conn = conn

            def delete(self):
                cur = self.conn.cursor()
                cur.execute("DELETE FROM trade_logs")
                self.conn.commit()

            def all(self):
                cur = self.conn.cursor()
                cur.execute("SELECT id, timestamp, symbol, side, qty, price, status, reason FROM trade_logs ORDER BY id ASC")
                rows = cur.fetchall()
                out = []
                for r in rows:
                    out.append(
                        TradeLog(
                            id=r[0],
                            timestamp=datetime.fromisoformat(r[1]) if r[1] else None,
                            symbol=r[2],
                            side=r[3],
                            qty=r[4],
                            price=r[5],
                            status=r[6],
                            reason=r[7],
                        )
                    )
                return out

            def first(self):
                all_ = self.all()
                return all_[0] if all_ else None

        return Q(self.conn)


def get_db():
    """Return an iterator that yields a Session.

    Using a simple list iterator avoids generator-finalizer closing the
    connection prematurely when calling `next(get_db())` (a pattern used
    throughout the tests).
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    return iter([Session(conn)])
