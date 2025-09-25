from typing import Dict
from backend.database import get_db


def calculate_pnl() -> float:
    """Return total PnL from trades table (simple sum of pnl column)"""
    db = next(get_db())
    try:
        cursor = db.conn.cursor()
        try:
            cursor.execute("SELECT SUM(pnl) FROM trades")
            row = cursor.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            return 0.0
    finally:
        try:
            db.close()
        except Exception:
            pass


def calculate_pnl_per_symbol() -> Dict[str, float]:
    """Return per-symbol PnL mapping by aggregating trades table."""
    db = next(get_db())
    out = {}
    try:
        cursor = db.conn.cursor()
        try:
            cursor.execute("SELECT symbol, SUM(pnl) FROM trades GROUP BY symbol")
            rows = cursor.fetchall()
            for r in rows:
                out[r[0]] = float(r[1]) if r[1] is not None else 0.0
            return out
        except Exception:
            return {}
    finally:
        try:
            db.close()
        except Exception:
            pass
