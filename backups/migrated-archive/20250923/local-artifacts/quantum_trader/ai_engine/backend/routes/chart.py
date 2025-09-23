from fastapi import APIRouter
import sqlite3

router = APIRouter()


@router.get("/equity")
def get_equity_curve():
    conn = sqlite3.connect("backend/data/trades.db")
    df = conn.execute("SELECT timestamp, equity FROM trades").fetchall()
    conn.close()
    return [{"timestamp": ts, "equity": eq} for ts, eq in df]
