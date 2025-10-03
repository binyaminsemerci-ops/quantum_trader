import sqlite3

DB_PATH = "backend/data/trades.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        side TEXT,
        qty REAL,
        price REAL,
        pnl REAL,
        equity REAL,
        timestamp TEXT
    )"""
    )
    conn.commit()
    conn.close()


def log_trade(symbol, side, qty, price, pnl, equity, timestamp):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trades (symbol, side, qty, price, pnl, equity, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (symbol, side, qty, price, pnl, equity, timestamp),
    )
    conn.commit()
    conn.close()


def get_trades():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_balance_and_pnl():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(pnl), MAX(equity) FROM trades")
    row = cursor.fetchone()
    conn.close()
    return {"total_pnl": row[0] or 0, "max_equity": row[1] or 0}


init_db()
