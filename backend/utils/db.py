import sqlite3
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "trades.db")
DB_FILE = os.path.abspath(DB_FILE)


def get_connection():
    """Åpner SQLite-tilkobling."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_trades():
    """Hent alle trades fra databasen."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, symbol, side, quantity, price, timestamp FROM trades ORDER BY timestamp DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
