import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "trades.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def init_db():
    if os.path.exists(DB_PATH):
        print("🗑️ Sletter eksisterende database...")
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()
        cur.executescript(schema)

    conn.commit()
    conn.close()
    print("✅ Database initialisert!")


if __name__ == "__main__":
    init_db()
