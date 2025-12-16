import sqlite3

databases = [
    '/app/database/trades.db',
    '/app/backend/data/trades.db', 
    '/app/backend/data/trading.db',
    '/app/backend/quantum_trader.db',
    '/app/backend/trades.db'
]

for db_path in databases:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n{db_path}: {len(tables)} tables")
        
        if tables:
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"  - {table[0]}: {count} rows")
        
        conn.close()
    except Exception as e:
        print(f"\n{db_path}: ERROR - {e}")
