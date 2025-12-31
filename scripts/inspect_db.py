import sqlite3
import pandas as pd

conn = sqlite3.connect("/app/data/quantum_trader.db")

# List tables
tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print(f"Tables: {tables}")

# Count rows
for table in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {count} rows")
    
    # Show sample columns
    if count > 0:
        sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 2", conn)
        print(f"    Columns: {list(sample.columns)}")

conn.close()
