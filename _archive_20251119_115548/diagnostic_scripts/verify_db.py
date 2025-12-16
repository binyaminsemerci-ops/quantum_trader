import sqlite3

# Verifiser database tabeller
conn = sqlite3.connect('backend/quantum_trader.db')

# List alle tabeller
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

print("Available tables:", tables)
print("liquidity_runs exists:", "liquidity_runs" in tables)

# Test liquidity_runs tabell
if "liquidity_runs" in tables:
    cursor.execute("SELECT COUNT(*) FROM liquidity_runs")
    count = cursor.fetchone()[0]
    print(f"liquidity_runs has {count} records")

conn.close()
print("Database verification complete!")