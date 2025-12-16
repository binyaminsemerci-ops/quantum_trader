import sqlite3

conn = sqlite3.connect('/app/backend/data/trading.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Get sample of latest trades
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%position%'")
print("Position tables:", [t[0] for t in cursor.fetchall()])

conn.close()
