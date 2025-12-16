#!/usr/bin/env python3
"""Check database structure"""
import sqlite3

conn = sqlite3.connect('/app/backend/data/trades.db')
cursor = conn.cursor()

# List all tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print("Tables found:", tables)
print()

# Get column info for each table
for table in tables:
    table_name = table[0]
    print(f"=== Table: {table_name} ===")
    cursor.execute(f'PRAGMA table_info({table_name})')
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    print()

conn.close()
