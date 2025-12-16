#!/usr/bin/env python3
"""
Fix backend database by applying schema to the correct database file.
"""
import sqlite3
import os

# Backend uses backend/data/trades.db
backend_db_path = 'c:/quantum_trader/backend/data/trades.db'
schema_path = 'c:/quantum_trader/database/schema.sql'

print(f"Applying schema to: {backend_db_path}")

# Connect to the correct database
conn = sqlite3.connect(backend_db_path)

# Read and execute schema
with open(schema_path, 'r') as f:
    schema_sql = f.read()

conn.executescript(schema_sql)
conn.commit()

# Verify tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print('Available tables:', tables)
print('liquidity_runs exists:', 'liquidity_runs' in tables)

# Check liquidity_runs table structure
if 'liquidity_runs' in tables:
    cursor.execute('PRAGMA table_info(liquidity_runs);')
    columns = cursor.fetchall()
    print('liquidity_runs columns:', [col[1] for col in columns])

conn.close()
print('Schema applied to backend/data/trades.db successfully!')