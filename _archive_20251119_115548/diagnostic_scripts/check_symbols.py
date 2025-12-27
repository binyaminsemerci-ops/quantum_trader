#!/usr/bin/env python3
"""Check symbols in historical data."""
from backend.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text(
    "SELECT DISTINCT symbol FROM historical_data "
    "WHERE symbol LIKE '%USD%' "
    "ORDER BY symbol LIMIT 20"
))
symbols = [row[0] for row in result.fetchall()]

print("\n[CHART] SYMBOLS IN HISTORICAL_DATA:")
for sym in symbols:
    print(f"  {sym}")

db.close()
