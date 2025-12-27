#!/usr/bin/env python3
"""Check latest filled orders."""
from backend.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text(
    "SELECT symbol, side, status, reason, created_at FROM execution_journal "
    "WHERE status = 'filled' ORDER BY created_at DESC LIMIT 5"
))
rows = result.fetchall()

print("\n[OK] SISTE 5 FILLED ORDERS:")
for row in rows:
    print(f"\n⏰ {row.created_at}")
    print(f"   Symbol: {row.symbol} {row.side}")
    print(f"   Status: {row.status}")
    print(f"   Reason: {row.reason}")

db.close()
