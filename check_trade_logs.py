import sys
sys.path.insert(0, '/app')

from backend.database import SessionLocal
from sqlalchemy import text
import pandas as pd

db = SessionLocal()

# Check trade_logs structure and data
print("trade_logs structure:")
result = db.execute(text("PRAGMA table_info(trade_logs)"))
for row in result.fetchall():
    print(f"  {row[1]} ({row[2]})")

print("\ntrade_logs sample data (first 5 rows):")
result = db.execute(text("SELECT * FROM trade_logs LIMIT 5"))
columns = [col for col in result.keys()]
print("Columns:", columns)

for row in result.fetchall():
    print(dict(zip(columns, row)))

print("\ntrade_logs with close data:")
result = db.execute(text("SELECT COUNT(*) FROM trade_logs WHERE exit_price IS NOT NULL OR close_time IS NOT NULL"))
closed_count = result.scalar()
print(f"  Rows with exit/close data: {closed_count}")

print("\ntrade_logs date range:")
result = db.execute(text("SELECT MIN(timestamp), MAX(timestamp) FROM trade_logs"))
date_range = result.fetchone()
print(f"  From {date_range[0]} to {date_range[1]}")

db.close()
