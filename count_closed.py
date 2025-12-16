import sys
sys.path.insert(0, '/app')

from backend.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()

result = db.execute(text("SELECT COUNT(*) FROM trade_logs WHERE status='CLOSED' AND exit_price IS NOT NULL"))
closed_count = result.scalar()
print(f"Closed trades: {closed_count}")

result = db.execute(text("SELECT MIN(timestamp), MAX(timestamp) FROM trade_logs"))
date_range = result.fetchone()
print(f"Date range: {date_range[0]} to {date_range[1]}")

db.close()
