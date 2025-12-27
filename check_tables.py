import sys
sys.path.insert(0, '/app')

from backend.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
tables = [row[0] for row in result.fetchall()]
print("Available tables:")
for table in tables:
    print(f"  - {table}")

# Check for trade history tables
for table in tables:
    if 'position' in table.lower() or 'trade' in table.lower() or 'order' in table.lower():
        print(f"\n{table} row count:")
        count_result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
        print(f"  {count_result.scalar()} rows")

db.close()
