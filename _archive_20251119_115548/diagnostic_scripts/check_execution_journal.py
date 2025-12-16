"""Check recent execution journal entries"""
from backend.database import SessionLocal
from backend.models.liquidity import ExecutionJournal
from sqlalchemy import desc

db = SessionLocal()
recent = db.query(ExecutionJournal).order_by(desc(ExecutionJournal.created_at)).limit(10).all()

print('[CHART] RECENT EXECUTION JOURNAL (Last 10):')
print('=' * 100)

if not recent:
    print('No execution entries found')
else:
    for entry in recent:
        print(f'Symbol: {entry.symbol:10s} | Side: {entry.side:4s} | Qty: {entry.quantity:10.6f} | Status: {entry.status:8s}')
        print(f'Reason: {entry.reason}')
        if entry.error:
            print(f'Error: {entry.error}')
        print('-' * 100)
    
db.close()
