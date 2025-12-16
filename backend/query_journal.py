#!/usr/bin/env python3
"""Query execution journal for a given run_id."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.database import SessionLocal
from backend.models.liquidity import ExecutionJournal

run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 94
db = SessionLocal()
journals = db.query(ExecutionJournal).filter_by(run_id=run_id).all()
for j in journals:
    reason = j.reason[:80] if j.reason and len(j.reason) > 80 else j.reason
    error = j.error[:150] if j.error and len(j.error) > 150 else j.error
    print(f"{j.symbol} {j.status} reason={reason} error={error}")
db.close()
