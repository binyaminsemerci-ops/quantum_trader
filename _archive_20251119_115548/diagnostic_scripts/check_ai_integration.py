#!/usr/bin/env python3
"""Check AI model integration in latest liquidity run."""
import sys
sys.path.insert(0, '.')
from backend.database import SessionLocal
from backend.models.liquidity import PortfolioAllocation, LiquidityRun

db = SessionLocal()
run = db.query(LiquidityRun).order_by(LiquidityRun.id.desc()).first()
print(f"\n=== Latest Liquidity Run #{run.id} ===")
print(f"Universe: {run.universe_size}, Selection: {run.selection_size}")
print(f"Provider: {run.provider_primary}, Status: {run.status}")

allocs = db.query(PortfolioAllocation).filter_by(run_id=run.id).order_by(PortfolioAllocation.weight.desc()).all()
print(f"\n=== Portfolio Allocations (Top 10) ===")
for a in allocs[:10]:
    fields = a.__dict__
    print(f"{a.symbol:12} weight={a.weight:.4f} score={a.score:.2f} reason={fields.get('reason', 'N/A')[:50]}")

db.close()
