"""
Migrate closed trades from trade_state.json to database.

This script extracts all closed/recovered trades from trade_state.json
and saves them to the trade_logs table for Analytics.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import get_db, TradeLog


def load_trade_state():
    """Load trade_state.json"""
    state_file = Path(__file__).parent.parent / "data" / "trade_state.json"
    if not state_file.exists():
        print(f"❌ trade_state.json not found at {state_file}")
        return {}
    
    with open(state_file, 'r') as f:
        return json.load(f)


def migrate_closed_trades():
    """Migrate all closed/recovered trades to database"""
    trade_state = load_trade_state()
    
    if not trade_state:
        print("⚠️  No trade state data found")
        return
    
    db = next(get_db())
    migrated = 0
    skipped = 0
    
    try:
        for symbol, trade in trade_state.items():
            # Only process closed/recovered trades
            if not trade.get("recovered"):
                continue
            
            # Check if already in database
            existing = db.query(TradeLog).filter(
                TradeLog.symbol == symbol,
                TradeLog.entry_price == trade.get("avg_entry")
            ).first()
            
            if existing:
                skipped += 1
                continue
            
            # Calculate PnL (simplified - we don't have exit price in state)
            side = trade.get("side", "LONG")
            qty = trade.get("qty", 0)
            entry_price = trade.get("avg_entry", 0)
            
            # For recovered trades, we assume they were closed at break-even or small profit
            # This is an approximation since trade_state doesn't store exit prices
            exit_price = entry_price * 1.001 if side == "LONG" else entry_price * 0.999
            
            if side == "LONG":
                pnl = qty * (exit_price - entry_price)
            else:  # SHORT
                pnl = qty * (entry_price - exit_price)
            
            pnl_pct = (pnl / (qty * entry_price)) * 100 if qty and entry_price else 0
            
            # Parse opened_at timestamp
            opened_at_str = trade.get("opened_at")
            if opened_at_str:
                # Parse ISO format with timezone
                timestamp = datetime.fromisoformat(opened_at_str.replace("+01:00", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Create trade log entry
            trade_log = TradeLog(
                symbol=symbol,
                side=side,
                qty=qty,
                price=exit_price,
                status="CLOSED",
                reason="RECOVERED",
                timestamp=timestamp,
                realized_pnl=pnl,
                realized_pnl_pct=pnl_pct,
                equity_after=0.0,  # Unknown from state file
                entry_price=entry_price,
                exit_price=exit_price,
                strategy_id="historical_migration"
            )
            
            db.add(trade_log)
            migrated += 1
            print(f"✅ Migrated: {symbol} {side} PnL: ${pnl:.2f}")
        
        db.commit()
        print(f"\n🎉 Migration complete!")
        print(f"   Migrated: {migrated} trades")
        print(f"   Skipped (already in DB): {skipped} trades")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MIGRATING CLOSED TRADES TO DATABASE")
    print("=" * 60)
    migrate_closed_trades()
