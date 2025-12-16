#!/usr/bin/env python3
"""
Check Last 2 Days Trading Results
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

db_path = Path(__file__).parent / "backend" / "data" / "trades.db"

if not db_path.exists():
    print(f"ERROR: Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("=" * 80)
print("TRADING RESULTS - LAST 2 DAYS")
print("=" * 80)

# Get date range
two_days_ago = (datetime.now() - timedelta(days=2)).isoformat()
print(f"\nAnalyzing trades since: {two_days_ago[:19]}")

# Overall statistics
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN exit_time IS NOT NULL THEN 1 ELSE 0 END) as closed,
        SUM(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END) as open
    FROM trades
""")
total, closed, open_pos = cursor.fetchone()
print(f"\n--- OVERALL DATABASE STATS ---")
print(f"Total trades ever: {total}")
print(f"Closed trades: {closed}")
print(f"Open positions: {open_pos}")

# Last 2 days closed trades
cursor.execute("""
    SELECT 
        COUNT(*) as count,
        SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
        ROUND(SUM(pnl_usd), 2) as total_pnl,
        ROUND(AVG(pnl_usd), 2) as avg_pnl,
        ROUND(AVG(pnl_percent), 2) as avg_pnl_pct
    FROM trades 
    WHERE exit_time >= ? AND exit_time IS NOT NULL
""", (two_days_ago,))

result = cursor.fetchone()
count, wins, losses, total_pnl, avg_pnl, avg_pnl_pct = result

print(f"\n--- LAST 2 DAYS PERFORMANCE ---")
print(f"Closed trades: {count}")

if count > 0:
    win_rate = (wins / count) * 100 if wins else 0
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"Losses: {losses} ({(losses/count)*100:.1f}%)")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Avg P&L per trade: ${avg_pnl:.2f}")
    print(f"Avg P&L %: {avg_pnl_pct:+.2f}%")
    
    # Get individual trades
    cursor.execute("""
        SELECT 
            symbol, side, 
            ROUND(entry_price, 4) as entry,
            ROUND(exit_price, 4) as exit,
            ROUND(pnl_usd, 2) as pnl_usd,
            ROUND(pnl_percent, 2) as pnl_pct,
            exit_reason,
            exit_time
        FROM trades
        WHERE exit_time >= ? AND exit_time IS NOT NULL
        ORDER BY exit_time DESC
        LIMIT 20
    """, (two_days_ago,))
    
    trades = cursor.fetchall()
    
    print(f"\n--- INDIVIDUAL TRADES (Last {min(len(trades), 20)}) ---")
    for i, trade in enumerate(trades, 1):
        symbol, side, entry, exit, pnl_usd, pnl_pct, reason, exit_time = trade
        status = "WIN" if pnl_usd > 0 else "LOSS"
        emoji = "✅" if pnl_usd > 0 else "❌"
        
        print(f"\n{i}. {symbol} {side} - {status} {emoji}")
        print(f"   Entry: ${entry:.4f} -> Exit: ${exit:.4f}")
        print(f"   P&L: ${pnl_usd:.2f} ({pnl_pct:+.2f}%)")
        print(f"   Exit: {reason}")
        print(f"   Time: {exit_time[:19]}")
else:
    print("No closed trades in last 2 days")

# Check open positions
cursor.execute("""
    SELECT 
        symbol, side,
        ROUND(entry_price, 4) as entry,
        quantity,
        entry_time
    FROM trades
    WHERE exit_time IS NULL
    ORDER BY entry_time DESC
""")

open_trades = cursor.fetchall()

print(f"\n--- OPEN POSITIONS ({len(open_trades)}) ---")
if open_trades:
    for i, trade in enumerate(open_trades, 1):
        symbol, side, entry, qty, entry_time = trade
        print(f"\n{i}. {symbol} {side}")
        print(f"   Entry: ${entry:.4f}")
        print(f"   Quantity: {qty}")
        print(f"   Opened: {entry_time[:19]}")
else:
    print("No open positions")

# Check if TP/SL is being used
cursor.execute("""
    SELECT exit_reason, COUNT(*) as count
    FROM trades
    WHERE exit_time IS NOT NULL
    GROUP BY exit_reason
    ORDER BY count DESC
    LIMIT 10
""")

print(f"\n--- EXIT REASONS DISTRIBUTION ---")
exit_reasons = cursor.fetchall()
for reason, count in exit_reasons:
    print(f"{reason}: {count} trades")

conn.close()

print("\n" + "=" * 80)
