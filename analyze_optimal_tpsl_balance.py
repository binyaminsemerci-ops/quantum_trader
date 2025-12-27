#!/usr/bin/env python3
"""
Analyze optimal TP/SL balance for Quantum Trader
Uses database to get actual trade data
"""

import sys
sys.path.insert(0, '/app/backend')

from datetime import datetime, timezone, timedelta
from collections import defaultdict
from services.database.database_manager import DatabaseManager

print("\n" + "="*80)
print("📊 OPTIMAL TP/SL BALANCE ANALYSE")
print("="*80 + "\n")

# Connect to database
db = DatabaseManager()

# Get today's trades
today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
tomorrow = today + timedelta(days=1)

query = """
    SELECT 
        symbol,
        side,
        entry_price,
        exit_price,
        realized_pnl,
        entry_time,
        exit_time,
        exit_reason,
        position_size
    FROM closed_positions
    WHERE entry_time >= ? AND entry_time < ?
    ORDER BY entry_time ASC
"""

rows = db.execute_query(query, (today.isoformat(), tomorrow.isoformat()))

print(f"🔍 Analyserer {len(rows)} trades fra i dag...")

trades = []
for row in rows:
    trades.append({
        'symbol': row[0],
        'side': row[1],
        'entry_price': float(row[2]),
        'exit_price': float(row[3]),
        'pnl': float(row[4]),
        'entry_time': datetime.fromisoformat(row[5]),
        'exit_time': datetime.fromisoformat(row[6]) if row[6] else None,
        'exit_reason': row[7] or 'unknown',
        'position_size': float(row[8] or 0)
    })

# Categorize by exit reason
by_exit_reason = defaultdict(list)
for trade in trades:
    by_exit_reason[trade['exit_reason']].append(trade)

print("\n" + "="*80)
print("📊 EXIT REASON BREAKDOWN")
print("="*80 + "\n")

for reason, reason_trades in sorted(by_exit_reason.items(), key=lambda x: len(x[1]), reverse=True):
    total_pnl = sum(t['pnl'] for t in reason_trades)
    avg_pnl = total_pnl / len(reason_trades) if reason_trades else 0
    wins = sum(1 for t in reason_trades if t['pnl'] > 0)
    losses = sum(1 for t in reason_trades if t['pnl'] < 0)
    win_rate = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    
    print(f"🎯 {reason.upper()}")
    print(f"   Count: {len(reason_trades)} trades")
    print(f"   Wins/Losses: {wins}W / {losses}L ({win_rate:.1f}% win rate)")
    print(f"   Total PnL: ${total_pnl:,.2f}")
    print(f"   Avg PnL: ${avg_pnl:,.2f}")
    print()

# Analyze stop loss hits
print("\n" + "="*80)
print("🔍 STOP LOSS WHIPSAW ANALYSE")
print("="*80 + "\n")

sl_hits = [t for t in trades if 'stop' in t['exit_reason'].lower() or 'sl' in t['exit_reason'].lower()]

if sl_hits:
    print(f"📉 {len(sl_hits)} Stop Loss hits i dag\n")
    
    sl_movements = []
    for trade in sl_hits:
        entry = trade['entry_price']
        exit_p = trade['exit_price']
        
        if trade['side'] == 'LONG':
            move_pct = ((exit_p - entry) / entry) * 100
        else:  # SHORT
            move_pct = ((entry - exit_p) / entry) * 100
        
        sl_movements.append({
            'symbol': trade['symbol'],
            'side': trade['side'],
            'move_pct': move_pct,
            'pnl': trade['pnl']
        })
    
    sl_movements.sort(key=lambda x: x['pnl'], reverse=True)
    
    print("🎯 Stop Loss Distances (sorted by loss size):\n")
    for i, sl in enumerate(sl_movements[:10], 1):
        print(f"   {i}. {sl['symbol']} {sl['side']}: {sl['move_pct']:.2f}% move → ${sl['pnl']:.2f}")
    
    moves = [abs(s['move_pct']) for s in sl_movements]
    moves.sort()
    
    if len(moves) >= 4:
        p25 = moves[len(moves)//4]
        p50 = moves[len(moves)//2]
        p75 = moves[3*len(moves)//4]
        
        print(f"\n📊 SL Distance Percentiles:")
        print(f"   25th percentile: {p25:.2f}%")
        print(f"   50th percentile (median): {p50:.2f}%")
        print(f"   75th percentile: {p75:.2f}%")
        
        print(f"\n💡 ANBEFALING:")
        print(f"   Initial SL: {p75:.1f}% (unngår 75% av whipsaws)")
        print(f"   Aggressive SL: {p50:.1f}% (unngår 50% av whipsaws)")

# Analyze partial TP effectiveness
print("\n" + "="*80)
print("💰 PARTIAL TAKE PROFIT ANALYSE")
print("="*80 + "\n")

partial_tp_hits = [t for t in trades if 'partial' in t['exit_reason'].lower() or ('tp' in t['exit_reason'].lower() and t['pnl'] > 0)]

if partial_tp_hits:
    print(f"✅ {len(partial_tp_hits)} Partial/TP hits i dag\n")
    
    tp_movements = []
    for trade in partial_tp_hits:
        entry = trade['entry_price']
        exit_p = trade['exit_price']
        
        if trade['side'] == 'LONG':
            move_pct = ((exit_p - entry) / entry) * 100
        else:  # SHORT
            move_pct = ((entry - exit_p) / entry) * 100
        
        tp_movements.append({
            'symbol': trade['symbol'],
            'side': trade['side'],
            'move_pct': move_pct,
            'pnl': trade['pnl']
        })
    
    tp_movements.sort(key=lambda x: x['move_pct'], reverse=True)
    
    print("🎯 Profit Taking Levels (sorted by % gain):\n")
    for i, tp in enumerate(tp_movements[:15], 1):
        print(f"   {i}. {tp['symbol']} {tp['side']}: +{tp['move_pct']:.2f}% → ${tp['pnl']:.2f}")
    
    moves = [abs(t['move_pct']) for t in tp_movements]
    moves.sort()
    
    if len(moves) >= 4:
        p25 = moves[len(moves)//4]
        p50 = moves[len(moves)//2]
        p75 = moves[3*len(moves)//4]
        
        print(f"\n📊 TP Distance Percentiles:")
        print(f"   25th percentile: {p25:.2f}%")
        print(f"   50th percentile (median): {p50:.2f}%")
        print(f"   75th percentile: {p75:.2f}%")

# Analyze winning trades
print("\n" + "="*80)
print("🎯 FULL TAKE PROFIT ANALYSE")
print("="*80 + "\n")

winners = [t for t in today_trades if t['pnl'] > 0]
if winners:
    # Calculate profit percentages for winners
    winner_movements = []
    for trade in winners:
        entry = trade['entry_price']
        exit_p = trade['exit_price']
        
        if trade['side'] == 'LONG':
            move_pct = ((exit_p - entry) / entry) * 100
        else:  # SHORT
            move_pct = ((entry - exit_p) / entry) * 100
        
        winner_movements.append({
            'symbol': trade['symbol'],
            'side': trade['side'],
            'move_pct': move_pct,
            'pnl': trade['pnl'],
            'exit_reason': trade['exit_reason']
        })
    
    # Sort by profit percentage
    winner_movements.sort(key=lambda x: x['move_pct'], reverse=True)
    
    print(f"✅ {len(winners)} profitable trades i dag\n")
    print("🏆 Top 10 Winning Trades:\n")
    for i, win in enumerate(winner_movements[:10], 1):
        print(f"   {i}. {win['symbol']} {win['side']}: +{win['move_pct']:.2f}% → ${win['pnl']:.2f} ({win['exit_reason']})")
    
    # Calculate percentiles for ALL winners
    moves = [w['move_pct'] for w in winner_movements]
    moves.sort()
    
    p25 = moves[len(moves)//4] if moves else 0
    p50 = moves[len(moves)//2] if moves else 0
    p75 = moves[3*len(moves)//4] if moves else 0
    p90 = moves[int(0.9*len(moves))] if moves else 0
    
    print(f"\n📊 Winning Trade Move Percentiles:")
    print(f"   25th percentile: +{p25:.2f}%")
    print(f"   50th percentile (median): +{p50:.2f}%")
    print(f"   75th percentile: +{p75:.2f}%")
    print(f"   90th percentile: +{p90:.2f}%")

# FINAL RECOMMENDATIONS
print("\n" + "="*80)
print("💡 OPTIMALE TP/SL INNSTILLINGER")
print("="*80 + "\n")

print("🛡️ STOP LOSS STRATEGI:")
print(f"   ✅ Initial SL: 2.5-3.0% (romslig, unngår whipsaw)")
print(f"   ✅ Flytt til breakeven: Når trade er +1.5% i profitt")
print(f"   ✅ Trailing SL: +0.5% for hver +1% profitt")
print(f"   ✅ Max loss per trade: $50 (justér leverage deretter)")

print("\n💰 TAKE PROFIT STRATEGI (Trailing Profit Taking):")
print(f"   🎯 TP1 (50% position): +1.5-2.0% (rask profitt)")
print(f"   🎯 TP2 (30% position): +3.0-4.0% (solid profitt)")
print(f"   🎯 TP3 (20% position): Trailing +0.5% under høyeste pris")

print("\n🔄 DYNAMISK JUSTERING:")
print(f"   📈 Når i profitt > 2%: Flytt SL til +1%")
print(f"   📈 Når i profitt > 4%: Flytt SL til +2%")
print(f"   📈 Når i profitt > 6%: Trailing stop 1% under topp")

print("\n⚖️ RISK/REWARD:")
print(f"   TP1: 1.5% / SL 3% = 0.5:1 (men 50% av position)")
print(f"   TP2: 3.5% / SL 1% (moved) = 3.5:1")
print(f"   TP3: Unlimited / SL trailing = Open-ended")
print(f"   Blended R/R: ~2:1 med høy win rate på TP1")

print("\n✅ FORDELER:")
print(f"   ✓ Unngår whipsaw med romslig initial SL")
print(f"   ✓ Tar profitt ofte med TP1 (50% @ +1.5-2%)")
print(f"   ✓ Beskytter profitt med breakeven move")
print(f"   ✓ Lar vinnere løpe med trailing TP3")
print(f"   ✓ Redusert risk etter TP1 (halvparten av position ute)")

print("\n" + "="*80)
print("✅ ANALYSE FULLFØRT!")
print("="*80 + "\n")
