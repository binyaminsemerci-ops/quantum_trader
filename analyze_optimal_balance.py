#!/usr/bin/env python3
"""
Analyze optimal TP/SL balance using direct SQL
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from collections import defaultdict

print("\n" + "="*80)
print("📊 OPTIMAL TP/SL BALANCE ANALYSE")
print("="*80 + "\n")

# Connect to database
conn = sqlite3.connect('/app/backend/data/trading.db')
cursor = conn.cursor()

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
        exit_reason
    FROM closed_positions
    WHERE entry_time >= ? AND entry_time < ?
    ORDER BY entry_time ASC
"""

cursor.execute(query, (today.isoformat(), tomorrow.isoformat()))
rows = cursor.fetchall()

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
        'exit_reason': row[7] or 'unknown'
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
    
    print("🎯 Stop Loss Distances (sorted by smallest loss):\n")
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
        print(f"   Konservativ SL: {p75+0.5:.1f}% (unngår 75% av whipsaws)")
        print(f"   Balansert SL: {p50+0.5:.1f}% (unngår 50% av whipsaws)")
else:
    print("✅ Ingen SL hits i dag!")

# Analyze partial TP effectiveness
print("\n" + "="*80)
print("💰 TAKE PROFIT ANALYSE")
print("="*80 + "\n")

tp_hits = [t for t in trades if t['pnl'] > 0]

if tp_hits:
    print(f"✅ {len(tp_hits)} profitable exits i dag\n")
    
    tp_movements = []
    for trade in tp_hits:
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
            'pnl': trade['pnl'],
            'exit_reason': trade['exit_reason']
        })
    
    tp_movements.sort(key=lambda x: x['move_pct'], reverse=True)
    
    print("🏆 Top 15 Winning Trades:\n")
    for i, tp in enumerate(tp_movements[:15], 1):
        print(f"   {i}. {tp['symbol']} {tp['side']}: +{tp['move_pct']:.2f}% → ${tp['pnl']:.2f} ({tp['exit_reason']})")
    
    moves = [t['move_pct'] for t in tp_movements]
    moves.sort()
    
    if len(moves) >= 4:
        p25 = moves[len(moves)//4]
        p50 = moves[len(moves)//2]
        p75 = moves[3*len(moves)//4]
        p90 = moves[int(0.9*len(moves))] if len(moves) >= 10 else moves[-1]
        
        print(f"\n📊 Profit Move Percentiles:")
        print(f"   25th percentile: +{p25:.2f}%")
        print(f"   50th percentile (median): +{p50:.2f}%")
        print(f"   75th percentile: +{p75:.2f}%")
        print(f"   90th percentile: +{p90:.2f}%")
        
        print(f"\n💡 ANBEFALING:")
        print(f"   TP1 (50%): {p25:.1f}% (rask profitt)")
        print(f"   TP2 (30%): {p50:.1f}% (median)")
        print(f"   TP3 (20%): Trailing fra {p75:.1f}%")

# FINAL RECOMMENDATIONS
print("\n" + "="*80)
print("💡 OPTIMALE TP/SL INNSTILLINGER")
print("="*80 + "\n")

print("🛡️ STOP LOSS STRATEGI:")
print("   ✅ Initial SL: 2.5-3.0% (romslig, unngår whipsaw)")
print("   ✅ Flytt til breakeven: Når trade er +1.5% i profitt")
print("   ✅ Trailing SL: Følg pris med 1% avstand når > +3%")
print("   ✅ Max loss per trade: $50 (justér leverage deretter)")

print("\n💰 TAKE PROFIT STRATEGI (Trailing Profit Taking):")
print("   🎯 TP1 (50% position): +1.5-2.0% (rask profitt)")
print("   🎯 TP2 (30% position): +3.0-4.0% (solid profitt)")
print("   🎯 TP3 (20% position): Trailing -0.5% fra topp når > +5%")

print("\n🔄 DYNAMISK JUSTERING:")
print("   📈 Ved +1.5%: Flytt SL til breakeven (0%)")
print("   📈 Ved +3.0%: Flytt SL til +1.5%")
print("   📈 Ved +5.0%: Trailing SL 1% under høyeste")

print("\n⚖️ RISK/REWARD:")
print("   Initial: TP avg 2-4% / SL 2.5-3% = ~1:1 to 1.5:1")
print("   Etter breakeven move: R/R blir 2:1 til 3:1")
print("   Med trailing: Unlimited upside potential")

print("\n✅ FORDELER:")
print("   ✓ Unngår whipsaw med romslig initial SL")
print("   ✓ Sikrer profitt raskt med TP1 (50% @ 1.5-2%)")
print("   ✓ Reduserer risk med breakeven move")
print("   ✓ Lar vinnere løpe med trailing TP3")
print("   ✓ Høy frekvens profit taking (ofte små gevinster)")

print("\n" + "="*80)
print("✅ ANALYSE FULLFØRT!")
print("="*80 + "\n")

conn.close()
