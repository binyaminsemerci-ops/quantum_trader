#!/usr/bin/env python3
"""
Analyser ALLE lukkede positions fra:
1. Backend database (TradeLog)
2. Docker logs (Position closed events)
3. Binance position history
"""
import sqlite3
import subprocess
import re
from datetime import datetime, timedelta
from collections import defaultdict

print("="*80)
print("📊 COMPREHENSIVE CLOSED POSITIONS ANALYSIS")
print("="*80)

# METHOD 1: Check backend database
print("\n🔍 METHOD 1: Backend Database (TradeLog)")
print("-" * 80)
try:
    conn = sqlite3.connect('backend/data/trading.db')
    cursor = conn.cursor()
    
    # Check if TradeLog table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TradeLog'")
    if not cursor.fetchone():
        print("⚠️  TradeLog table does not exist in database")
    else:
        cursor.execute("SELECT COUNT(*) FROM TradeLog WHERE exit_time IS NOT NULL")
        closed_count = cursor.fetchone()[0]
        
        if closed_count == 0:
            print("❌ No closed positions in TradeLog (exit_time IS NULL for all)")
        else:
            print(f"✅ Found {closed_count} closed positions in database\n")
            
            # Get detailed stats
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM TradeLog 
                WHERE exit_time IS NOT NULL
                GROUP BY symbol
                ORDER BY total_pnl DESC
            """)
            
            print(f"{'Symbol':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'PnL':<12}")
            print("-" * 80)
            
            for row in cursor.fetchall():
                symbol, trades, wins, losses, total_pnl, avg_pnl = row
                win_rate = (wins / trades * 100) if trades > 0 else 0
                print(f"{symbol:<12} {trades:<8} {wins:<6} {losses:<8} {win_rate:<7.1f}% {total_pnl:<12.2f}")
    
    conn.close()
except Exception as e:
    print(f"❌ Database error: {e}")

# METHOD 2: Parse Docker logs for Position closed events
print("\n🔍 METHOD 2: Docker Logs (Position closed events)")
print("-" * 80)
try:
    # Get all logs (might take a moment)
    result = subprocess.run(
        ['docker', 'logs', 'quantum_backend', '2>&1'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    logs = result.stdout
    
    # Pattern 1: "Position closed" events
    closed_pattern = r'Position closed.*?(\w+USDT).*?PnL:\s*([-+]?\d+\.?\d*)'
    matches = re.findall(closed_pattern, logs, re.IGNORECASE)
    
    if not matches:
        print("⚠️  No 'Position closed' events found in Docker logs")
    else:
        print(f"✅ Found {len(matches)} Position closed events\n")
        
        symbol_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'count': 0})
        
        for symbol, pnl_str in matches:
            pnl = float(pnl_str)
            symbol_stats[symbol]['count'] += 1
            symbol_stats[symbol]['pnl'] += pnl
            if pnl > 0:
                symbol_stats[symbol]['wins'] += 1
            elif pnl < 0:
                symbol_stats[symbol]['losses'] += 1
        
        print(f"{'Symbol':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'PnL':<12}")
        print("-" * 80)
        
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        
        for symbol, stats in sorted_symbols:
            wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
            print(f"{symbol:<12} {stats['count']:<8} {stats['wins']:<6} {stats['losses']:<8} {wr:<7.1f}% {stats['pnl']:<12.2f}")
        
        # Overall stats
        total_trades = sum(s['count'] for s in symbol_stats.values())
        total_wins = sum(s['wins'] for s in symbol_stats.values())
        total_losses = sum(s['losses'] for s in symbol_stats.values())
        total_pnl = sum(s['pnl'] for s in symbol_stats.values())
        
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        print("\n📈 OVERALL:")
        print(f"  Total: {total_trades} | Wins: {total_wins} ({overall_wr:.1f}%) | Losses: {total_losses}")
        print(f"  PnL: {total_pnl:.2f} USDT")
        
        if overall_wr < 30:
            print("\n❌ CRITICAL: Win rate <30% - Emergency stop should trigger!")
        elif overall_wr < 40:
            print("\n⚠️  WARNING: Win rate <40% - Performance needs improvement")
        else:
            print("\n✅ ACCEPTABLE: Win rate >=40%")
    
    # Pattern 2: "RL updating Q-table" after closed positions
    rl_pattern = r'Detected (\d+) closed positions - updating Q-table'
    rl_matches = re.findall(rl_pattern, logs)
    
    if rl_matches:
        total_rl_updates = sum(int(m) for m in rl_matches)
        print(f"\n🧠 RL LEARNING:")
        print(f"  Q-table updates: {len(rl_matches)} events")
        print(f"  Total positions learned from: {total_rl_updates}")
    
except subprocess.TimeoutExpired:
    print("❌ Docker logs timeout (logs too large)")
except Exception as e:
    print(f"❌ Docker logs error: {e}")

# METHOD 3: Check recent Docker logs (last 10 minutes)
print("\n🔍 METHOD 3: Recent Activity (last 10 min)")
print("-" * 80)
try:
    result = subprocess.run(
        ['docker', 'logs', 'quantum_backend', '--since', '10m', '2>&1'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    recent_logs = result.stdout
    
    # Count position closed events
    recent_closed = len(re.findall(r'Position closed', recent_logs, re.IGNORECASE))
    
    # Count TP/SL hits
    tp_hits = len(re.findall(r'TP hit|Take profit', recent_logs, re.IGNORECASE))
    sl_hits = len(re.findall(r'SL hit|Stop loss', recent_logs, re.IGNORECASE))
    
    print(f"  Positions closed: {recent_closed}")
    print(f"  TP hits: {tp_hits}")
    print(f"  SL hits: {sl_hits}")
    
    if recent_closed == 0:
        print("\n⚠️  NO positions closed in last 10 minutes")
        print("💡 This means:")
        print("   - TP/SL levels not being hit")
        print("   - Positions stuck open (market not moving enough)")
        print("   - No portfolio rotation happening")
    else:
        print(f"\n✅ {recent_closed} positions closed recently - system is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("🎯 CONCLUSION:")
print("-" * 80)
print("""
If ALL methods show 0 closed positions:
  ❌ PROBLEM: Positions NOT closing (TP/SL levels too wide)
  
If Docker logs show closed positions but database empty:
  ⚠️  PROBLEM: Database not persisting trade history
  
If recent activity (10min) shows 0 closed:
  ⚠️  PROBLEM: Current TP/SL settings prevent position rotation
  
RECOMMENDED ACTION:
  1. Reduce TP from 6% to 3% (more realistic for crypto volatility)
  2. Reduce SL from 2.5% to 1.5% (tighter risk control)
  3. Monitor next 1-2 hours for position closes
  4. Verify Smart Position Sizer starts tracking win rates
""")
print("="*80)
