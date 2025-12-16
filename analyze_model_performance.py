"""
Analyze AI model prediction quality and win rate
"""
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_predictions():
    """Analyze model predictions vs actual results."""
    
    # Connect to database
    conn = sqlite3.connect('backend/data/trades.db')
    cursor = conn.cursor()
    
    print("=" * 80)
    print("📊 AI MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Get all closed positions
    cursor.execute("""
        SELECT 
            symbol,
            side,
            entry_time,
            exit_time,
            entry_price,
            exit_price,
            quantity,
            pnl,
            status
        FROM trade_log
        WHERE status IN ('closed', 'liquidated')
        ORDER BY exit_time DESC
        LIMIT 100
    """)
    
    trades = cursor.fetchall()
    
    if not trades:
        print("\n⚠️  NO CLOSED TRADES FOUND IN DATABASE")
        print("   Systemet har kjørt men ingen posisjoner er stengt ennå")
        conn.close()
        return
    
    print(f"\n📈 TRADE HISTORY ANALYSIS (Last 100 closed trades)")
    print(f"   Found: {len(trades)} closed positions\n")
    
    # Calculate statistics
    total_trades = len(trades)
    wins = 0
    losses = 0
    total_pnl = 0.0
    total_pnl_pct = 0.0
    
    symbol_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    side_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    for trade in trades:
        symbol, side, entry_time, exit_time, entry_price, exit_price, qty, pnl, status = trade
        
        if pnl is None:
            # Calculate PnL if not stored
            if side == 'BUY':
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
        
        pnl_pct = (pnl / (entry_price * qty)) * 100 if entry_price and qty else 0
        
        total_pnl += pnl
        total_pnl_pct += pnl_pct
        
        is_win = pnl > 0
        if is_win:
            wins += 1
        else:
            losses += 1
        
        # Symbol stats
        symbol_stats[symbol]['trades'] += 1
        symbol_stats[symbol]['pnl'] += pnl
        if is_win:
            symbol_stats[symbol]['wins'] += 1
        
        # Side stats
        side_stats[side]['trades'] += 1
        side_stats[side]['pnl'] += pnl
        if is_win:
            side_stats[side]['wins'] += 1
    
    # Overall statistics
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    avg_pnl_pct = total_pnl_pct / total_trades if total_trades > 0 else 0
    
    print("📊 OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"   Total Trades: {total_trades}")
    print(f"   Wins: {wins} ({win_rate:.1f}%)")
    print(f"   Losses: {losses} ({100-win_rate:.1f}%)")
    print(f"   Total PnL: ${total_pnl:+.2f}")
    print(f"   Average PnL: ${avg_pnl:+.2f} ({avg_pnl_pct:+.2f}%)")
    
    # Win rate quality assessment
    print(f"\n🎯 WIN RATE QUALITY ASSESSMENT")
    print("-" * 80)
    if win_rate >= 50:
        print(f"   ✅ GOOD: {win_rate:.1f}% win rate is profitable")
    elif win_rate >= 40:
        print(f"   ⚠️  ACCEPTABLE: {win_rate:.1f}% win rate (needs improvement)")
    elif win_rate >= 30:
        print(f"   ❌ POOR: {win_rate:.1f}% win rate (losing money)")
    else:
        print(f"   🚨 CRITICAL: {win_rate:.1f}% win rate (severe losses)")
    
    # Per symbol analysis
    print(f"\n📊 PER SYMBOL PERFORMANCE (Top 10)")
    print("-" * 80)
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['trades'], reverse=True)[:10]
    for symbol, stats in sorted_symbols:
        sym_win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"   {symbol:10s}: {stats['trades']:3d} trades | {sym_win_rate:5.1f}% win | ${stats['pnl']:+8.2f} PnL")
    
    # Per side analysis
    print(f"\n📊 LONG vs SHORT PERFORMANCE")
    print("-" * 80)
    for side, stats in sorted(side_stats.items()):
        side_win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"   {side:5s}: {stats['trades']:3d} trades | {side_win_rate:5.1f}% win | ${stats['pnl']:+8.2f} PnL")
    
    # Recent performance
    print(f"\n📅 RECENT PERFORMANCE (Last 20 trades)")
    print("-" * 80)
    recent_trades = trades[:20]
    recent_wins = sum(1 for t in recent_trades if (t[7] or 0) > 0)
    recent_win_rate = (recent_wins / len(recent_trades) * 100) if recent_trades else 0
    recent_pnl = sum(t[7] or 0 for t in recent_trades)
    
    print(f"   Recent Win Rate: {recent_win_rate:.1f}%")
    print(f"   Recent PnL: ${recent_pnl:+.2f}")
    
    if recent_win_rate < 30:
        print(f"   🚨 WARNING: Recent performance critically poor!")
        print(f"      Smart Position Sizer should be BLOCKING trades!")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    if win_rate < 40:
        print(f"   1. 🔴 URGENT: Win rate {win_rate:.1f}% is too low")
        print(f"      • Consider retraining AI models with recent data")
        print(f"      • Check if market conditions changed")
        print(f"      • Smart Position Sizer should help reduce losses")
    
    if avg_pnl < 0:
        print(f"   2. 🔴 URGENT: Average PnL is negative (${avg_pnl:.2f})")
        print(f"      • Stop/loss management needs improvement")
        print(f"      • Position sizing too aggressive")
    
    if total_pnl < -100:
        print(f"   3. 🚨 CRITICAL: Total loss ${total_pnl:.2f}")
        print(f"      • Consider stopping system and retraining")
        print(f"      • Review trading strategy")
    
    # Check for model issues
    print(f"\n🔍 MODEL HEALTH CHECK")
    print("-" * 80)
    
    # Check if all models agree (low diversity)
    cursor.execute("""
        SELECT COUNT(*) FROM trade_log 
        WHERE status IN ('closed', 'liquidated')
        AND entry_time > datetime('now', '-7 days')
    """)
    recent_count = cursor.fetchone()[0]
    
    if recent_count == 0:
        print(f"   ⚠️  NO TRADES in last 7 days")
        print(f"      • System may be too conservative")
        print(f"      • Check confidence threshold (currently 0.20)")
    elif recent_count < 10:
        print(f"   ⚠️  VERY FEW TRADES: {recent_count} in last 7 days")
        print(f"      • Models may be uncertain")
        print(f"      • Consider lowering confidence threshold")
    else:
        print(f"   ✅ Active trading: {recent_count} trades in last 7 days")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_predictions()
