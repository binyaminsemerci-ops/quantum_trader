#!/usr/bin/env python3
"""
Analyser ALLE lukkede positions fra Binance (ikke bare testnet)
Hent fra REALIZED_PNL income history
"""
import os
import sys
from dotenv import load_dotenv
from binance.client import Client
from datetime import datetime, timedelta
from collections import defaultdict

load_dotenv()

# Bruk REAL Binance keys (ikke testnet)
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

if not api_key or not api_secret:
    print("❌ BINANCE_API_KEY or BINANCE_API_SECRET not set in .env")
    sys.exit(1)

# Real Binance (ikke testnet)
client = Client(api_key, api_secret)
client.API_URL = 'https://fapi.binance.com'

print("="*80)
print("📊 BINANCE FUTURES - LUKKEDE POSITIONS ANALYSE")
print("="*80)

try:
    # Hent siste 1000 REALIZED PNL events
    print("\n🔍 Henter lukkede positions fra Binance...")
    income_history = client.futures_income_history(
        incomeType="REALIZED_PNL",
        limit=1000
    )
    
    if not income_history:
        print("⚠️  Ingen lukkede positions funnet")
        print("💡 Dette kan bety:")
        print("   - Systemet bruker testnet (ingen real trades)")
        print("   - Ingen positions har lukket enda")
        print("   - API keys har ikke futures trading aktivert")
        sys.exit(0)
    
    print(f"✅ Hentet {len(income_history)} income events\n")
    
    # Filtrer og analyser
    now = datetime.now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    all_trades = []
    today_trades = []
    week_trades = []
    month_trades = []
    
    symbol_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'count': 0})
    
    for income in income_history:
        trade_time = datetime.fromtimestamp(income['time'] / 1000)
        pnl = float(income['income'])
        symbol = income['symbol']
        
        trade_data = {
            'symbol': symbol,
            'pnl': pnl,
            'time': trade_time,
            'asset': income['asset']
        }
        
        all_trades.append(trade_data)
        
        # Update symbol stats
        symbol_stats[symbol]['count'] += 1
        symbol_stats[symbol]['pnl'] += pnl
        if pnl > 0:
            symbol_stats[symbol]['wins'] += 1
        elif pnl < 0:
            symbol_stats[symbol]['losses'] += 1
        
        # Time filtering
        if trade_time >= day_ago:
            today_trades.append(trade_data)
        if trade_time >= week_ago:
            week_trades.append(trade_data)
        if trade_time >= month_ago:
            month_trades.append(trade_data)
    
    # Calculate overall stats
    def calc_stats(trades):
        if not trades:
            return None
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        return {
            'total': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    print("📈 OVERALL STATISTIKK:")
    print("-" * 80)
    
    all_stats = calc_stats(all_trades)
    if all_stats:
        print(f"  Total closed positions: {all_stats['total']}")
        print(f"  🟢 Wins:  {all_stats['wins']} ({all_stats['win_rate']:.1f}%)")
        print(f"  🔴 Losses: {all_stats['losses']}")
        print(f"  💰 Total PnL: {all_stats['pnl']:.2f} USDT")
        print(f"  📊 Avg Win:  +{all_stats['avg_win']:.2f} USDT")
        print(f"  📉 Avg Loss: {all_stats['avg_loss']:.2f} USDT")
        if all_stats['avg_loss'] != 0:
            rr = abs(all_stats['avg_win'] / all_stats['avg_loss'])
            print(f"  🎲 Win/Loss Ratio: {rr:.2f}x")
    
    print("\n⏰ TIMEFRAME BREAKDOWN:")
    print("-" * 80)
    
    for label, trades in [("Siste 24t", today_trades), ("Siste 7 dager", week_trades), ("Siste 30 dager", month_trades)]:
        stats = calc_stats(trades)
        if stats and stats['total'] > 0:
            print(f"\n{label}: {stats['total']} trades")
            print(f"  Win rate: {stats['win_rate']:.1f}% | PnL: {stats['pnl']:.2f} USDT")
    
    if not week_trades:
        print("\n⚠️  INGEN positions lukket siste 7 dager!")
        print("💡 Dette forklarer hvorfor:")
        print("   - Smart Position Sizer har ingen data å tracke")
        print("   - RL Agent har ikke fått nye læring samples")
        print("   - Database er tom (ingen closed trades)")
        print("\n🔍 MULIGE ÅRSAKER:")
        print("   1. TP/SL levels er for langt unna (markedet når dem ikke)")
        print("   2. Systemet kjører på testnet (ingen real positions)")
        print("   3. Positions er åpne men ikke lukket enda")
    
    print("\n📊 PER-SYMBOL PERFORMANCE:")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'PnL (USDT)':<12}")
    print("-" * 80)
    
    # Sort by PnL
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    
    for symbol, stats in sorted_symbols[:15]:  # Top 15
        wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
        print(f"{symbol:<12} {stats['count']:<8} {stats['wins']:<6} {stats['losses']:<8} {wr:<7.1f}% {stats['pnl']:<12.2f}")
    
    print("\n" + "="*80)
    print("🎯 VURDERING:")
    print("-" * 80)
    
    if all_stats:
        if all_stats['win_rate'] < 30:
            print("❌ KRITISK: Win rate <30% - AI modeller fungerer IKKE bra")
            print("   HANDLING: Retrain AI modeller eller juster strategy")
        elif all_stats['win_rate'] < 40:
            print("⚠️  DÅRLIG: Win rate <40% - Forbedring nødvendig")
            print("   HANDLING: Tune parameters eller feature engineering")
        elif all_stats['win_rate'] < 50:
            print("⚡ AKSEPTABELT: Win rate 40-50%")
            print("   HANDLING: Continue monitoring, small tweaks")
        elif all_stats['win_rate'] < 60:
            print("✅ GODT: Win rate 50-60% - Modeller fungerer bra")
            print("   HANDLING: Maintain current approach")
        else:
            print("🏆 UTMERKET: Win rate >60% - Veldig bra!")
            print("   HANDLING: Consider increasing position sizing")
        
        if all_stats['pnl'] < 0:
            print(f"\n💸 NETTO TAP: {all_stats['pnl']:.2f} USDT")
            print("   Smart Position Sizer vil blokkere hvis win rate <30%")
        else:
            print(f"\n💰 NETTO PROFIT: {all_stats['pnl']:.2f} USDT")
    
    # Siste 20 trades trend
    if len(all_trades) >= 20:
        print("\n📉 RECENT TREND (siste 20 trades):")
        print("-" * 80)
        recent_20 = all_trades[:20]
        recent_stats = calc_stats(recent_20)
        print(f"  Win rate: {recent_stats['win_rate']:.1f}%")
        print(f"  PnL: {recent_stats['pnl']:.2f} USDT")
        
        if recent_stats['win_rate'] < all_stats['win_rate'] - 10:
            print("  ⚠️  TREND: Performance FORVERRER seg!")
        elif recent_stats['win_rate'] > all_stats['win_rate'] + 10:
            print("  ✅ TREND: Performance FORBEDRER seg!")
        else:
            print("  ➡️  TREND: Stabil performance")
    
    print("\n" + "="*80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n💡 MULIGE ÅRSAKER:")
    print("   1. API keys har ikke Futures trading aktivert")
    print("   2. IP ikke whitelisted (hvis API restrictions enabled)")
    print("   3. Systemet kjører på testnet (ingen real data)")
    print("   4. Binance API temporarily down")
    print("\n🔧 FIX:")
    print("   - Sjekk at API keys har 'Enable Futures' permission")
    print("   - Whitelist IP på Binance API settings")
    print("   - Eller bruk testnet keys hvis det er testnet system")
    sys.exit(1)
