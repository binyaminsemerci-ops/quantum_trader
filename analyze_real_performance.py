"""
Analyser REAL performance fra Binance (siste 3 uker)
Vis faktisk win rate, PnL, og model kvalitet
"""

import os
import sys
from dotenv import load_dotenv
from binance.client import Client
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

if not api_key or not api_secret:
    print("❌ BINANCE_API_KEY or BINANCE_API_SECRET not set in .env")
    sys.exit(1)

client = Client(api_key, api_secret)

# Få alle lukket positions (max 1000)
print("🔍 Henter alle lukket positions fra Binance...")
all_income = client.futures_income_history(incomeType="REALIZED_PNL", limit=1000)

# Filtrer siste 3 uker
three_weeks_ago = datetime.now() - timedelta(weeks=3)
recent_positions = []

for income in all_income:
    timestamp = income['time'] / 1000  # Convert ms to seconds
    trade_date = datetime.fromtimestamp(timestamp)
    
    if trade_date >= three_weeks_ago:
        recent_positions.append({
            'symbol': income['symbol'],
            'pnl': float(income['income']),
            'date': trade_date
        })

print(f"\n📅 Posisjoner lukket siste 3 uker: {len(recent_positions)}")

if len(recent_positions) == 0:
    print("⚠️ Ingen lukkede posisjoner siste 3 uker!")
    print("💡 Systemet har kjørt men ingen positions har blitt close ELLER data er cleared")
    sys.exit(0)

# Beregn statistikk
winners = [p for p in recent_positions if p['pnl'] > 0]
losers = [p for p in recent_positions if p['pnl'] < 0]
breakeven = [p for p in recent_positions if p['pnl'] == 0]

total_pnl = sum(p['pnl'] for p in recent_positions)
win_rate = (len(winners) / len(recent_positions)) * 100 if recent_positions else 0

avg_win = sum(p['pnl'] for p in winners) / len(winners) if winners else 0
avg_loss = sum(p['pnl'] for p in losers) / len(losers) if losers else 0

# Symbol stats
symbol_stats = {}
for p in recent_positions:
    sym = p['symbol']
    if sym not in symbol_stats:
        symbol_stats[sym] = {'wins': 0, 'losses': 0, 'pnl': 0}
    
    if p['pnl'] > 0:
        symbol_stats[sym]['wins'] += 1
    elif p['pnl'] < 0:
        symbol_stats[sym]['losses'] += 1
    
    symbol_stats[sym]['pnl'] += p['pnl']

print("\n" + "="*70)
print("📊 PERFORMANCE SISTE 3 UKER (REAL DATA FRA BINANCE)")
print("="*70)
print(f"Total closed positions: {len(recent_positions)}")
print(f"  🟢 Wins:  {len(winners)}")
print(f"  🔴 Losses: {len(losers)}")
print(f"  ⚪ Breakeven: {len(breakeven)}")
print(f"\n💰 Win Rate: {win_rate:.1f}%")
print(f"   Total PnL: {total_pnl:.2f} USDT")
print(f"   Avg Win:  +{avg_win:.2f} USDT")
print(f"   Avg Loss: {avg_loss:.2f} USDT")

if avg_loss != 0:
    win_loss_ratio = abs(avg_win / avg_loss)
    print(f"   Win/Loss Ratio: {win_loss_ratio:.2f}x")

print("\n📈 PER SYMBOL PERFORMANCE:")
print("-" * 70)
print(f"{'Symbol':<12} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} {'PnL':<10}")
print("-" * 70)

for sym, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
    total_trades = stats['wins'] + stats['losses']
    sym_win_rate = (stats['wins'] / total_trades * 100) if total_trades > 0 else 0
    
    print(f"{sym:<12} {stats['wins']:<6} {stats['losses']:<8} {sym_win_rate:>6.1f}%    {stats['pnl']:>8.2f}")

print("="*70)

# VURDERING
print("\n🎯 VURDERING:")
if win_rate < 30:
    print("❌ KRITISK: Win rate <30% - AI modeller predikerer feil!")
    print("   ANBEFALING: Retrain alle modeller med nye data")
elif win_rate < 40:
    print("⚠️ DÅRLIG: Win rate <40% - Modeller underperformer")
    print("   ANBEFALING: Sjekk feature engineering og retrain")
elif win_rate < 50:
    print("⚡ AKSEPTABELT: Win rate 40-50% - OK men kan forbedres")
    print("   ANBEFALING: Tune hyperparameters eller feature selection")
elif win_rate < 60:
    print("✅ GODT: Win rate 50-60% - Modeller fungerer bra")
    print("   ANBEFALING: Fortsett current approach")
else:
    print("🏆 UTMERKET: Win rate >60% - Veldig bra modeller!")
    print("   ANBEFALING: Increase position sizing gradvis")

if total_pnl < 0:
    print(f"\n💸 NETTO TAP: {total_pnl:.2f} USDT")
    print("   Smart Position Sizer vil BLOKKERE trades hvis win rate <30%")
else:
    print(f"\n💰 NETTO PROFIT: {total_pnl:.2f} USDT")

# Siste 20 trades trend
print("\n📉 SISTE 20 TRADES TREND:")
recent_20 = sorted(recent_positions, key=lambda x: x['date'], reverse=True)[:20]
recent_20_wins = len([p for p in recent_20 if p['pnl'] > 0])
recent_20_rate = (recent_20_wins / len(recent_20)) * 100 if recent_20 else 0
recent_20_pnl = sum(p['pnl'] for p in recent_20)

print(f"Win rate (20 siste): {recent_20_rate:.1f}%")
print(f"PnL (20 siste): {recent_20_pnl:.2f} USDT")

if recent_20_rate < win_rate - 10:
    print("⚠️ TREND: Performance forverrer seg nylig!")
elif recent_20_rate > win_rate + 10:
    print("✅ TREND: Performance forbedrer seg!")
else:
    print("➡️ TREND: Stabil performance")
