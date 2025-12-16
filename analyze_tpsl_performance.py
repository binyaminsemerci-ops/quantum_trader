"""Analyze TP/SL and Partial Profit Performance"""

import sys
sys.path.insert(0, '/app')

from binance.client import Client
from datetime import datetime, timedelta
import os
import json

print("🔍 ANALYZING TP/SL & PARTIAL PROFIT PERFORMANCE")
print("=" * 80)

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Get today's trades
today_start = datetime(2025, 12, 8, 0, 0, 0)
now = datetime.utcnow()

symbols = ['ADAUSDT', 'LINKUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 
           'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'DOTUSDT', 'OPUSDT']

print("\n📊 ANALYZING EXIT STRATEGIES:")
print("=" * 80)

total_wins = 0
total_losses = 0
tp_hits = 0
sl_hits = 0
partial_tp_hits = 0
small_profits = []
big_profits = []
losses = []

for symbol in symbols:
    try:
        trades = client.futures_account_trades(
            symbol=symbol,
            startTime=int(today_start.timestamp() * 1000),
            endTime=int(now.timestamp() * 1000)
        )
        
        if trades:
            # Group trades by position
            position_pnls = {}
            
            for trade in trades:
                pnl = float(trade.get('realizedPnl', 0))
                if pnl != 0:
                    time = datetime.fromtimestamp(trade['time'] / 1000)
                    
                    if pnl > 0:
                        total_wins += 1
                        if pnl < 5:
                            small_profits.append({'symbol': symbol, 'pnl': pnl, 'time': time})
                        else:
                            big_profits.append({'symbol': symbol, 'pnl': pnl, 'time': time})
                        
                        # Check if likely partial TP (small profit from larger position)
                        if pnl < 10:
                            partial_tp_hits += 1
                    else:
                        total_losses += 1
                        losses.append({'symbol': symbol, 'pnl': pnl, 'time': time})
    
    except Exception as e:
        pass

print(f"\n1️⃣ OVERALL STATS:")
print(f"   Total Winning Trades: {total_wins}")
print(f"   Total Losing Trades: {total_losses}")
print(f"   Win Rate: {(total_wins/(total_wins+total_losses)*100):.1f}%")

print(f"\n2️⃣ PROFIT DISTRIBUTION:")
print(f"   Small Profits (< $5): {len(small_profits)} trades")
print(f"   Big Profits (≥ $5): {len(big_profits)} trades")
print(f"   Likely Partial TPs: {partial_tp_hits}")

print(f"\n3️⃣ LOSS ANALYSIS:")
print(f"   Total Losses: {len(losses)}")
if losses:
    avg_loss = sum(l['pnl'] for l in losses) / len(losses)
    max_loss = min(l['pnl'] for l in losses)
    print(f"   Average Loss: ${avg_loss:.2f}")
    print(f"   Biggest Loss: ${max_loss:.2f}")

print("\n4️⃣ DETAILED SMALL PROFITS (might be partial TPs):")
print("-" * 80)
for profit in sorted(small_profits, key=lambda x: x['pnl'], reverse=True)[:20]:
    print(f"   {profit['time'].strftime('%H:%M:%S')} - {profit['symbol']:10} ${profit['pnl']:6.2f}")

print("\n5️⃣ BIG PROFITS (full position closes):")
print("-" * 80)
for profit in sorted(big_profits, key=lambda x: x['pnl'], reverse=True)[:20]:
    print(f"   {profit['time'].strftime('%H:%M:%S')} - {profit['symbol']:10} ${profit['pnl']:6.2f}")

print("\n6️⃣ BIGGEST LOSSES (SL hits):")
print("-" * 80)
for loss in sorted(losses, key=lambda x: x['pnl'])[:10]:
    print(f"   {loss['time'].strftime('%H:%M:%S')} - {loss['symbol']:10} ${loss['pnl']:6.2f}")

# Check trade_state.json
print("\n7️⃣ CHECKING TRADE STATE CONFIGURATION:")
print("-" * 80)
try:
    with open('/app/backend/data/trade_state.json', 'r') as f:
        state = json.load(f)
        
    if state:
        print("Current tracked positions:")
        for symbol, config in state.items():
            print(f"\n{symbol}:")
            print(f"   TP: {config.get('ai_tp_pct', 'N/A')*100 if config.get('ai_tp_pct') else 'N/A'}%")
            print(f"   SL: {config.get('ai_sl_pct', 'N/A')*100 if config.get('ai_sl_pct') else 'N/A'}%")
            print(f"   Trail: {config.get('ai_trail_pct', 'N/A')*100 if config.get('ai_trail_pct') else 'N/A'}%")
            print(f"   Partial TP 1: {config.get('partial_tp_1_pct', 'N/A')}%")
            print(f"   Partial TP 2: {config.get('partial_tp_2_pct', 'N/A')}%")
            print(f"   Partial 1 Hit: {config.get('partial_tp_1_hit', False)}")
            print(f"   Partial 2 Hit: {config.get('partial_tp_2_hit', False)}")
    else:
        print("⚠️  No positions in trade_state.json")
        
except FileNotFoundError:
    print("⚠️  trade_state.json not found")
except Exception as e:
    print(f"Error reading state: {e}")

# Calculate risk/reward ratios
print("\n8️⃣ RISK/REWARD ANALYSIS:")
print("-" * 80)

if total_wins > 0 and total_losses > 0:
    avg_win = (sum(p['pnl'] for p in small_profits + big_profits) / len(small_profits + big_profits))
    avg_loss = sum(l['pnl'] for l in losses) / len(losses)
    rr_ratio = abs(avg_win / avg_loss)
    
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Risk/Reward Ratio: {rr_ratio:.2f}:1")
    
    if rr_ratio < 2.0:
        print("\n⚠️  WARNING: Risk/Reward ratio < 2:1")
        print("   Consider: Wider TPs or tighter SLs")
    
    # Win rate needed to break even
    breakeven_wr = 1 / (1 + rr_ratio) * 100
    print(f"\nBreakeven Win Rate: {breakeven_wr:.1f}%")
    print(f"Current Win Rate: {(total_wins/(total_wins+total_losses)*100):.1f}%")
    
    if total_wins/(total_wins+total_losses) > breakeven_wr/100:
        print("✅ Profitable (Win Rate > Breakeven)")
    else:
        print("❌ Unprofitable (Win Rate < Breakeven)")

print("\n" + "=" * 80)
print("📊 RECOMMENDATIONS:")
print("=" * 80)

# Analyze patterns
small_profit_ratio = len(small_profits) / (len(small_profits) + len(big_profits)) if (len(small_profits) + len(big_profits)) > 0 else 0

print(f"\n1️⃣ PARTIAL PROFIT EFFECTIVENESS:")
if small_profit_ratio > 0.7:
    print("   ⚠️  TOO MANY small profits (>70%)")
    print("   → Partial TPs might be TOO AGGRESSIVE")
    print("   → Consider: Reduce partial size or increase TP distance")
elif small_profit_ratio < 0.3:
    print("   ⚠️  TOO FEW small profits (<30%)")
    print("   → Partial TPs might not be triggering")
    print("   → Consider: More aggressive partial TP levels")
else:
    print("   ✅ Good balance of partial and full profits")

print(f"\n2️⃣ STOP LOSS EFFECTIVENESS:")
if losses:
    large_losses = [l for l in losses if l['pnl'] < -50]
    if large_losses:
        print(f"   ⚠️  Found {len(large_losses)} losses > $50")
        print("   → SL might be TOO WIDE")
        print("   → Consider: Tighter stop losses")
    else:
        print("   ✅ All losses < $50 (SL working)")

print(f"\n3️⃣ TRAILING STOP:")
consecutive_small_profits = 0
for i in range(len(small_profits)-1):
    if small_profits[i]['symbol'] == small_profits[i+1]['symbol']:
        consecutive_small_profits += 1

if consecutive_small_profits > 5:
    print("   ✅ Trailing stops working (consecutive partial profits)")
else:
    print("   ⚠️  Few consecutive partials - check trailing stop activation")

print("\n✅ ANALYSIS COMPLETE")
