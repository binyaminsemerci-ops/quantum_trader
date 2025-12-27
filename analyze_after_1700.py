"""Analyze detailed trading activity after 17:00 today"""

import sys
sys.path.insert(0, '/app')

from binance.client import Client
from datetime import datetime, timedelta
import os

print("🔍 DETAILED ANALYSIS AFTER 17:00")
print("=" * 80)

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Time window after 17:00
start_time = datetime(2025, 12, 8, 17, 0, 0)
now = datetime.utcnow()

print(f"\n📅 Period: {start_time.strftime('%H:%M')} - {now.strftime('%H:%M:%S')} UTC")
print(f"Duration: {(now - start_time).total_seconds() / 3600:.1f} hours")
print("=" * 80)

# Check all symbols
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SEIUSDT',
    'TRXUSDT', 'TONUSDT'
]

print("\n📊 HOURLY BREAKDOWN:")
print("=" * 80)

# Group by hour
hourly_data = {}
for hour in range(17, 23):
    hourly_data[hour] = {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0}

total_trades = 0
total_pnl = 0.0
symbols_with_trades = {}

for symbol in symbols:
    try:
        trades = client.futures_account_trades(
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(now.timestamp() * 1000)
        )
        
        if trades:
            if symbol not in symbols_with_trades:
                symbols_with_trades[symbol] = {'trades': [], 'pnl': 0.0, 'wins': 0, 'losses': 0}
            
            for trade in trades:
                total_trades += 1
                time = datetime.fromtimestamp(trade['time'] / 1000)
                hour = time.hour
                pnl = float(trade.get('realizedPnl', 0))
                
                symbols_with_trades[symbol]['trades'].append({
                    'time': time,
                    'side': trade['side'],
                    'qty': float(trade['qty']),
                    'price': float(trade['price']),
                    'pnl': pnl
                })
                
                if pnl != 0:
                    symbols_with_trades[symbol]['pnl'] += pnl
                    total_pnl += pnl
                    
                    if hour in hourly_data:
                        hourly_data[hour]['pnl'] += pnl
                        if pnl > 0:
                            hourly_data[hour]['wins'] += 1
                            symbols_with_trades[symbol]['wins'] += 1
                        else:
                            hourly_data[hour]['losses'] += 1
                            symbols_with_trades[symbol]['losses'] += 1
                
                if hour in hourly_data:
                    hourly_data[hour]['trades'] += 1
    
    except Exception as e:
        pass

# Print hourly breakdown
for hour in sorted(hourly_data.keys()):
    data = hourly_data[hour]
    if data['trades'] > 0:
        pnl_emoji = "✅" if data['pnl'] > 0 else "❌" if data['pnl'] < 0 else "➖"
        print(f"\n{hour:02d}:00-{hour:02d}:59")
        print(f"   Trades: {data['trades']} | Wins: {data['wins']} | Losses: {data['losses']}")
        print(f"   {pnl_emoji} PnL: ${data['pnl']:.2f}")

# Detailed symbol breakdown
print("\n\n📈 SYMBOL-BY-SYMBOL BREAKDOWN:")
print("=" * 80)

for symbol in sorted(symbols_with_trades.keys()):
    data = symbols_with_trades[symbol]
    print(f"\n{symbol}:")
    print("-" * 80)
    
    # Show trades
    for trade in data['trades']:
        emoji = "📈" if trade['side'] == "BUY" else "📉"
        time_str = trade['time'].strftime('%H:%M:%S')
        print(f"   {emoji} {time_str} - {trade['side']} {trade['qty']} @ ${trade['price']}")
        
        if trade['pnl'] != 0:
            pnl_emoji = "✅" if trade['pnl'] > 0 else "❌"
            print(f"      {pnl_emoji} PnL: ${trade['pnl']:.2f}")
    
    # Summary
    pnl_emoji = "✅" if data['pnl'] > 0 else "❌" if data['pnl'] < 0 else "➖"
    win_rate = (data['wins'] / (data['wins'] + data['losses']) * 100) if (data['wins'] + data['losses']) > 0 else 0
    
    print(f"\n   {pnl_emoji} Total: ${data['pnl']:.2f} | Wins: {data['wins']} | Losses: {data['losses']} | Win Rate: {win_rate:.1f}%")

# Overall summary
print("\n\n" + "=" * 80)
print("📊 OVERALL SUMMARY AFTER 17:00:")
print("=" * 80)

print(f"\nTotal Trades: {total_trades}")
print(f"Symbols Traded: {len(symbols_with_trades)}")
print(f"Symbol List: {', '.join(sorted(symbols_with_trades.keys()))}")

total_wins = sum(d['wins'] for d in symbols_with_trades.values())
total_losses = sum(d['losses'] for d in symbols_with_trades.values())
overall_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

print(f"\nWins: {total_wins}")
print(f"Losses: {total_losses}")
print(f"Win Rate: {overall_win_rate:.1f}%")

pnl_emoji = "✅" if total_pnl > 0 else "❌" if total_pnl < 0 else "➖"
print(f"\n{pnl_emoji} Total PnL: ${total_pnl:.2f}")

# Check current positions
print("\n📊 CURRENT POSITIONS:")
print("-" * 80)

try:
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    if open_positions:
        print(f"\nOpen: {len(open_positions)}\n")
        for pos in open_positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry = float(pos['entryPrice'])
            unrealized = float(pos['unRealizedProfit'])
            
            side = "LONG" if amt > 0 else "SHORT"
            emoji = "🟢" if unrealized > 0 else "🔴"
            
            print(f"   {emoji} {symbol} {side}")
            print(f"      Entry: ${entry:.4f} | Qty: {abs(amt)}")
            print(f"      Unrealized: ${unrealized:.2f}")
    else:
        print("\n⚠️  No open positions")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
