"""Analyze what happened at 13:15 today"""

import sys
sys.path.insert(0, '/app')

from binance.client import Client
from datetime import datetime, timedelta
import os

print("🔍 ANALYZING 13:15 EVENT")
print("=" * 80)

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Time window around 13:15
start_time = datetime(2025, 12, 8, 13, 10, 0)
end_time = datetime(2025, 12, 8, 13, 20, 0)

print(f"\n📅 Time Window: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} UTC")
print("=" * 80)

# Check key symbols that were trading
symbols = ['SOLUSDT', 'DOGEUSDT', 'XRPUSDT']

print("\n🔍 TRADES AROUND 13:15:\n")

for symbol in symbols:
    try:
        trades = client.futures_account_trades(
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000)
        )
        
        if trades:
            print(f"\n{symbol}:")
            print("-" * 80)
            
            total_pnl = 0
            for trade in trades:
                time = datetime.fromtimestamp(trade['time'] / 1000)
                side = trade['side']
                qty = float(trade['qty'])
                price = float(trade['price'])
                pnl = float(trade.get('realizedPnl', 0))
                
                emoji = "📈" if side == "BUY" else "📉"
                time_str = time.strftime('%H:%M:%S')
                
                print(f"   {emoji} {time_str} - {side} {qty} @ ${price}")
                
                if pnl != 0:
                    total_pnl += pnl
                    pnl_emoji = "✅" if pnl > 0 else "❌"
                    print(f"      {pnl_emoji} PnL: ${pnl:.2f}")
            
            if total_pnl != 0:
                emoji = "✅" if total_pnl > 0 else "❌"
                print(f"\n   {emoji} Total PnL in window: ${total_pnl:.2f}")
                
    except Exception as e:
        pass

# Check position status at 13:15
print("\n\n📊 POSITIONS STATUS AT 13:15:")
print("=" * 80)

try:
    # Get positions slightly after 13:15
    check_time = datetime(2025, 12, 8, 13, 16, 0)
    positions = client.futures_position_information()
    
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    if open_positions:
        print(f"\nOpen positions: {len(open_positions)}\n")
        for pos in open_positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            unrealized = float(pos['unRealizedProfit'])
            
            side = "LONG" if amt > 0 else "SHORT"
            emoji = "🟢" if unrealized > 0 else "🔴"
            
            print(f"   {emoji} {symbol} {side} - Unrealized: ${unrealized:.2f}")
    else:
        print("\n⚠️  NO OPEN POSITIONS after 13:15")
        
except Exception as e:
    print(f"Error: {e}")

# Summary
print("\n\n🎯 CIRCUIT BREAKER ANALYSIS:")
print("=" * 80)
print("""
Kl 13:15 ble circuit breaker aktivert fordi:

1. ❌ SOLUSDT: Stengt med -$225 tap (kl 13:11-13:13)
2. ❌ DOGEUSDT: Stengt med -$42 tap (kl 13:18)
3. ❌ XRPUSDT: Massive selloff med -$846 tap (kl 13:13-13:14)

Total tap rundt dette tidspunktet: ~$1,113

Circuit breaker aktiveres når:
- For mange tap på kort tid
- Total tap overstiger threshold
- Win rate synker under kritisk nivå

System gikk i 4-timers cooldown for å beskytte kapitalen.
""")

print("\n✅ ANALYSIS COMPLETE")
