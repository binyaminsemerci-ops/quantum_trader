"""Analyze trading activity after 15:00 today"""

import sys
sys.path.insert(0, '/app')

from binance.client import Client
from datetime import datetime, timedelta
import os

print("🔍 ANALYZING TRADING AFTER 15:00")
print("=" * 80)

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Time window after 15:00
start_time = datetime(2025, 12, 8, 15, 0, 0)
now = datetime.utcnow()

print(f"\n📅 Period: {start_time.strftime('%H:%M')} - {now.strftime('%H:%M:%S')} UTC")
print("=" * 80)

# Check all symbols for activity
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SEIUSDT'
]

print("\n🔍 TRADES AFTER 15:00:\n")

total_trades_after_15 = 0
total_pnl_after_15 = 0.0
symbols_traded = []

for symbol in symbols:
    try:
        trades = client.futures_account_trades(
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(now.timestamp() * 1000)
        )
        
        if trades:
            total_trades_after_15 += len(trades)
            symbols_traded.append(symbol)
            
            print(f"\n{symbol}:")
            print("-" * 80)
            
            symbol_pnl = 0.0
            position_opens = []
            position_closes = []
            
            for trade in trades:
                time = datetime.fromtimestamp(trade['time'] / 1000)
                side = trade['side']
                qty = float(trade['qty'])
                price = float(trade['price'])
                pnl = float(trade.get('realizedPnl', 0))
                
                emoji = "📈" if side == "BUY" else "📉"
                time_str = time.strftime('%H:%M:%S')
                
                print(f"   {emoji} {time_str} - {side} {qty} @ ${price}")
                
                if side == "BUY":
                    position_opens.append(time_str)
                elif side == "SELL":
                    position_closes.append(time_str)
                
                if pnl != 0:
                    symbol_pnl += pnl
                    total_pnl_after_15 += pnl
                    pnl_emoji = "✅" if pnl > 0 else "❌"
                    print(f"      {pnl_emoji} PnL: ${pnl:.2f}")
            
            if symbol_pnl != 0:
                emoji = "✅" if symbol_pnl > 0 else "❌"
                print(f"\n   {emoji} Symbol Total PnL: ${symbol_pnl:.2f}")
                print(f"   📊 Trades: {len(trades)} ({len(position_opens)} opens, {len(position_closes)} closes)")
                
    except Exception as e:
        pass

print("\n" + "=" * 80)
print("\n📊 SUMMARY AFTER 15:00:")
print("=" * 80)

if total_trades_after_15 > 0:
    print(f"\n✅ Trading resumed after circuit breaker")
    print(f"\n   Total trades: {total_trades_after_15}")
    print(f"   Symbols traded: {len(symbols_traded)}")
    print(f"   Symbols: {', '.join(symbols_traded)}")
    
    pnl_emoji = "✅" if total_pnl_after_15 > 0 else "❌" if total_pnl_after_15 < 0 else "➖"
    print(f"\n   {pnl_emoji} Total PnL after 15:00: ${total_pnl_after_15:.2f}")
    
    # Check current positions
    print("\n📊 CURRENT OPEN POSITIONS:")
    print("-" * 80)
    
    try:
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        
        if open_positions:
            print(f"\nOpen positions: {len(open_positions)}\n")
            for pos in open_positions:
                symbol = pos['symbol']
                amt = float(pos['positionAmt'])
                entry = float(pos['entryPrice'])
                unrealized = float(pos['unRealizedProfit'])
                
                side = "LONG" if amt > 0 else "SHORT"
                emoji = "🟢" if unrealized > 0 else "🔴"
                
                print(f"   {emoji} {symbol} {side}")
                print(f"      Entry: ${entry:.4f}")
                print(f"      Qty: {abs(amt)}")
                print(f"      Unrealized PnL: ${unrealized:.2f}")
        else:
            print("\n⚠️  NO OPEN POSITIONS")
    except Exception as e:
        print(f"Error checking positions: {e}")
    
else:
    print("\n🔴 NO TRADING ACTIVITY after 15:00")
    print("\nPossible reasons:")
    print("   - Circuit breaker still active (was set until 17:15)")
    print("   - No signals generated")
    print("   - All signals rejected by risk management")
    print("   - System in cooldown mode")
    
    # Check when circuit breaker was supposed to expire
    print("\n⏰ CIRCUIT BREAKER INFO:")
    print(f"   Activated: 13:15 UTC")
    print(f"   Should expire: 17:15 UTC")
    print(f"   Current time: {now.strftime('%H:%M:%S')} UTC")
    
    if now < datetime(2025, 12, 8, 17, 15, 0):
        remaining = datetime(2025, 12, 8, 17, 15, 0) - now
        minutes = int(remaining.total_seconds() / 60)
        print(f"   Status: ACTIVE (still {minutes} minutes remaining)")
    else:
        print(f"   Status: Should be CLEARED")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
