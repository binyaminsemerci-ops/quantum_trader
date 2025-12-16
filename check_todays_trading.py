"""Check Today's Trading Performance"""

import sys
sys.path.insert(0, '/app')

from binance.client import Client
from datetime import datetime, timedelta
import os
import json

print("📈 TODAY'S TRADING PERFORMANCE")
print("=" * 80)

# Initialize Binance client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Get today's date range
today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
now = datetime.utcnow()

print(f"\n📅 Period: {today_start.strftime('%Y-%m-%d %H:%M')} - {now.strftime('%Y-%m-%d %H:%M')} UTC")
print("=" * 80)

# Get all trades today
all_trades = []
total_pnl = 0.0
winning_trades = 0
losing_trades = 0

# Define symbols to check (most active ones)
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SEIUSDT',
    'TRXUSDT', 'TONUSDT', 'FILUSDT', 'INJUSDT', 'DYMUSDT'
]

print("\n1️⃣ ANALYZING TRADES PER SYMBOL:")
print("-" * 80)

for symbol in symbols:
    try:
        # Get trades for symbol
        trades = client.futures_account_trades(
            symbol=symbol,
            startTime=int(today_start.timestamp() * 1000),
            endTime=int(now.timestamp() * 1000)
        )
        
        if trades:
            print(f"\n{symbol}:")
            
            # Group by position (buy/sell pairs)
            buys = [t for t in trades if t['side'] == 'BUY']
            sells = [t for t in trades if t['side'] == 'SELL']
            
            # Calculate PnL
            symbol_pnl = 0.0
            for trade in trades:
                pnl = float(trade.get('realizedPnl', 0))
                symbol_pnl += pnl
                
                # Print trade details
                side = trade['side']
                qty = float(trade['qty'])
                price = float(trade['price'])
                time = datetime.fromtimestamp(trade['time'] / 1000)
                commission = float(trade['commission'])
                
                emoji = "📈" if side == "BUY" else "📉"
                print(f"   {emoji} {time.strftime('%H:%M:%S')} - {side} {qty} @ ${price}")
                if pnl != 0:
                    pnl_emoji = "✅" if pnl > 0 else "❌"
                    print(f"      {pnl_emoji} PnL: ${pnl:.2f}")
            
            if symbol_pnl != 0:
                total_pnl += symbol_pnl
                if symbol_pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                pnl_emoji = "✅" if symbol_pnl > 0 else "❌"
                print(f"   {pnl_emoji} Total PnL: ${symbol_pnl:.2f}")
            
            all_trades.extend(trades)
    
    except Exception as e:
        # Skip symbols with no trades
        pass

print("\n" + "=" * 80)
print("\n2️⃣ TODAY'S SUMMARY:")
print("-" * 80)

if all_trades:
    print(f"   Total Trades: {len(all_trades)}")
    print(f"   Winning Positions: {winning_trades}")
    print(f"   Losing Positions: {losing_trades}")
    
    if winning_trades + losing_trades > 0:
        win_rate = (winning_trades / (winning_trades + losing_trades)) * 100
        print(f"   Win Rate: {win_rate:.1f}%")
    
    total_emoji = "✅" if total_pnl > 0 else "❌" if total_pnl < 0 else "➖"
    print(f"\n   {total_emoji} Total PnL: ${total_pnl:.2f}")
    
    # Get current open positions
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    print(f"\n   📊 Current Open Positions: {len(open_positions)}")
    
    if open_positions:
        print("\n3️⃣ OPEN POSITIONS:")
        print("-" * 80)
        for pos in open_positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry = float(pos['entryPrice'])
            unrealized = float(pos['unRealizedProfit'])
            
            side = "LONG" if amt > 0 else "SHORT"
            emoji = "🟢" if unrealized > 0 else "🔴"
            
            print(f"   {emoji} {symbol} {side}")
            print(f"      Entry: ${entry:.4f}")
            print(f"      Unrealized PnL: ${unrealized:.2f}")
else:
    print("   ⚠️  No trades found today")
    
    # Check if there are any open positions
    try:
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        print(f"   Current Open Positions: {len(open_positions)}")
    except:
        pass

print("\n" + "=" * 80)
print("✅ TRADING PERFORMANCE CHECK COMPLETE")
