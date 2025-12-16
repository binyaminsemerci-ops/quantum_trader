"""Check recent trade history to see when positions closed"""

from binance.client import Client
import os
from datetime import datetime, timedelta

client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)

print("📊 SJEKKER SISTE TRADE HISTORY")
print("=" * 80)

symbols = ['DOTUSDT', 'DOGEUSDT', 'SOLUSDT']

for symbol in symbols:
    print(f"\n🔍 {symbol}:")
    try:
        # Get recent trades
        trades = client.futures_account_trades(symbol=symbol, limit=10)
        
        if not trades:
            print("   Ingen trades funnet")
            continue
        
        # Group by time to find position opens/closes
        recent = trades[-5:]  # Last 5 trades
        
        for trade in recent:
            time = datetime.fromtimestamp(trade['time'] / 1000)
            side = trade['side']
            qty = float(trade['qty'])
            price = float(trade['price'])
            commission = float(trade['commission'])
            realized_pnl = float(trade['realizedPnl'])
            
            emoji = "📈" if side == "BUY" else "📉"
            pnl_emoji = "✅" if realized_pnl > 0 else "❌"
            
            print(f"   {emoji} {time.strftime('%H:%M:%S')} - {side} {qty} @ ${price}")
            if realized_pnl != 0:
                print(f"      {pnl_emoji} PnL: ${realized_pnl:.2f}")
                
    except Exception as e:
        print(f"   Error: {e}")

print("\n" + "=" * 80)
print("\n💡 KONKLUSJON:")
print("   Hvis siste trade er BUY (for SHORT) eller SELL (for LONG),")
print("   betyr det at posisjonen ble stengt.")
