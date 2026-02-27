import os, sys
from binance.client import Client

# Working credentials from /etc/quantum/position-monitor-secrets/binance.env
api_key = "h3eVBK36xNmmFYbdLOkFJkDN2jTSd8HvJpY0qhvVDrTITwLppYLmA8bCLdV8kmpS"
api_secret = "aauoN8u33khQpHdQRwDxrjnI5I8iCtTzfWdCp7dmbxKoXi45PpN9H48UqUx2qKq9"

try:
    client = Client(api_key, api_secret)
    account = client.futures_account()
    print(f"✓ Connected to Binance Futures (canTrade: {account['canTrade']})")
    
    # Hent alle trades for alle mulige symbols
    symbols = ["RIVERUSDT", "ARCUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    all_trades = []
    
    for symbol in symbols:
        try:
            trades = client.futures_account_trades(symbol=symbol, limit=500)
            if trades:
                print(f"  {symbol}: {len(trades)} trades")
                all_trades.extend(trades)
        except Exception as e:
            if "Invalid symbol" not in str(e):
                print(f"  {symbol}: {e}")
    
    print(f"\n📊 TOTAL: {len(all_trades)} trades from Binance account history")
    
    if all_trades:
        # Beregn total PnL
        total_pnl = sum(float(t.get('realizedPnl', 0)) for t in all_trades)
        print(f"💰 Total Realized PnL: ${total_pnl:.2f}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
