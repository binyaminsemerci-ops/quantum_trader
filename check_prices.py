from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)
tickers = c.futures_symbol_ticker()

symbols = ['SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'BTCUSDT']
print("\n=== Current Prices ===")
for t in tickers:
    if t['symbol'] in symbols:
        print(f"{t['symbol']:12} ${float(t['price']):>10.2f}")

print("\n=== TP Levels ===")
print("SOLUSDT  TP0: $137.53, TP1: $138.59, TP2: $140.18")
print("ETHUSDT  TP0: $3277.68, TP1: $3302.95, TP2: $3340.85")
print("XRPUSDT  TP0: $2.0549, TP1: $2.0707, TP2: $2.0945")
print("BTCUSDT  TP0: $93320.12, TP1: $94039.60, TP2: $95118.82")
