#!/usr/bin/env python3
"""Check Binance Futures symbol precision requirements."""
from binance.client import Client
import os

client = Client(os.getenv("BINANCE_TESTNET_API_KEY"), os.getenv("BINANCE_TESTNET_SECRET_KEY"), testnet=True)
client.FUTURES_URL = "https://testnet.binancefuture.com"

# Get precision for common symbols
symbols_to_check = ["SOLUSDT", "BNBUSDT", "DOTUSDT", "BTCUSDT", "ETHUSDT"]

info = client.futures_exchange_info()
for symbol_name in symbols_to_check:
    for s in info["symbols"]:
        if s["symbol"] == symbol_name:
            print(f"\n{s['symbol']}:")
            print(f"  Quantity Precision: {s['quantityPrecision']}")
            print(f"  Price Precision: {s['pricePrecision']}")
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    print(f"  LOT_SIZE: stepSize={f['stepSize']}")
            break
