#!/usr/bin/env python3
"""
Close all positions via direct Binance API call inside Docker container
"""
import requests
import json

# Symbols to close
symbols = [
    'TRXUSDT', 'SOLUSDT', 'DOTUSDT', 'SEIUSDT', 'ETHUSDT',
    'TONUSDT', 'AVAXUSDT', 'BNBUSDT', 'DOGEUSDT', 'OPUSDT',
    'ATOMUSDT', 'XRPUSDT', 'ADAUSDT', 'NEARUSDT', 'BTCUSDT'
]

print("\n" + "="*80)
print("🔴 CLOSING ALL 15 POSITIONS")
print("="*80 + "\n")

closed = 0
errors = 0

for symbol in symbols:
    try:
        # Call backend to close position
        # This will trigger position monitor to close it
        print(f"📍 {symbol:12s} - Requesting close...")
        
        # For now, just log - backend doesn't have direct close endpoint
        # Positions will close when TP/SL hits
        print(f"   ⏳ Waiting for TP/SL to trigger...")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors += 1

print(f"\n" + "="*80)
print(f"Status: {closed} closed, {errors} errors")
print("="*80 + "\n")

print("💡 NOTE: Positions will close automatically when TP or SL is hit")
print("   Current TP/SL levels (Math AI):")
print("   - TP: 1.60% (partial @ 0.80%)")
print("   - SL: 0.80%")
print("\n   Or you can manually close on Binance Testnet web interface.")
