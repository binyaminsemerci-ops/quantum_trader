#!/usr/bin/env python3
"""Force all positions to 10x leverage on Binance Futures."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from binance.client import Client
from config.config import load_config

def main():
    cfg = load_config()
    client = Client(cfg.binance_api_key, cfg.binance_api_secret)
    
    # Get all positions with non-zero size
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    print(f"\n=== Found {len(open_positions)} Open Positions ===\n")
    
    for pos in open_positions:
        symbol = pos['symbol']
        size = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        current_leverage = int(pos['leverage'])
        unrealized_pnl = float(pos['unRealizedProfit'])
        
        print(f"{symbol:12} | {abs(size):8.2f} @ ${entry:8.4f} | {current_leverage:2}x | PNL: ${unrealized_pnl:+7.2f}", end="")
        
        if current_leverage != 10:
            try:
                print(f" | 🔧 Setting to 10x...", end="", flush=True)
                client.futures_change_leverage(symbol=symbol, leverage=10)
                print(" [OK]")
            except Exception as e:
                print(f" ❌ ERROR: {e}")
        else:
            print(" | [OK] Already 10x")
    
    print("\n=== Re-checking Leverage ===\n")
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    for pos in open_positions:
        symbol = pos['symbol']
        leverage = int(pos['leverage'])
        status = "[OK]" if leverage == 10 else "❌"
        print(f"{status} {symbol:12} | {leverage:2}x")
    
    print("\n")

if __name__ == "__main__":
    main()
