#!/usr/bin/env python3
"""Quick check of open positions"""
import os
import sys
from binance.client import Client

# Testnet API credentials
api_key = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")

if not api_key or not api_secret:
    print("❌ Missing API credentials!")
    sys.exit(1)

# Initialize client for TESTNET
client = Client(api_key, api_secret, testnet=True)

# Get all open positions
positions = client.futures_position_information()

print("\n[CHART] OPEN POSITIONS:")
print("=" * 80)

open_count = 0
for pos in positions:
    amt = float(pos['positionAmt'])
    if amt != 0:
        open_count += 1
        symbol = pos['symbol']
        entry = float(pos['entryPrice'])
        mark = float(pos['markPrice'])
        leverage = int(pos.get('leverage', 30))  # Default to 30x
        unrealized_pnl = float(pos['unRealizedProfit'])
        
        # Calculate PnL %
        if entry > 0:
            price_change_pct = ((mark - entry) / entry) * 100
            if amt < 0:  # SHORT
                price_change_pct = -price_change_pct
            margin_pnl_pct = price_change_pct * leverage
        else:
            margin_pnl_pct = 0
        
        side = "LONG" if amt > 0 else "SHORT"
        print(f"{symbol:12} {side:5} {abs(amt):10.4f} @ ${entry:10.4f} "
              f"| Mark: ${mark:10.4f} | {leverage}x | "
              f"PnL: ${unrealized_pnl:8.2f} ({margin_pnl_pct:+.2f}%)")

print("=" * 80)
print(f"Total Open Positions: {open_count}")
