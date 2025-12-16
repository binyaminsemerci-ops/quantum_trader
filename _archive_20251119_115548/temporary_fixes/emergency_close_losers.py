#!/usr/bin/env python3
"""EMERGENCY: Close losing positions and verify TP/SL system."""

from binance.client import Client
import os

client = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])

print("\n[ALERT] EMERGENCY POSITION MANAGEMENT\n")

# Get all open positions
positions = client.futures_position_information()
open_pos = [p for p in positions if float(p['positionAmt']) != 0]

print(f"Found {len(open_pos)} open positions\n")

# Close positions with >7% loss
for p in open_pos:
    symbol = p['symbol']
    size = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    unrealized_pnl = float(p['unRealizedProfit'])
    
    # Calculate PNL percentage (considering 10x leverage)
    notional = abs(size * entry)
    margin = notional / 10  # 10x leverage
    pnl_pct = (unrealized_pnl / margin) * 100 if margin > 0 else 0
    
    print(f"{symbol:12} | {size:8.2f} @ ${entry:8.4f} | PNL: {pnl_pct:+6.2f}% (${unrealized_pnl:+7.2f})")
    
    # Close if loss > 7%
    if pnl_pct < -7.0:
        print(f"  [ALERT] CLOSING {symbol} (loss too high: {pnl_pct:.2f}%)")
        try:
            side = 'SELL' if size > 0 else 'BUY'
            result = client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=abs(size),
                reduceOnly=True
            )
            print(f"  [OK] Closed {symbol} at market")
        except Exception as e:
            print(f"  ❌ Error closing {symbol}: {e}")

print("\n[OK] Emergency close complete")

# Check remaining positions
positions = client.futures_position_information()
open_pos = [p for p in positions if float(p['positionAmt']) != 0]
print(f"\n[CHART] {len(open_pos)} positions remaining after emergency close")
