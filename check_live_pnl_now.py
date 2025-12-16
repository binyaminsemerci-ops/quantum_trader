#!/usr/bin/env python3
import os
from binance.um_futures import UMFutures

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

client = UMFutures(key=api_key, secret=api_secret)

# Get account info
account = client.account()
balance = float(account['totalWalletBalance'])
unrealized = float(account['totalUnrealizedProfit'])
print(f"\n[MONEY] BALANCE: ${balance:.2f}")
print(f"[CHART] Unrealized PnL: ${unrealized:.2f}")
print(f"[TARGET] Balance + PnL: ${balance + unrealized:.2f}\n")

# Get positions
positions = client.get_position_risk()
open_pos = [p for p in positions if float(p['positionAmt']) != 0]

print(f"📍 OPEN POSITIONS: {len(open_pos)}/4\n")
print("-" * 80)

for p in open_pos:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    mark = float(p['markPrice'])
    upnl = float(p['unRealizedProfit'])
    leverage = int(p['leverage'])
    
    # Calculate margin and PnL percentage
    margin = abs(amt * entry) / leverage
    pnl_pct = (upnl / margin * 100) if margin > 0 else 0
    
    side = "LONG" if amt > 0 else "SHORT"
    emoji = "[OK]" if upnl > 0 else "❌"
    
    print(f"{emoji} {symbol} {side}")
    print(f"   Entry: ${entry:.6f} | Mark: ${mark:.6f}")
    print(f"   Amount: {amt} | Leverage: {leverage}x")
    print(f"   Margin: ${margin:.2f} | PnL: ${upnl:.2f} ({pnl_pct:+.2f}%)")
    print()
