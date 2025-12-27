#!/usr/bin/env python3
"""Check current balance and positions after restart."""
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Get account info
account = client.futures_account()
balance = float(account['totalWalletBalance'])
available = float(account['availableBalance'])

# Get open positions
positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]

print("\n" + "=" * 60)
print("[MONEY] BALANCE AFTER CLOSING LOSERS:")
print(f"   Total: ${balance:.2f} USDT")
print(f"   Available: ${available:.2f} USDT")
print("\n[CHART] REMAINING POSITIONS:", len(positions))

for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    pnl = float(p['unRealizedProfit'])
    pnl_pct = (pnl / float(p['notional'])) * 100 if float(p['notional']) != 0 else 0
    
    color = "[OK]" if pnl > 0 else "📉"
    print(f"   {color} {symbol}: {amt} @ ${entry:.4f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

print("\n[TARGET] TARGET: $2720 (double from $1360)")
print(f"   Current Capital: ${balance:.2f}")
print(f"   Needed: ${2720 - balance:.2f}")
print(f"   Progress: {(balance / 2720) * 100:.1f}%")
print("=" * 60)
