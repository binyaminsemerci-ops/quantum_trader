#!/usr/bin/env python3
"""Check recent trades to see what happened"""
import os
from binance.client import Client
from datetime import datetime, timezone

api_key = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")

client = Client(api_key, api_secret, testnet=True)

# Get ALL trades
trades = client.futures_account_trades(limit=500)

print(f"\n📜 ALL TRADES (Total: {len(trades)}):")
print("=" * 120)
print(f"{'Time':<20} {'Symbol':<12} {'Side':<5} {'Qty':<12} {'Price':<12} {'Realized PnL':<15} {'Commission':<12}")
print("=" * 120)

total_pnl = 0
total_commission = 0

for trade in trades:
    time = datetime.fromtimestamp(int(trade['time'])/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    symbol = trade['symbol']
    side = trade['side']
    qty = float(trade['qty'])
    price = float(trade['price'])
    realized_pnl = float(trade['realizedPnl'])
    commission = float(trade['commission'])
    
    total_pnl += realized_pnl
    total_commission += commission
    
    pnl_str = f"${realized_pnl:+.2f}" if realized_pnl != 0 else "-"
    comm_str = f"${commission:.4f}"
    
    print(f"{time} {symbol:<12} {side:<5} {qty:>11.4f} ${price:>10.4f} {pnl_str:>14} {comm_str:>11}")

print("=" * 120)
print(f"Total Realized PnL: ${total_pnl:+.2f}")
print(f"Total Commission: ${total_commission:.4f}")
print("=" * 120)
