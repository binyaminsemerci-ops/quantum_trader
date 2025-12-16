#!/usr/bin/env python3
from binance.client import Client
import os
from datetime import datetime

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print("="*70)
print("OPUSDT TRADE HISTORY (Last 20 trades)")
print("="*70)

trades = client.futures_account_trades(symbol='OPUSDT', limit=20)

for trade in reversed(trades):
    timestamp = datetime.fromtimestamp(trade['time'] / 1000)
    side = trade['side']
    qty = trade['qty']
    price = trade['price']
    pnl = float(trade['realizedPnl'])
    commission = trade['commission']
    
    pnl_symbol = "💰" if pnl > 0 else "❌" if pnl < 0 else "➖"
    
    print(f"\n{pnl_symbol} {timestamp.strftime('%H:%M:%S')}: {side} {qty} @ ${price}")
    print(f"   Realized PnL: ${pnl:.2f}")
    print(f"   Commission: {commission} {trade['commissionAsset']}")
