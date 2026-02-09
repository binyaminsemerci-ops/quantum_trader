#!/usr/bin/env python3
"""
Binance Proof of Trade - Direct API Query
Fetches hard evidence from Binance Futures Testnet
"""
import os
import sys
sys.path.insert(0, "/home/qt/quantum_trader")

from binance.client import Client
from datetime import datetime

# Get Binance client (Intent Executor uses TESTNET_ prefix)
api_key = os.getenv("BINANCE_TESTNET_API_KEY") or os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_API_SECRET") or os.getenv("BINANCE_API_SECRET")
testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

client = Client(api_key, api_secret, testnet=testnet)

print("=" * 80)
print("🔍 BINANCE PROOF OF TRADE - DIRECT API QUERY")
print("=" * 80)
print(f"Testnet Mode: {testnet}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print()

# Get account info
account = client.futures_account()
print("📊 ACCOUNT BALANCE:")
print(f"Total Wallet Balance: ${float(account['totalWalletBalance']):.2f} USDT")
print(f"Total Unrealized PnL: ${float(account['totalUnrealizedProfit']):.2f} USDT")
print(f"Available Balance: ${float(account['availableBalance']):.2f} USDT")
print()

# Get positions
positions = client.futures_position_information()
active_positions = [p for p in positions if float(p['positionAmt']) != 0]

print("📍 OPEN POSITIONS:")
if not active_positions:
    print("  No open positions")
else:
    for pos in active_positions:
        amt = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        mark = float(pos['markPrice'])
        pnl = float(pos['unRealizedProfit'])
        leverage = int(pos.get('leverage', 1))
        
        print(f"  Symbol: {pos['symbol']}")
        print(f"  Amount: {amt:.8f} ({'LONG' if amt > 0 else 'SHORT'})")
        print(f"  Entry Price: ${entry:.2f}")
        print(f"  Mark Price: ${mark:.2f}")
        print(f"  Unrealized PnL: ${pnl:.2f}")
        print(f"  Leverage: {leverage}x")
        print(f"  Notional: ${abs(amt * mark):.2f}")
        print()

# Get recent trades for BTCUSDT
print("📜 RECENT BTCUSDT TRADES (Last 10):")
try:
    trades = client.futures_account_trades(symbol="BTCUSDT", limit=10)
    trades.reverse()  # Most recent first
    for trade in trades:
        time_str = datetime.fromtimestamp(int(trade['time'])/1000).strftime('%Y-%m-%d %H:%M:%S')
        side = trade['side']
        qty = trade['qty']
        price = trade['price']
        commission = trade['commission']
        realized_pnl = trade.get('realizedPnl', '0')
        order_id = trade['orderId']
        print(f"  {time_str} | Order #{order_id} | {side} {qty} @ ${price} | Fee: ${commission} | PnL: ${realized_pnl}")
except Exception as e:
    print(f"  Error: {e}")
print()

# Get order details for order 12154300118
print("🎯 ORDER 12154300118 DETAILS:")
try:
    order = client.futures_get_order(symbol="BTCUSDT", orderId=12154300118)
    print(f"  Order ID: {order['orderId']}")
    print(f"  Client Order ID: {order['clientOrderId']}")
    print(f"  Symbol: {order['symbol']}")
    print(f"  Side: {order['side']}")
    print(f"  Type: {order['type']}")
    print(f"  Status: {order['status']}")
    print(f"  Quantity: {order['origQty']}")
    print(f"  Executed Qty: {order['executedQty']}")
    print(f"  Price: ${order['price']}")
    print(f"  Avg Price: ${order['avgPrice']}")
    print(f"  Time: {datetime.fromtimestamp(int(order['time'])/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Update Time: {datetime.fromtimestamp(int(order['updateTime'])/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Reduce Only: {order['reduceOnly']}")
except Exception as e:
    print(f"  Error fetching order: {e}")
print()

print("=" * 80)
