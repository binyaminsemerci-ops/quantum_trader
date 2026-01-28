#!/usr/bin/env python3
"""
Test Binance Testnet Balance - Using Execution Service Credentials
"""
import os
import sys
from binance.client import Client

# Load execution service credentials
API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")

if not API_KEY or not API_SECRET:
    print("❌ Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_SECRET_KEY")
    sys.exit(1)

print(f"Using API Key: {API_KEY[:20]}...")
print(f"Using Secret: {API_SECRET[:20]}...")

# Initialize client
client = Client(API_KEY, API_SECRET, testnet=True)
client.FUTURES_URL = "https://testnet.binancefuture.com"

# Get account info
try:
    account = client.futures_account()
    total_balance = float(account["totalWalletBalance"])
    available_balance = float(account["availableBalance"])
    can_trade = account["canTrade"]
    
    print(f"\n✅ Binance Futures Testnet Connection Success!")
    print(f"Total Balance: {total_balance:.2f} USDT")
    print(f"Available Balance: {available_balance:.2f} USDT")
    print(f"Can Trade: {can_trade}")
    
    # Get open positions
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p["positionAmt"]) != 0]
    print(f"Open Positions: {len(open_positions)}")
    
    if open_positions:
        print("\nOpen Positions:")
        for pos in open_positions[:5]:
            symbol = pos["symbol"]
            amt = float(pos["positionAmt"])
            entry_price = float(pos["entryPrice"])
            unrealized_pnl = float(pos["unRealizedProfit"])
            print(f"  {symbol}: {amt:+.4f} @ ${entry_price:.4f} | PnL: ${unrealized_pnl:+.2f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
