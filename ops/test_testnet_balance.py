#!/usr/bin/env python3
import os
import sys

# Set credentials
os.environ["BINANCE_TESTNET_API_KEY"] = "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD"
os.environ["BINANCE_TESTNET_SECRET_KEY"] = "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"

from binance.client import Client

print("🔍 Testing Binance Testnet Futures API...")
print(f"API Key: {os.environ['BINANCE_TESTNET_API_KEY'][:20]}...")

try:
    client = Client(
        os.environ["BINANCE_TESTNET_API_KEY"],
        os.environ["BINANCE_TESTNET_SECRET_KEY"],
        testnet=True
    )
    client.FUTURES_URL = "https://testnet.binancefuture.com"
    
    print("\n✅ Client initialized")
    print(f"Futures URL: {client.FUTURES_URL}")
    
    # Test account endpoint
    print("\n📊 Fetching account info...")
    account = client.futures_account()
    
    print(f"\n💰 ACCOUNT BALANCE:")
    print(f"  Available Balance: {account['availableBalance']} USDT")
    print(f"  Total Wallet Balance: {account['totalWalletBalance']} USDT")
    print(f"  Total Cross Wallet: {account['totalCrossWalletBalance']} USDT")
    
    # Test balance endpoint
    print("\n📊 Fetching detailed balances...")
    balances = client.futures_account_balance()
    for b in balances:
        if b["asset"] == "USDT":
            print(f"  USDT Asset:")
            print(f"    Available: {b['availableBalance']} USDT")
            print(f"    Balance: {b['balance']} USDT")
            
    # Test positions
    print("\n📍 Open Positions:")
    positions = client.futures_position_information()
    open_pos = [p for p in positions if float(p['positionAmt']) != 0]
    if open_pos:
        for p in open_pos:
            print(f"  {p['symbol']}: {p['positionAmt']} @ {p['entryPrice']}")
    else:
        print("  No open positions")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Test complete!")
