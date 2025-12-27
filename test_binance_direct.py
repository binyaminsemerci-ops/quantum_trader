#!/usr/bin/env python3
"""
Test Binance Testnet connection directly
"""
import os
from dotenv import load_dotenv
from binance.client import Client

# Load .env file
load_dotenv()

print("=" * 70)
print("BINANCE TESTNET CONNECTION TEST")
print("=" * 70)

# Get API keys from environment
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

print(f"\n📊 Configuration:")
print(f"   API Key: {api_key[:20]}..." if api_key else "   API Key: ❌ MISSING")
print(f"   Testnet: {testnet}")

if not api_key or not api_secret:
    print("\n❌ API keys missing in environment!")
    print("   Run: .venv/Scripts/Activate.ps1")
    exit(1)

try:
    # Create client
    client = Client(api_key, api_secret, testnet=testnet)
    
    # Test 1: Account info
    print("\n✅ Test 1: Account Info")
    account = client.futures_account()
    balance = float(account['totalWalletBalance'])
    print(f"   Balance: ${balance:,.2f}")
    
    # Test 2: Open positions
    print("\n✅ Test 2: Open Positions")
    positions = client.futures_position_information()
    
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    print(f"   Found: {len(open_positions)} open positions")
    
    for pos in open_positions:
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        pnl = float(pos['unRealizedProfit'])
        
        side = "LONG" if amt > 0 else "SHORT"
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        
        print(f"   {pnl_emoji} {symbol:12} {side:6} {amt:>12,.2f} @ ${entry:>8,.2f}  PnL: ${pnl:>8,.2f}")
    
    print("\n" + "=" * 70)
    print("✅ BINANCE TESTNET CONNECTION WORKING!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n   Possible causes:")
    print("   - API keys not whitelisted for this IP")
    print("   - API keys expired or invalid")
    print("   - Testnet API temporarily down")
    print("=" * 70)
