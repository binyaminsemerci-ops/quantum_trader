#!/usr/bin/env python3
"""
Test multiple scenarios to find why Binance Testnet fails
"""
import os
from binance.client import Client

api_key = os.getenv('BINANCE_API_KEY', 'IsY3mFpko7Z8joZr8clWwpJZuZcFdAtnDBy4g4ULQu827Gf6kJPAPS9VyVOVrR0r')
api_secret = os.getenv('BINANCE_API_SECRET', '')

print("=" * 70)
print("DEBUGGING BINANCE TESTNET CONNECTION ISSUES")
print("=" * 70)

# Test 1: Check key format
print("\n[TEST 1] API Key Format Check")
print(f"API Key length: {len(api_key)} chars")
print(f"API Key starts with: {api_key[:10]}...")
print(f"API Secret length: {len(api_secret)} chars")
print(f"API Secret starts with: {'*' * 10 if api_secret else 'MISSING!'}")

if not api_secret:
    print("\n❌ PROBLEM FOUND: API SECRET IS EMPTY!")
    print("Solution: Set BINANCE_API_SECRET in .env file")
    exit(1)

# Test 2: Try different testnet endpoints
print("\n[TEST 2] Testing Different Endpoints")

# Try with tld='com'
try:
    print("Trying tld='com'...")
    client = Client(api_key, api_secret, testnet=True, tld='com')
    server_time = client.get_server_time()
    print(f"✅ Server time with tld=com: {server_time['serverTime']}")
except Exception as e:
    print(f"❌ Failed with tld=com: {e}")

# Try futures_coin_symbol_ticker instead
try:
    print("\nTrying futures API directly...")
    client = Client(api_key, api_secret, testnet=True)
    
    # Skip account() call, try futures directly
    ticker = client.futures_symbol_ticker(symbol='BTCUSDT')
    print(f"✅ Futures ticker works: BTCUSDT @ ${ticker['price']}")
    
    # Now try account
    futures_account = client.futures_account()
    print(f"✅ Futures account works!")
    print(f"   Balance: {futures_account['totalWalletBalance']} USDT")
    
except Exception as e:
    print(f"❌ Futures API failed: {e}")
    error_code = str(e)
    
    if '-2015' in error_code:
        print("\n🔍 ERROR CODE -2015: Invalid API-key, IP, or permissions")
        print("   Possible causes:")
        print("   1. API key expired (Testnet keys expire after 90 days)")
        print("   2. IP restriction enabled (VPS IP not whitelisted)")
        print("   3. Futures trading not enabled for this API key")
        print("   4. Using wrong testnet URL")
    elif '-2019' in error_code:
        print("\n🔍 ERROR CODE -2019: Margin insufficient")
        print("   Your API key works, but account has no USDT!")
        print("   Get testnet funds: https://testnet.binancefuture.com")

print("\n" + "=" * 70)
print("POSSIBLE SOLUTIONS:")
print("=" * 70)
print("1. Generate NEW API keys at: https://testnet.binancefuture.com")
print("   - Make sure to ENABLE FUTURES trading")
print("   - Don't set IP restriction (or add VPS IP: 46.224.116.254)")
print("2. Get testnet USDT: Click 'Get Test Funds' button")
print("3. Verify keys don't have special characters that break .env parsing")
print("4. Consider using Paper Trading mode for development")
print("=" * 70)
