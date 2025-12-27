"""
Verify Binance Testnet Connection
Tests if backend can connect to Binance Testnet and fetch real data.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from binance.client import Client
from backend.integrations.exchanges.binance_adapter import BinanceAdapter

async def main():
    print("=" * 60)
    print("BINANCE TESTNET VERIFICATION")
    print("=" * 60)
    
    # Check env vars
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
    qt_testnet = os.getenv("QT_EXECUTION_BINANCE_TESTNET", "false").lower() == "true"
    
    print(f"\n1. Environment Variables:")
    print(f"   BINANCE_API_KEY: {'✓ Set' if api_key else '✗ Missing'}")
    print(f"   BINANCE_API_SECRET: {'✓ Set' if api_secret else '✗ Missing'}")
    print(f"   BINANCE_USE_TESTNET: {use_testnet}")
    print(f"   QT_EXECUTION_BINANCE_TESTNET: {qt_testnet}")
    
    if not api_key or not api_secret:
        print("\n❌ ERROR: Binance API credentials not set in .env")
        return
    
    # Initialize client
    print(f"\n2. Creating Binance Client...")
    try:
        client = Client(api_key, api_secret, testnet=use_testnet)
        
        if use_testnet:
            client.API_URL = 'https://testnet.binancefuture.com/fapi'
            print(f"   ✓ Testnet URL: {client.API_URL}")
        else:
            print(f"   ⚠ Production mode (not testnet)")
        
        adapter = BinanceAdapter(client, testnet=use_testnet)
        print(f"   ✓ BinanceAdapter created")
        
    except Exception as e:
        print(f"   ❌ Failed to create client: {e}")
        return
    
    # Test connection - get account info
    print(f"\n3. Testing Connection (Account Info)...")
    try:
        account_info = await asyncio.to_thread(client.futures_account)
        print(f"   ✓ Connected successfully")
        print(f"   Total Wallet Balance: {account_info.get('totalWalletBalance', 0)} USDT")
        print(f"   Available Balance: {account_info.get('availableBalance', 0)} USDT")
    except Exception as e:
        print(f"   ❌ Failed to get account info: {e}")
        return
    
    # Test balances
    print(f"\n4. Fetching Balances...")
    try:
        balances = await adapter.get_balances()
        print(f"   ✓ Found {len(balances)} balances")
        for balance in balances[:5]:  # Show first 5
            if balance.total > 0:
                print(f"   - {balance.asset}: {balance.total} (free: {balance.free})")
    except Exception as e:
        print(f"   ❌ Failed to get balances: {e}")
    
    # Test positions
    print(f"\n5. Fetching Open Positions...")
    try:
        positions = await adapter.get_open_positions()
        print(f"   ✓ Found {len(positions)} positions")
        active_count = 0
        for pos in positions:
            if pos.quantity != 0:
                active_count += 1
                print(f"   - {pos.symbol}: {pos.side.value} {pos.quantity} @ {pos.entry_price}")
                print(f"     Mark: {pos.mark_price}, PnL: {pos.unrealized_pnl} USDT")
        print(f"   Active positions: {active_count}/{len(positions)}")
    except Exception as e:
        print(f"   ❌ Failed to get positions: {e}")
    
    print(f"\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
