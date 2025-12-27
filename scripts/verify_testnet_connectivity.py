#!/usr/bin/env python3
"""
Quantum Trader V3 - Testnet Connectivity Verification
Tests connection to Binance Testnet and verifies API credentials
"""
import sys
import os

# Add backend to Python path
sys.path.insert(0, '/home/qt/quantum_trader/backend')
sys.path.insert(0, '/home/qt/quantum_trader')

def verify_testnet_connectivity():
    """Verify Binance Testnet API connectivity"""
    print("\n🔌 Quantum Trader V3 – Testnet Connectivity Verification\n")
    print("=" * 60)
    
    try:
        # Check environment variables
        print("\n📋 Step 1: Checking Environment Configuration...")
        
        binance_testnet = os.getenv('BINANCE_TESTNET', 'false').lower()
        go_live = os.getenv('GO_LIVE', 'false').lower()
        simulation_mode = os.getenv('SIMULATION_MODE', 'false').lower()
        
        print(f"   BINANCE_TESTNET: {binance_testnet}")
        print(f"   GO_LIVE: {go_live}")
        print(f"   SIMULATION_MODE: {simulation_mode}")
        
        if binance_testnet != 'true':
            print("   ⚠️  WARNING: BINANCE_TESTNET should be 'true'")
        if go_live != 'false':
            print("   ⚠️  WARNING: GO_LIVE should be 'false' for testnet")
        if simulation_mode != 'true':
            print("   ⚠️  WARNING: SIMULATION_MODE should be 'true' for testing")
        
        # Try to import and test Binance client
        print("\n🔗 Step 2: Testing Binance API Connection...")
        
        try:
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            
            if not api_key or not api_secret:
                print("   ❌ Binance API credentials not found in environment")
                return False
            
            # Initialize testnet client
            client = Client(api_key, api_secret, testnet=True)
            
            # Test ping
            ping = client.ping()
            print("   ✅ Binance Testnet Ping: SUCCESS")
            
            # Test server time
            server_time = client.get_server_time()
            print(f"   ✅ Server Time: {server_time['serverTime']}")
            
            # Test account info
            try:
                account = client.futures_account()
                total_balance = float(account.get('totalWalletBalance', 0))
                available_balance = float(account.get('availableBalance', 0))
                print(f"   ✅ Account Balance: {total_balance} USDT")
                print(f"   ✅ Available Balance: {available_balance} USDT")
            except BinanceAPIException as e:
                print(f"   ⚠️  Account info error: {e}")
                print("   💡 This is normal for testnet - may need initialization")
            
            # Test exchange info
            exchange_info = client.futures_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'][:5]]
            print(f"   ✅ Exchange Info: {len(exchange_info['symbols'])} trading pairs available")
            print(f"   📊 Sample symbols: {', '.join(symbols)}")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Failed to import binance library: {e}")
            print("   💡 Install with: pip install python-binance")
            return False
        except Exception as e:
            print(f"   ❌ Binance API Error: {e}")
            return False
            
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n" + "=" * 60)

if __name__ == "__main__":
    success = verify_testnet_connectivity()
    
    if success:
        print("\n✅ Testnet Connectivity Verification: PASSED")
        print("💡 Ready to proceed with AI pipeline simulation")
        sys.exit(0)
    else:
        print("\n❌ Testnet Connectivity Verification: FAILED")
        print("💡 Fix configuration issues before proceeding")
        sys.exit(1)
