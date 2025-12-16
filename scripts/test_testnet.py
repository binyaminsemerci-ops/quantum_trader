"""
Test Binance Futures Testnet Connection
Quick verification that API keys work correctly
"""
from binance.client import Client as BinanceClient
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import load_config

def test_testnet():
    print("=" * 80)
    print("[TEST_TUBE] TESTING BINANCE FUTURES TESTNET")
    print("=" * 80)
    
    cfg = load_config()
    
    print(f"\n🔑 API Key: {cfg.binance_testnet_api_key[:20]}...")
    print(f"🔐 Secret: {cfg.binance_testnet_secret_key[:20]}...")
    
    try:
        print("\n[SIGNAL] Connecting to testnet...")
        client = BinanceClient(
            api_key=cfg.binance_testnet_api_key,
            api_secret=cfg.binance_testnet_secret_key,
            testnet=True
        )
        
        # Test 1: Account info
        print("\n[OK] TEST 1: Account Information")
        account = client.futures_account()
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        
        print(f"  Total Balance: ${balance:,.2f} USDT")
        print(f"  Available: ${available:,.2f} USDT")
        print(f"  Can Trade: {account.get('canTrade', False)}")
        
        # Test 2: Get exchange info
        print("\n[OK] TEST 2: Exchange Information")
        exchange_info = client.futures_exchange_info()
        symbols = [s for s in exchange_info['symbols'] if s['symbol'].endswith('USDT')][:5]
        print(f"  Available Symbols: {len(symbols)} (showing first 5)")
        for s in symbols:
            print(f"    - {s['symbol']}")
        
        # Test 3: Get current price
        print("\n[OK] TEST 3: Market Data")
        btc_price = client.futures_symbol_ticker(symbol='BTCUSDT')
        print(f"  BTCUSDT Price: ${float(btc_price['price']):,.2f}")
        
        # Test 4: Check positions
        print("\n[OK] TEST 4: Open Positions")
        positions = client.futures_position_information()
        active = [p for p in positions if float(p['positionAmt']) != 0]
        print(f"  Active Positions: {len(active)}")
        
        if active:
            for pos in active[:5]:
                symbol = pos['symbol']
                amt = float(pos['positionAmt'])
                entry = float(pos['entryPrice'])
                pnl = float(pos['unRealizedProfit'])
                print(f"    {symbol}: {amt:+.4f} @ ${entry:.2f} | P&L: ${pnl:+.2f}")
        
        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80)
        print("\n[ROCKET] Ready to start testnet trading!")
        print("\n📌 Next: python scripts/testnet_trading.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n[CLIPBOARD] CHECKLIST:")
        print("  ☐ API keys created on testnet.binancefuture.com")
        print("  ☐ 'Enable Futures' is checked")
        print("  ☐ IP restrictions = Unrestricted (or includes your IP)")
        print("  ☐ API key status = ENABLED")
        print("  ☐ Wait 1-2 minutes after creating keys")
        
        return False

if __name__ == "__main__":
    success = test_testnet()
    sys.exit(0 if success else 1)
