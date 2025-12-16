"""
Diagnostic Script: Check Current Position Status and Hedge Mode
"""
import asyncio
import os
from dotenv import load_dotenv
from backend.integrations.exchanges.binance_adapter import BinanceAdapter
from binance.client import Client

load_dotenv()

async def check_status():
    """Check current positions and hedge mode status."""
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ Missing BINANCE_API_KEY or BINANCE_API_SECRET")
        return
    
    print("=" * 80)
    print("🔍 POSITION & HEDGE MODE DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Initialize client
    client = Client(api_key, api_secret, testnet=True)
    adapter = BinanceAdapter(client=client)
    
    try:
        # 1. Check hedge mode status
        print("📊 CHECKING HEDGE MODE STATUS...")
        print("-" * 80)
        
        import time
        import hmac
        import hashlib
        from urllib.parse import urlencode
        import aiohttp
        
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp, "recvWindow": 5000}
        query_string = urlencode(params)
        signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        url = f"https://testnet.binancefuture.com/fapi/v1/positionSide/dual?{query_string}&signature={signature}"
        
        async with aiohttp.ClientSession() as session:
            headers = {"X-MBX-APIKEY": api_key}
            async with session.get(url, headers=headers) as resp:
                data = await resp.json()
                hedge_mode = data.get("dualSidePosition", False)
                
                if hedge_mode:
                    print("🚨 HEDGE MODE: ✅ ENABLED (Dual-Side Position)")
                    print("   ⚠️  This allows BOTH long AND short positions simultaneously!")
                    print("   ⚠️  This is WHY you're seeing XRP long and short at the same time!")
                else:
                    print("✅ HEDGE MODE: ❌ DISABLED (One-Way Mode)")
                    print("   ✓ Only one direction allowed per symbol")
        
        print()
        
        # 2. Get all open positions
        print("📊 CHECKING OPEN POSITIONS...")
        print("-" * 80)
        
        positions = await adapter.get_open_positions()
        
        if not positions:
            print("✅ No open positions")
        else:
            print(f"📊 Found {len(positions)} open positions:\n")
            
            # Group by symbol to detect conflicts
            by_symbol = {}
            for pos in positions:
                symbol = pos.symbol
                if symbol not in by_symbol:
                    by_symbol[symbol] = []
                by_symbol[symbol].append(pos)
            
            # Display positions
            for symbol, symbol_positions in by_symbol.items():
                print(f"🔹 {symbol}:")
                
                if len(symbol_positions) > 1:
                    print(f"   🚨 CONFLICT DETECTED: {len(symbol_positions)} positions on same symbol!")
                
                for i, pos in enumerate(symbol_positions, 1):
                    side_emoji = "🟢" if pos.side.lower() == "long" else "🔴"
                    position_side = getattr(pos, 'position_side', None)
                    position_side_str = f" ({position_side.value})" if position_side else ""
                    
                    print(f"   {side_emoji} Position {i}: {pos.side.upper()}{position_side_str}")
                    print(f"      Quantity: {pos.quantity}")
                    print(f"      Entry: {pos.entry_price}")
                    print(f"      Mark: {pos.mark_price}")
                    print(f"      PNL: {pos.unrealized_pnl}")
                    print(f"      Leverage: {pos.leverage}x")
                
                print()
        
        # 3. Summary and recommendations
        print("=" * 80)
        print("📋 SUMMARY & RECOMMENDATIONS")
        print("=" * 80)
        
        if hedge_mode:
            conflicts = [s for s, positions in by_symbol.items() if len(positions) > 1]
            
            print("🚨 CRITICAL ISSUE DETECTED:")
            print(f"   • Hedge mode is ENABLED on exchange")
            print(f"   • System is designed for ONE-WAY mode")
            
            if conflicts:
                print(f"   • {len(conflicts)} symbols have conflicting positions:")
                for symbol in conflicts:
                    print(f"     - {symbol}: {len(by_symbol[symbol])} positions")
            
            print("\n✅ RECOMMENDED FIX:")
            print("   1. Close ALL open positions")
            print("   2. Run: python disable_hedge_mode.py")
            print("   3. Restart trading bot")
            print("\n⚠️  Alternative (NOT RECOMMENDED):")
            print("   • Set QT_ALLOW_HEDGING=true in .env")
            print("   • This allows hedge mode but requires careful strategy design")
        else:
            print("✅ Configuration looks good!")
            print("   • Hedge mode is disabled")
            print("   • System should prevent conflicting positions")
            
            if positions:
                print(f"\n📊 {len(positions)} open positions detected")
                print("   Review them above for any issues")
        
        print()
        
    except Exception as e:
        print(f"❌ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_status())
