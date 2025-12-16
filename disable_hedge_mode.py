"""
Disable Hedge Mode on Binance Testnet Futures Account
This prevents LONG and SHORT positions simultaneously on same symbol
"""
import asyncio
import os
import aiohttp
import time
import hmac
import hashlib
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = "https://testnet.binancefuture.com"

async def disable_hedge_mode():
    """Disable Hedge Mode (return to one-way mode)"""
    
    if not API_KEY or not API_SECRET:
        print("❌ Missing BINANCE_API_KEY or BINANCE_API_SECRET")
        return
    
    async with aiohttp.ClientSession() as session:
        # Check current mode first
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": 5000
        }
        
        query_string = urlencode(params)
        signature = hmac.new(
            API_SECRET.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        url = f"{BASE_URL}/fapi/v1/positionSide/dual?{query_string}&signature={signature}"
        headers = {"X-MBX-APIKEY": API_KEY}
        
        try:
            async with session.get(url, headers=headers) as resp:
                data = await resp.json()
                print(f"📊 Current Position Mode: {data}")
                
                if data.get("dualSidePosition") is False:
                    print("✅ Hedge Mode ALREADY DISABLED (One-Way Mode Active)!")
                    return
                
                print("🔄 Disabling Hedge Mode (switching to One-Way Mode)...")
                
        except Exception as e:
            print(f"⚠️ Could not check current mode: {e}")
        
        # Disable Hedge Mode
        timestamp = int(time.time() * 1000)
        params = {
            "dualSidePosition": "false",
            "timestamp": timestamp,
            "recvWindow": 5000
        }
        
        query_string = urlencode(params)
        signature = hmac.new(
            API_SECRET.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        url = f"{BASE_URL}/fapi/v1/positionSide/dual?{query_string}&signature={signature}"
        
        try:
            async with session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Hedge Mode DISABLED: {data}")
                    print("\n🎯 One-Way Mode Active:")
                    print("   ✓ Cannot open LONG and SHORT simultaneously")
                    print("   ✓ New order in opposite direction will CLOSE existing position")
                    print("   ✓ positionSide will be 'BOTH'")
                    print("\n⚠️ IMPORTANT: Close ALL open positions before switching modes!")
                else:
                    text = await resp.text()
                    print(f"❌ Failed to disable Hedge Mode ({resp.status}): {text}")
                    if "PositionSide is not BOTH" in text:
                        print("\n⚠️ You have open positions with LONG/SHORT sides.")
                        print("   Please CLOSE ALL positions first, then try again.")
                    
        except Exception as e:
            print(f"❌ Error disabling Hedge Mode: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🛑 DISABLE HEDGE MODE - Binance Futures")
    print("=" * 60)
    print("\n⚠️ WARNING: You MUST close ALL open positions first!")
    print("   If you have open positions, this will FAIL.\n")
    
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() == "yes":
        asyncio.run(disable_hedge_mode())
    else:
        print("❌ Cancelled")
