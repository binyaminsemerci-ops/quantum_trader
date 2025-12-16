"""
Enable Hedge Mode on Binance Testnet Futures Account
This allows LONG and SHORT positions simultaneously on same symbol
"""
import asyncio
import os
import aiohttp
import time
import hmac
import hashlib
from urllib.parse import urlencode

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = "https://testnet.binancefuture.com"

async def enable_hedge_mode():
    """Enable Hedge Mode (dual-side position)"""
    
    if not API_KEY or not API_SECRET:
        print("❌ Missing BINANCE_API_KEY or BINANCE_API_SECRET")
        return
    
    # Check current position mode
    async with aiohttp.ClientSession() as session:
        # Get current mode
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
                
                if data.get("dualSidePosition") is True:
                    print("✅ Hedge Mode ALREADY ENABLED!")
                    return
                
                print("🔄 Enabling Hedge Mode...")
                
        except Exception as e:
            print(f"⚠️ Could not check current mode: {e}")
        
        # Enable Hedge Mode
        timestamp = int(time.time() * 1000)
        params = {
            "dualSidePosition": "true",
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
                    print(f"✅ Hedge Mode ENABLED: {data}")
                    print("\n🎯 You can now open LONG and SHORT positions simultaneously!")
                    print("🔥 System will use positionSide=LONG/SHORT in orders")
                else:
                    text = await resp.text()
                    print(f"❌ Failed to enable Hedge Mode ({resp.status}): {text}")
                    
        except Exception as e:
            print(f"❌ Error enabling Hedge Mode: {e}")

if __name__ == "__main__":
    asyncio.run(enable_hedge_mode())
