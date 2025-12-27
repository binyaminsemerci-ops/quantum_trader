#!/usr/bin/env python3
"""
Cancel ALL open orders on Binance Testnet
"""
import os
import asyncio
import aiohttp
import hmac
import hashlib
import time
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

USE_TESTNET = os.getenv("USE_BINANCE_TESTNET", "true").lower() == "true"

if USE_TESTNET:
    API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_TESTNET_SECRET_KEY")
    BASE_URL = "https://testnet.binancefuture.com"
else:
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    BASE_URL = "https://fapi.binance.com"

if not API_KEY or not SECRET_KEY:
    raise Exception("Missing Binance API credentials in .env file")

def sign_request(params: dict) -> str:
    """Sign request with HMAC SHA256"""
    query_string = urlencode(params)
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

async def get_open_orders():
    """Get all open orders"""
    async with aiohttp.ClientSession() as session:
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 10000
        }
        params['signature'] = sign_request(params)
        
        headers = {'X-MBX-APIKEY': API_KEY}
        
        url = f"{BASE_URL}/fapi/v1/openOrders"
        
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to get open orders: {resp.status} - {text}")
            
            return await resp.json()

async def cancel_order(symbol: str, order_id: int):
    """Cancel a specific order"""
    async with aiohttp.ClientSession() as session:
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 10000
        }
        params['signature'] = sign_request(params)
        
        headers = {'X-MBX-APIKEY': API_KEY}
        
        url = f"{BASE_URL}/fapi/v1/order"
        
        async with session.delete(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                return False, text
            
            return True, await resp.json()

async def main():
    print("\n" + "="*80)
    print("🔴 CANCELLING ALL OPEN ORDERS")
    print("="*80 + "\n")
    
    try:
        # Get all open orders
        print("📊 Fetching open orders...\n")
        orders = await get_open_orders()
        
        if not orders:
            print("✅ No open orders found!")
            return
        
        print(f"🔍 Found {len(orders)} open order(s):\n")
        
        cancelled = 0
        failed = 0
        
        for order in orders:
            symbol = order['symbol']
            order_id = order['orderId']
            order_type = order['type']
            side = order['side']
            price = order.get('stopPrice') or order.get('price', 'N/A')
            
            print(f"📍 {symbol:12s} | {order_type:15s} | {side:4s} | Price: ${price} | ID: {order_id}")
            print(f"   🔄 Cancelling...")
            
            success, result = await cancel_order(symbol, order_id)
            
            if success:
                print(f"   ✅ Cancelled!")
                cancelled += 1
            else:
                print(f"   ❌ Failed: {result}")
                failed += 1
            
            await asyncio.sleep(0.1)  # Small delay
        
        print(f"\n" + "="*80)
        print(f"✅ Cancelled {cancelled}/{len(orders)} orders")
        if failed > 0:
            print(f"❌ Failed: {failed} orders")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
