#!/usr/bin/env python3
import os, redis, time, hmac, hashlib
import aiohttp, asyncio

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"
BASE_URL = "https://testnet.binancefuture.com"

async def fetch_account():
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    
    url = f"{BASE_URL}/fapi/v2/account?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": API_KEY}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                print(f"ERROR {response.status}: {error}")
                return None

async def sync_positions():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    print("Fetching from Binance testnet...")
    
    data = await fetch_account()
    if not data:
        return
    
    positions = data.get("positions", [])
    active_symbols = set()
    
    for pos in positions:
        amt = float(pos["positionAmt"])
        if amt == 0:
            continue
        
        symbol = pos["symbol"]
        active_symbols.add(symbol)
        
        pos_key = f"quantum:position:{symbol}"
        side = "LONG" if amt > 0 else "SHORT"
        
        entry_price = pos["entryPrice"]
        unrealized_pnl = pos["unRealizedProfit"]
        leverage = pos["leverage"]
        
        position_data = {
            "symbol": symbol,
            "side": side,
            "quantity": str(abs(amt)),
            "entry_price": str(entry_price),
            "unrealized_pnl": str(unrealized_pnl),
            "leverage": str(leverage),
            "last_sync": str(int(time.time()))
        }
        
        r.hset(pos_key, mapping=position_data)
        print(f"Active: {symbol} {side} {abs(amt):.2f} @ {entry_price}")
    
    # Remove ghost positions
    all_pos_keys = r.keys("quantum:position:*")
    removed = 0
    for key in all_pos_keys:
        symbol = key.split(":")[-1]
        if symbol not in active_symbols:
            r.delete(key)
            print(f"Removed ghost: {symbol}")
            removed += 1
    
    print(f"\nDone: {len(active_symbols)} active, {removed} removed")

if __name__ == "__main__":
    asyncio.run(sync_positions())
