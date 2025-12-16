#!/usr/bin/env python3
"""
Emergency script to close ALL open positions on Binance Testnet
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

async def get_open_positions():
    """Get all open positions"""
    async with aiohttp.ClientSession() as session:
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 10000
        }
        params['signature'] = sign_request(params)
        
        headers = {'X-MBX-APIKEY': API_KEY}
        
        async with session.get(
            f"{BASE_URL}/fapi/v3/positionRisk",
            params=params,
            headers=headers
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to get positions: {resp.status} {text}")
            
            positions = await resp.json()
            # Filter only positions with non-zero amount
            open_positions = [
                p for p in positions 
                if float(p['positionAmt']) != 0
            ]
            return open_positions

async def close_position(symbol: str, position_amt: float):
    """Close a single position using market order"""
    async with aiohttp.ClientSession() as session:
        # Determine side: if positionAmt > 0 (LONG), we SELL to close
        # if positionAmt < 0 (SHORT), we BUY to close
        side = "SELL" if position_amt > 0 else "BUY"
        quantity = abs(position_amt)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'reduceOnly': 'true',  # Important: only close existing position
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        params['signature'] = sign_request(params)
        
        headers = {'X-MBX-APIKEY': API_KEY}
        
        print(f"🔴 Closing {symbol}: {side} {quantity} (reduceOnly=true)")
        
        async with session.post(
            f"{BASE_URL}/fapi/v1/order",
            params=params,
            headers=headers
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✅ {symbol} closed successfully! OrderId: {result.get('orderId')}")
                return True
            else:
                text = await resp.text()
                print(f"❌ Failed to close {symbol}: {resp.status} {text}")
                return False

async def main():
    print("=" * 60)
    print("🚨 EMERGENCY POSITION CLOSER 🚨")
    print("=" * 60)
    print(f"Environment: {'TESTNET' if USE_TESTNET else 'MAINNET'}")
    print()
    
    # Get open positions
    print("📊 Fetching open positions...")
    positions = await get_open_positions()
    
    if not positions:
        print("✅ No open positions found!")
        return
    
    print(f"\n🔍 Found {len(positions)} open position(s):\n")
    
    total_pnl = 0
    for pos in positions:
        symbol = pos['symbol']
        position_amt = float(pos['positionAmt'])
        entry_price = float(pos['entryPrice'])
        mark_price = float(pos['markPrice'])
        unrealized_pnl = float(pos['unRealizedProfit'])
        
        side = "LONG" if position_amt > 0 else "SHORT"
        
        print(f"  {symbol:12s} {side:5s} | Qty: {abs(position_amt):,.4f} | "
              f"Entry: ${entry_price:.4f} | Mark: ${mark_price:.4f} | "
              f"PnL: ${unrealized_pnl:+,.2f}")
        
        total_pnl += unrealized_pnl
    
    print(f"\n💰 Total Unrealized PnL: ${total_pnl:+,.2f}")
    print()
    
    # Ask for confirmation
    response = input("⚠️  Are you sure you want to CLOSE ALL positions? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled by user")
        return
    
    print("\n🔴 Closing all positions...\n")
    
    # Close all positions
    success_count = 0
    for pos in positions:
        symbol = pos['symbol']
        position_amt = float(pos['positionAmt'])
        
        success = await close_position(symbol, position_amt)
        if success:
            success_count += 1
        
        # Small delay between orders
        await asyncio.sleep(0.5)
    
    print()
    print("=" * 60)
    print(f"✅ Closed {success_count}/{len(positions)} positions successfully")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
