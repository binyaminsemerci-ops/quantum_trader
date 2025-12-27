import asyncio
import aiohttp
import time
import hmac
import hashlib
import os
from urllib.parse import urlencode

async def check_positions():
    api_key = os.environ['BINANCE_API_KEY']
    api_secret = os.environ['BINANCE_API_SECRET']
    
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp,
        'recvWindow': 5000
    }
    
    query_string = urlencode(params)
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params['signature'] = signature
    
    headers = {'X-MBX-APIKEY': api_key}
    
    async with aiohttp.ClientSession() as session:
        # Check position mode
        async with session.get(
            'https://testnet.binancefuture.com/fapi/v1/positionSide/dual',
            params=params,
            headers=headers
        ) as resp:
            mode = await resp.text()
            print(f"Position Mode: {mode}")
        
        # Check current positions
        params2 = {'timestamp': int(time.time() * 1000), 'recvWindow': 5000}
        query_string2 = urlencode(params2)
        signature2 = hmac.new(api_secret.encode(), query_string2.encode(), hashlib.sha256).hexdigest()
        params2['signature'] = signature2
        
        async with session.get(
            'https://testnet.binancefuture.com/fapi/v2/positionRisk',
            params=params2,
            headers=headers
        ) as resp:
            positions = await resp.text()
            print(f"\nCurrent Positions: {positions}")

asyncio.run(check_positions())
