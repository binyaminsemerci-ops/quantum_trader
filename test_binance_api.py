#!/usr/bin/env python3
import asyncio
import time
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
import os

async def test_api():
    api_key = os.getenv('BINANCE_API_KEY', '')
    secret = os.getenv('BINANCE_API_SECRET', '')
    
    print(f'API Key: {api_key[:10]}... (len={len(api_key)})')
    print(f'Secret: {secret[:10]}... (len={len(secret)})')
    
    # Test 1: Get exchange info (no signature needed)
    base_url = 'https://testnet.binancefuture.com'
    
    async with aiohttp.ClientSession() as session:
        # Unsigned request
        async with session.get(f'{base_url}/fapi/v1/exchangeInfo') as resp:
            print(f'\n✅ Exchange Info: {resp.status}')
        
        # Signed request - get account info
        timestamp = str(int(time.time() * 1000))
        params = {'timestamp': timestamp, 'recvWindow': '5000'}
        query_string = urlencode(params)
        signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        
        headers = {'X-MBX-APIKEY': api_key}
        url = f'{base_url}/fapi/v2/account?{query_string}&signature={signature}'
        
        print(f'\nTesting signed request:')
        print(f'URL: {url[:100]}...')
        print(f'Signature: {signature}')
        
        async with session.get(url, headers=headers) as resp:
            text = await resp.text()
            print(f'\n{"✅" if resp.status == 200 else "❌"} Account Info: {resp.status}')
            print(f'Response: {text[:200]}')

asyncio.run(test_api())
