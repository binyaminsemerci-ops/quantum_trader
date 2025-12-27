import asyncio, time, hmac, hashlib, aiohttp, os
from urllib.parse import urlencode

async def main():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    timestamp = int(time.time() * 1000)
    params = {'dualSidePosition': 'true', 'timestamp': timestamp, 'recvWindow': 5000}
    query = urlencode(params)
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params['signature'] = sig
    headers = {'X-MBX-APIKEY': api_key}
    async with aiohttp.ClientSession() as session:
        async with session.post('https://testnet.binancefuture.com/fapi/v1/positionSide/dual', params=params, headers=headers) as r:
            print(f'{r.status}: {await r.text()}')

asyncio.run(main())
