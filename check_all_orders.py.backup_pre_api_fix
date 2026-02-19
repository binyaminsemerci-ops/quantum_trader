import asyncio
import aiohttp
import os

async def check():
    api_key = os.getenv('BINANCE_TEST_API_KEY')
    
    async with aiohttp.ClientSession() as session:
        url = 'https://testnet.binancefuture.com/fapi/v1/openOrders'
        headers = {'X-MBX-APIKEY': api_key}
        params = {'symbol': 'SOLUSDT'}
        
        async with session.get(url, headers=headers, params=params) as resp:
            orders = await resp.json()
            
            print(f'\n📋 ALL SOLUSDT ORDERS ({len(orders)}):')
            for o in orders:
                order_type = o['type']
                side = o['side']
                stop_price = float(o.get('stopPrice', 0))
                price = float(o.get('price', 0))
                target = stop_price if stop_price > 0 else price
                
                print(f'\n  {order_type:25s} {side:5s}')
                print(f'    Order ID: {o["orderId"]}')
                print(f'    Price/Stop: ${target:.5f}')
                print(f'    Status: {o["status"]}')
                print(f'    Time: {o.get("updateTime", o["time"])}')

if __name__ == "__main__":
    asyncio.run(check())
