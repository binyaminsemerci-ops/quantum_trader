#!/usr/bin/env python3
"""Check SOLUSDT position and TP/SL orders"""

import asyncio
import os
import hmac
import hashlib
import time
import aiohttp

async def check():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    # Get positions
    timestamp = int(time.time() * 1000)
    params = f'timestamp={timestamp}'
    signature = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
    
    url = f'https://testnet.binancefuture.com/fapi/v3/positionRisk?{params}&signature={signature}'
    
    async with aiohttp.ClientSession() as session:
        headers = {'X-MBX-APIKEY': api_key}
        async with session.get(url, headers=headers) as resp:
            positions = await resp.json()
            
            sol_pos = None
            for pos in positions:
                if pos['symbol'] == 'SOLUSDT' and float(pos['positionAmt']) != 0:
                    sol_pos = pos
                    break
            
            if not sol_pos:
                print('❌ No SOLUSDT position found')
                return
            
            amt = float(sol_pos['positionAmt'])
            entry = float(sol_pos['entryPrice'])
            mark = float(sol_pos['markPrice'])
            unrealized = float(sol_pos['unRealizedProfit'])
            leverage = int(float(sol_pos.get('leverage', 1)))
            
            pnl_pct = ((mark - entry) / entry * 100) if amt > 0 else ((entry - mark) / entry * 100)
            
            print(f'📊 SOLUSDT POSITION:')
            side_str = "LONG" if amt > 0 else "SHORT"
            print(f'   Side: {side_str}')
            print(f'   Amount: {amt}')
            print(f'   Entry: ${entry:.2f}')
            print(f'   Mark: ${mark:.2f}')
            print(f'   PnL: ${unrealized:.2f} ({pnl_pct:+.2f}%)')
            print(f'   Leverage: {leverage}x')
            
            # Get open orders
            timestamp2 = int(time.time() * 1000)
            params2 = f'symbol=SOLUSDT&timestamp={timestamp2}'
            signature2 = hmac.new(api_secret.encode(), params2.encode(), hashlib.sha256).hexdigest()
            
            url2 = f'https://testnet.binancefuture.com/fapi/v1/openOrders?{params2}&signature={signature2}'
            
            async with session.get(url2, headers=headers) as resp2:
                orders = await resp2.json()
                
                print(f'\n📋 OPEN ORDERS ({len(orders)}):')
                
                tp_found = False
                sl_found = False
                
                for order in orders:
                    order_type = order['type']
                    side = order['side']
                    stop_price = float(order.get('stopPrice', 0))
                    price = float(order.get('price', 0))
                    
                    target_price = stop_price if stop_price > 0 else price
                    if target_price > 0:
                        distance_pct = ((target_price - entry) / entry * 100)
                    else:
                        distance_pct = 0
                    
                    print(f'\n   {order_type} {side}')
                    print(f'     Price: ${target_price:.5f}')
                    print(f'     Distance from entry: {distance_pct:+.2f}%')
                    
                    if 'TAKE_PROFIT' in order_type:
                        tp_found = True
                        expected_tp_long = entry * 1.03  # +3% for LONG (profit on rise)
                        print(f'     Expected TP (LONG +3%): ${expected_tp_long:.5f}')
                        
                        if abs(distance_pct - 3.0) < 0.5:
                            print(f'     ✅ TP KORREKT!')
                        else:
                            print(f'     ⚠️ TP FEIL! Burde være +3%, er {distance_pct:+.2f}%')
                    
                    if 'STOP' in order_type and 'TAKE' not in order_type:
                        sl_found = True
                        expected_sl_long = entry * 0.975  # -2.5% for LONG (stop on drop)
                        print(f'     Expected SL (LONG -2.5%): ${expected_sl_long:.5f}')
                        
                        if abs(distance_pct - (-2.5)) < 0.5:
                            print(f'     ✅ SL KORREKT!')
                        else:
                            print(f'     ⚠️ SL FEIL! Burde være -2.5%, er {distance_pct:+.2f}%')
                
                print(f'\n📝 SUMMARY:')
                tp_status = "✅ FOUND" if tp_found else "❌ MISSING"
                sl_status = "✅ FOUND" if sl_found else "❌ MISSING"
                print(f'   TP Order: {tp_status}')
                print(f'   SL Order: {sl_status}')
                
                if tp_found:
                    print(f'\n🎯 TP will trigger when price reaches expected level')
                    print(f'   Current: ${mark:.2f}')
                    print(f'   Target: ${entry * 1.03:.2f} (+3%)')
                    print(f'   Distance: {((entry * 1.03 - mark) / mark * 100):+.2f}%')

if __name__ == "__main__":
    asyncio.run(check())
