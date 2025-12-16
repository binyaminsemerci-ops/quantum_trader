#!/usr/bin/env python3
"""Force position monitor to re-check and fix all TP/SL orders."""
import sys
sys.path.insert(0, '/app')

from backend.services.monitoring.position_monitor import PositionMonitor
from binance.client import Client
import os
import asyncio

async def main():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')  # Changed from BINANCE_SECRET_KEY
    
    if not api_key or not api_secret:
        print("❌ Missing API credentials!")
        return
    
    client = Client(api_key, api_secret)
    monitor = PositionMonitor(
        client=client,
        sl_pct=0.02,  # 2%
        tp_pct=0.03,  # 3%
        trail_pct=0.015,  # 1.5%
        partial_tp=0.5  # 50%
    )
    
    print("\n🔄 TVINGER RE-KALKULERING AV ALLE TP/SL ORDERS\n")
    
    # Get all positions
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    
    print(f"Fant {len(open_positions)} åpne posisjoner\n")
    
    for pos in open_positions:
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        leverage = float(pos['leverage'])
        
        print(f"[CHART] {symbol}: {amt} @ ${entry} ({leverage}x leverage)")
        
        # Delete existing orders
        try:
            orders = client.futures_get_open_orders(symbol=symbol)
            for order in orders:
                if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'TRAILING_STOP_MARKET']:
                    client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                    print(f"   🗑️  Slettet {order['type']}")
        except Exception as e:
            print(f"   [WARNING]  Kunne ikke slette orders: {e}")
        
        # Re-create with correct leverage calculation
        success = await monitor._set_tpsl_for_position(pos)
        if success:
            print(f"   [OK] TP/SL re-kalkulert med leverage!\n")
        else:
            print(f"   ❌ Feil ved setting av TP/SL\n")

if __name__ == '__main__':
    asyncio.run(main())
