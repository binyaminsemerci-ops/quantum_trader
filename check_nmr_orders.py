#!/usr/bin/env python3
"""Check NMRUSDT orders on Binance testnet."""

import asyncio
from backend.services.binance_adapter import BinanceAdapter


async def main():
    adapter = BinanceAdapter()
    await adapter.initialize()
    
    # Get open orders
    orders = await adapter.client.futures_get_open_orders(symbol='NMRUSDT')
    
    if not orders:
        print("No open orders for NMRUSDT")
    else:
        print(f"Found {len(orders)} open orders for NMRUSDT:\n")
        for o in orders:
            print(f"Order ID: {o['orderId']}")
            print(f"  Type: {o['type']}")
            print(f"  Side: {o['side']}")
            print(f"  StopPrice: {o.get('stopPrice', 'N/A')}")
            print(f"  Quantity: {o.get('origQty', 'N/A')}")
            print(f"  Status: {o['status']}")
            print()
    
    # Get position
    positions = await adapter.get_positions()
    if 'NMRUSDT' in positions:
        print(f"\nNMRUSDT position: {positions['NMRUSDT']} units")
    else:
        print("\nNo NMRUSDT position found")


if __name__ == "__main__":
    asyncio.run(main())
