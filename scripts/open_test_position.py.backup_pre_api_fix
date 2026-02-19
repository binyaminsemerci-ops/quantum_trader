import asyncio
import sys
import os
sys.path.insert(0, "/app")
from binance import AsyncClient

async def main():
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    testnet = os.environ.get("BINANCE_USE_TESTNET") == "true"
    
    client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
    
    # Get current BTC price
    ticker = await client.futures_symbol_ticker(symbol="BTCUSDT")
    price = float(ticker.get("price", 0))
    print(f"📊 Current BTC Price: ${price:,.2f}")
    
    # Open small LONG position (0.001 BTC ~ $100 notional on testnet)
    print("\n🚀 Opening testnet position...")
    order = await client.futures_create_order(
        symbol="BTCUSDT", 
        side="BUY", 
        positionSide="LONG",
        type="MARKET", 
        quantity=0.001
    )
    
    sym = order.get("symbol")
    qty = order.get("origQty")
    status = order.get("status")
    side = order.get("side")
    pos_side = order.get("positionSide")
    
    print(f"\n✅ Position Opened Successfully!")
    print(f"   Symbol: {sym}")
    print(f"   Side: {side} {pos_side}")
    print(f"   Quantity: {qty} BTC")
    print(f"   Status: {status}")
    print(f"\n🎯 Exit Brain v3.5 will detect this position in next monitoring cycle (10s)...")
    print(f"   Watch logs: docker logs -f quantum_position_monitor")
    
    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())
