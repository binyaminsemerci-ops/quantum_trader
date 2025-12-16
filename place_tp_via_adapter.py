"""
Quick script to place missing TP order for SOLUSDT using execution adapter
"""
import sys
sys.path.insert(0, '/app')

from backend.services.execution.execution import BinanceFuturesExecutionAdapter
from backend.config import settings

async def main():
    # Init adapter
    adapter = BinanceFuturesExecutionAdapter(
        api_key=settings.BINANCE_TEST_API_KEY,
        api_secret=settings.BINANCE_TEST_API_SECRET,
        testnet=True
    )
    
    symbol = 'SOLUSDT'
    
    # Get position
    positions = await adapter.fetch_positions()
    if symbol not in positions:
        print(f"❌ No {symbol} position found!")
        return
    
    pos = positions[symbol]
    amt = float(pos['amount'])
    entry = float(pos['entryPrice'])
    
    print(f"\n📊 {symbol} POSITION:")
    print(f"   Amount: {amt}")
    print(f"   Entry: ${entry:.2f}")
    
    if amt <= 0:
        print("❌ Not a LONG position!")
        return
    
    # Calculate TP at +3%
    tp_price = entry * 1.03
    print(f"\n🎯 Placing TP:")
    print(f"   TP: ${tp_price:.2f} (+3%)")
    
    # Place order using adapter
    try:
        result = await adapter.submit_order(
            symbol=symbol,
            side='sell',
            quantity=abs(amt),
            price=tp_price,
            order_type='TAKE_PROFIT_MARKET',
            stop_price=tp_price,
            close_position=True
        )
        print(f"\n✅ TP PLACED!")
        print(f"   Order ID: {result['order_id']}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
