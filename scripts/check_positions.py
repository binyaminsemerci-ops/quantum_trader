import asyncio
from backend.adapters.binance_adapter import BinanceAdapter

async def check_positions():
    adapter = BinanceAdapter()
    await adapter.initialize()
    positions = await adapter.get_positions()
    open_pos = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
    
    print(f"Open positions: {len(open_pos)}")
    for p in open_pos:
        symbol = p["symbol"]
        amt = p["positionAmt"]
        price = p["entryPrice"]
        pnl = p.get("unRealizedProfit", 0)
        print(f"  {symbol}: {amt} @ ${price} | PnL: ${pnl}")

asyncio.run(check_positions())
