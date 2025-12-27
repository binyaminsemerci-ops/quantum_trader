"""Check what fields are available in Binance position data"""
import asyncio
from backend.services.execution.execution import build_execution_adapter
from backend.config.execution import load_execution_config

async def check():
    config = load_execution_config()
    adapter = build_execution_adapter(config)
    data = await adapter._signed_request('GET', '/fapi/v2/account')
    positions = [p for p in data.get('positions', []) if abs(float(p.get('positionAmt', 0))) > 0]
    
    if positions:
        print(f"Found {len(positions)} open positions")
        print(f"Available fields: {list(positions[0].keys())}")
        print("\nPosition details:")
        for p in positions[:4]:
            print(f"  {p['symbol']}: entryPrice={p.get('entryPrice')}, positionAmt={p.get('positionAmt')}, unrealizedProfit={p.get('unrealizedProfit')}")
    else:
        print("No open positions found")

asyncio.run(check())
