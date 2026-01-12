#!/usr/bin/env python3
"""Quick test of Tier 1 pipeline"""
from ai_engine.services.eventbus_bridge import *
import asyncio

async def test():
    print("📤 Publishing test signal: BTCUSDT BUY @ 0.85")
    await publish_trade_signal("BTCUSDT", "BUY", 0.85, "manual_test")
    print("✅ Signal published to trade.signal.v5")
    
    await asyncio.sleep(4)
    
    approved = await get_recent_signals("trade.signal.safe", 5)
    print(f"\n✅ Approved signals: {len(approved)}")
    if approved:
        a = approved[0]
        print(f"   Symbol: {a.get('symbol')}")
        print(f"   Action: {a.get('action')}")
        print(f"   Size: ${a.get('position_size_usd', 0):.2f}")
    
    executions = await get_recent_signals("trade.execution.res", 5)
    print(f"\n✅ Executions: {len(executions)}")
    if executions:
        e = executions[0]
        print(f"   Order: {e.get('order_id')}")
        print(f"   Price: ${e.get('entry_price', 0):.2f}")
        print(f"   Status: {e.get('status')}")
    
    positions = await get_recent_signals("trade.position.update", 5)
    print(f"\n✅ Position updates: {len(positions)}")

asyncio.run(test())
