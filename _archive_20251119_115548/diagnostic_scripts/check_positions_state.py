"""Check current positions and their tracking state"""
import asyncio
from backend.services.execution.execution import build_execution_adapter, TradeStateStore
from backend.config.execution import load_execution_config
from pathlib import Path

async def check():
    config = load_execution_config()
    adapter = build_execution_adapter(config)
    positions = await adapter.get_positions()
    
    print(f"[CHART] Open positions: {len(positions)}")
    for sym, qty in positions.items():
        side = "LONG" if qty > 0 else "SHORT"
        print(f"  {sym}: {side} {abs(qty):.4f}")
    
    # Check tracking state
    trade_state_path = Path(__file__).resolve().parent / "backend" / "data" / "trade_state.json"
    trade_store = TradeStateStore(trade_state_path)
    
    print(f"\n[MEMO] Position tracking states:")
    for sym in positions.keys():
        state = trade_store.get(sym)
        if state:
            recovered = "♻️ RECOVERED" if state.get('recovered') else "[OK] NORMAL"
            print(f"  {sym}: {recovered} | entry=${state.get('avg_entry'):.4f}, side={state.get('side')}")
        else:
            print(f"  {sym}: ❌ NO STATE")

asyncio.run(check())
