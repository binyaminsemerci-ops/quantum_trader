"""
Diagnose Exit Brain V3 Monitoring Loop Issue

This script checks:
1. If monitoring loop is running
2. If positions exist
3. If AI is setting SL/TP levels
4. Why no monitoring cycle logs appear
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def main():
    print("=" * 80)
    print("🔍 EXIT BRAIN V3 DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Check if executor is initialized
    print("\n[1] Checking executor initialization...")
    try:
        from backend.main import app_instance
        if hasattr(app_instance.state, 'exit_brain_executor'):
            executor = app_instance.state.exit_brain_executor
            print(f"✅ Executor found: {executor}")
            print(f"   - Mode: {executor.effective_mode}")
            print(f"   - Shadow: {executor.shadow_mode}")
            print(f"   - Running: {executor._running}")
            print(f"   - Loop interval: {executor.loop_interval_sec}s")
            print(f"   - State count: {len(executor._state)}")
        else:
            print("❌ Executor NOT found in app_instance.state")
            return
    except Exception as e:
        print(f"❌ Error accessing executor: {e}")
        return
    
    # 2. Check open positions
    print("\n[2] Checking open positions...")
    try:
        from backend.integrations.binance.binance_client import get_binance_client
        client = get_binance_client()
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        print(f"✅ Found {len(open_positions)} open positions:")
        for pos in open_positions[:5]:  # Show first 5
            symbol = pos['symbol']
            side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
            amt = abs(float(pos['positionAmt']))
            entry = float(pos['entryPrice'])
            mark = float(pos['markPrice'])
            pnl = float(pos['unRealizedProfit'])
            print(f"   - {symbol} {side}: {amt} @ ${entry:.4f}, mark=${mark:.4f}, PnL=${pnl:.2f}")
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Check executor state
    print("\n[3] Checking executor internal state...")
    if executor._state:
        print(f"✅ Executor has {len(executor._state)} position states:")
        for key, state in executor._state.items():
            print(f"   - {key}:")
            print(f"       active_sl: {state.active_sl}")
            print(f"       tp_levels: {len(state.tp_levels)} levels")
            print(f"       last_price: {state.last_price}")
            print(f"       last_updated: {state.last_updated}")
    else:
        print("⚠️  Executor state is EMPTY")
    
    # 4. Test monitoring cycle manually
    print("\n[4] Testing manual monitoring cycle...")
    try:
        print("   Running _monitoring_cycle(999)...")
        await executor._monitoring_cycle(999)
        print("✅ Monitoring cycle completed without error")
    except Exception as e:
        print(f"❌ Monitoring cycle FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Check for async task issues
    print("\n[5] Checking monitoring task status...")
    if executor._monitoring_task:
        print(f"   Task: {executor._monitoring_task}")
        print(f"   Done: {executor._monitoring_task.done()}")
        print(f"   Cancelled: {executor._monitoring_task.cancelled()}")
        if executor._monitoring_task.done():
            try:
                exc = executor._monitoring_task.exception()
                if exc:
                    print(f"   ❌ Task FAILED with exception: {exc}")
                else:
                    print(f"   ✅ Task completed successfully")
            except Exception as e:
                print(f"   ⚠️  Cannot get exception: {e}")
    else:
        print("   ❌ NO monitoring task found!")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
