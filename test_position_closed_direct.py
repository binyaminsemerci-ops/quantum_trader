"""
Direct test of position_closed event publishing.
Bypasses HTTP layer to test event flow directly.
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_position_closed():
    """Test position_closed event publishing directly."""
    print("=" * 80)
    print("DIRECT POSITION CLOSED EVENT TEST")
    print("=" * 80)
    
    # Import event publishers
    try:
        from backend.events.publishers import publish_position_closed
        print("✅ Successfully imported publish_position_closed")
    except ImportError as e:
        print(f"❌ Failed to import publisher: {e}")
        return
    
    # Import logger
    try:
        from backend.core.logger import get_logger
        logger = get_logger(__name__)
        print("✅ Successfully imported logger")
    except ImportError as e:
        print(f"⚠️  Logger import failed (continuing): {e}")
        logger = None
    
    # Test parameters
    symbol = "BTCUSDT"
    entry_price = 40000.0
    exit_price = 40500.0
    size_usd = 100.0
    leverage = 5.0
    is_long = True
    
    # Calculate P&L
    if is_long:
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    else:
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
    
    pnl_usd = (size_usd * leverage) * (pnl_pct / 100)
    
    duration_seconds = 3600.0
    exit_reason = "TEST"
    entry_confidence = 0.75
    model_version = "test_v1"
    trace_id = f"test-pos-direct-001"
    
    print("\n📊 TEST PARAMETERS:")
    print(f"   Symbol: {symbol}")
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Exit: ${exit_price:,.2f}")
    print(f"   Size: ${size_usd:.2f}")
    print(f"   Leverage: {leverage}x")
    print(f"   Position: {'LONG' if is_long else 'SHORT'}")
    print(f"   P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
    print(f"   Trace ID: {trace_id}")
    
    # Publish event
    print("\n🚀 Publishing position.closed event...")
    try:
        await publish_position_closed(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            size_usd=size_usd,
            leverage=leverage,
            is_long=is_long,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            duration_seconds=duration_seconds,
            exit_reason=exit_reason,
            trace_id=trace_id,
            entry_confidence=entry_confidence,
            model_version=model_version
        )
        print("✅ Event published successfully!")
        
        print("\n📋 EXPECTED BEHAVIOR:")
        print("   1. PositionSubscriber should receive this event")
        print("   2. Event data should be fed to 5 learning systems:")
        print("      - RL Position Sizing Agent")
        print("      - RL Meta Strategy Agent")
        print("      - Model Supervisor")
        print("      - Drift Detector")
        print("      - Continuous Learning Manager (CLM)")
        print("   3. Check backend logs for '[EVENT] position.closed' messages")
        
    except Exception as e:
        print(f"❌ Failed to publish event: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_position_closed())
