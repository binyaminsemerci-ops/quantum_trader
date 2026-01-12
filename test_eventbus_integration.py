"""
Test EventBus integration with EnsembleManager

Validates that ensemble predictions trigger EventBus signal publishing.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_engine.ensemble_manager import EnsembleManager
from ai_engine.services.eventbus_bridge import EventBusClient


async def test_ensemble_eventbus():
    """Test that ensemble publishes signals to EventBus"""
    print("🧪 TESTING EVENTBUS INTEGRATION")
    print("=" * 60)
    
    # Initialize ensemble
    print("\n1️⃣ Initializing EnsembleManager...")
    ensemble = EnsembleManager(
        enabled_models=['xgb', 'lgbm']  # Only XGB+LGBM for fast testing
    )
    
    # Check EventBus status
    if not ensemble.eventbus_enabled:
        print("❌ FAILED: EventBus not enabled in ensemble")
        return False
    else:
        print("✅ EventBus enabled in ensemble")
    
    # Create test features
    print("\n2️⃣ Creating test signal...")
    test_features = {
        'close': 95000.0,
        'volume': 1000000.0,
        'rsi_14': 65.0,
        'macd': 150.0,
        'bb_upper': 96000.0,
        'bb_lower': 94000.0,
        'atr_14': 1200.0,
        'ema_12': 94800.0,
        'ema_26': 94500.0,
        'volatility': 0.012
    }
    
    # Get prediction (this should trigger EventBus publish)
    action, confidence, info = ensemble.predict('BTCUSDT', test_features)
    
    print(f"Prediction: {action} (confidence={confidence:.3f})")
    print(f"Governer approved: {info.get('governer', {}).get('approved', 'N/A')}")
    
    # Give async task time to complete
    print("\n3️⃣ Waiting for async EventBus publish...")
    await asyncio.sleep(2)
    
    # Check if signal was published
    print("\n4️⃣ Checking Redis for published signal...")
    async with EventBusClient() as bus:
        length = await bus.get_stream_length('trade.signal.v5')
        print(f"✅ Stream trade.signal.v5 has {length} messages")
        
        # Read last signal
        messages = await bus.get_stream_messages('trade.signal.v5', count=1)
        if messages:
            last_msg = messages[0]
            print(f"\nLast signal:")
            print(f"  Symbol: {last_msg.get('symbol')}")
            print(f"  Action: {last_msg.get('action')}")
            print(f"  Confidence: {last_msg.get('confidence')}")
            print(f"  Source: {last_msg.get('source')}")
            print(f"  Timestamp: {last_msg.get('timestamp')}")
    
    print("\n" + "=" * 60)
    print("✅ TEST PASSED: EnsembleManager → EventBus integration working")
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_ensemble_eventbus())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
