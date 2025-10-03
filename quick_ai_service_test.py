#!/usr/bin/env python3
"""Quick AI Trading Service Test

A minimal test to verify the AI trading service works.
"""

import sys

# Add backend to path
sys.path.append("backend")


def test_ai_service():
    """Test AI trading service basic functionality."""
    print("🧪 Testing AI Auto Trading Service...")

    try:
        # Import the service
        from ai_auto_trading_service import AIAutoTradingService

        print("✅ Successfully imported AIAutoTradingService")

        # Create instance
        ai_service = AIAutoTradingService()
        print("✅ Successfully created AIAutoTradingService instance")

        # Test get_status method
        status = ai_service.get_status()
        print(f"✅ get_status() works: {status}")

        # Test config attribute
        config = ai_service.config
        print(f"✅ config attribute works: {config}")

        # Test start_trading with symbols
        symbols = ["BTCUSDC", "ETHUSDC"]
        result = ai_service.start_trading(symbols)
        print(f"✅ start_trading() works: {result}")

        # Test get_recent_signals
        signals = ai_service.get_recent_signals(limit=5)
        print(f"✅ get_recent_signals() works: {len(signals)} signals")

        # Test get_recent_executions
        executions = ai_service.get_recent_executions(limit=5)
        print(f"✅ get_recent_executions() works: {len(executions)} executions")

        # Stop trading
        ai_service.stop_trading()
        print("✅ stop_trading() works")

        print("\n🎉 All AI Trading Service tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ai_service()
    if success:
        print("\n✅ AI Trading Service is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ AI Trading Service has issues")
        sys.exit(1)
