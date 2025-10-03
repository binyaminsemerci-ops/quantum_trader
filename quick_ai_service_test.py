#!/usr/bin/env python3
"""Quick AI Trading Service Test.

A minimal test to verify the AI trading service works.
"""

import sys
from typing import Optional

# Add backend to path
sys.path.append("backend")


def test_ai_service() -> Optional[bool]:
    """Test AI trading service basic functionality."""
    try:
        # Import the service
        from ai_auto_trading_service import AIAutoTradingService

        # Create instance
        ai_service = AIAutoTradingService()

        # Test get_status method
        ai_service.get_status()

        # Test config attribute

        # Test start_trading with symbols
        symbols = ["BTCUSDC", "ETHUSDC"]
        ai_service.start_trading(symbols)

        # Test get_recent_signals
        ai_service.get_recent_signals(limit=5)

        # Test get_recent_executions
        ai_service.get_recent_executions(limit=5)

        # Stop trading
        ai_service.stop_trading()

        return True

    except Exception:
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ai_service()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
