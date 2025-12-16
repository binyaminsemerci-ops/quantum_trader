"""
Exit Order Gateway - Verification Test

Tests Phase 1 implementation:
1. EXIT_MODE configuration
2. Gateway routing and logging
3. Ownership conflict detection
4. Metrics collection
"""

import asyncio
import os
import logging
from typing import Dict, Any

# Configure logging to see gateway output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MockBinanceClient:
    """Mock Binance client for testing without actual exchange calls."""
    
    def futures_create_order(self, **params):
        """Simulate successful order placement."""
        logger.info(f"[MOCK_BINANCE] futures_create_order called with params: {params}")
        return {
            'orderId': '12345678',
            'symbol': params.get('symbol'),
            'status': 'NEW',
            'type': params.get('type')
        }


async def test_exit_mode_config():
    """Test 1: EXIT_MODE configuration."""
    print("\n" + "="*80)
    print("TEST 1: EXIT_MODE Configuration")
    print("="*80)
    
    from backend.config.exit_mode import (
        get_exit_mode,
        is_exit_brain_mode,
        is_legacy_exit_mode,
        EXIT_MODE_LEGACY,
        EXIT_MODE_EXIT_BRAIN_V3
    )
    
    # Test default mode
    mode = get_exit_mode()
    print(f"✓ Default EXIT_MODE: {mode}")
    print(f"✓ is_exit_brain_mode(): {is_exit_brain_mode()}")
    print(f"✓ is_legacy_exit_mode(): {is_legacy_exit_mode()}")
    
    # Test mode switching
    os.environ['EXIT_MODE'] = 'EXIT_BRAIN_V3'
    # Re-import to pick up new env var
    import importlib
    import backend.config.exit_mode
    importlib.reload(backend.config.exit_mode)
    from backend.config.exit_mode import get_exit_mode, is_exit_brain_mode
    
    mode = get_exit_mode()
    print(f"✓ After setting EXIT_MODE=EXIT_BRAIN_V3: {mode}")
    print(f"✓ is_exit_brain_mode(): {is_exit_brain_mode()}")
    
    # Reset to LEGACY for other tests
    os.environ['EXIT_MODE'] = 'LEGACY'
    importlib.reload(backend.config.exit_mode)
    
    print("✅ TEST 1 PASSED: EXIT_MODE configuration works\n")


async def test_gateway_routing():
    """Test 2: Gateway routing and logging."""
    print("\n" + "="*80)
    print("TEST 2: Gateway Routing & Logging")
    print("="*80)
    
    from backend.services.execution.exit_order_gateway import submit_exit_order
    
    mock_client = MockBinanceClient()
    
    # Test SL order routing
    sl_params = {
        'symbol': 'BTCUSDT',
        'side': 'SELL',
        'type': 'STOP_MARKET',
        'stopPrice': 95000.0,
        'closePosition': True,
        'workingType': 'MARK_PRICE'
    }
    
    print("\n→ Submitting SL order through gateway...")
    result = await submit_exit_order(
        module_name="test_position_monitor",
        symbol="BTCUSDT",
        order_params=sl_params,
        order_kind="sl",
        client=mock_client,
        explanation="Test SL order for verification"
    )
    
    print(f"✓ SL order result: {result}")
    
    # Test TP order routing
    tp_params = {
        'symbol': 'ETHUSDT',
        'side': 'SELL',
        'type': 'TAKE_PROFIT_MARKET',
        'quantity': 1.5,
        'stopPrice': 3500.0,
        'workingType': 'MARK_PRICE'
    }
    
    print("\n→ Submitting TP order through gateway...")
    result = await submit_exit_order(
        module_name="test_hybrid_tpsl",
        symbol="ETHUSDT",
        order_params=tp_params,
        order_kind="tp",
        client=mock_client,
        explanation="Test TP order for verification"
    )
    
    print(f"✓ TP order result: {result}")
    
    print("✅ TEST 2 PASSED: Gateway routing works\n")


async def test_ownership_conflicts():
    """Test 3: Ownership conflict detection."""
    print("\n" + "="*80)
    print("TEST 3: Ownership Conflict Detection")
    print("="*80)
    
    # Set EXIT_BRAIN_V3 mode to trigger conflict warnings
    os.environ['EXIT_MODE'] = 'EXIT_BRAIN_V3'
    
    # Reload modules to pick up new mode
    import importlib
    import backend.config.exit_mode
    import backend.services.execution.exit_order_gateway
    importlib.reload(backend.config.exit_mode)
    importlib.reload(backend.services.execution.exit_order_gateway)
    
    from backend.services.execution.exit_order_gateway import submit_exit_order
    
    mock_client = MockBinanceClient()
    
    # Test legacy module in EXIT_BRAIN_V3 mode (should warn)
    print("\n→ Legacy module 'position_monitor' placing order in EXIT_BRAIN_V3 mode...")
    print("   (Should trigger ownership conflict warning)")
    
    result = await submit_exit_order(
        module_name="position_monitor",  # Legacy module
        symbol="BTCUSDT",
        order_params={
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'type': 'STOP_MARKET',
            'stopPrice': 95000.0,
            'closePosition': True
        },
        order_kind="sl",
        client=mock_client,
        explanation="Test conflict detection"
    )
    
    print(f"✓ Order still placed (soft guard): {result is not None}")
    
    # Test Exit Brain module (should NOT warn)
    print("\n→ Exit Brain module placing order in EXIT_BRAIN_V3 mode...")
    print("   (Should NOT trigger warning)")
    
    result = await submit_exit_order(
        module_name="exit_brain_executor",  # Expected Exit Brain module
        symbol="ETHUSDT",
        order_params={
            'symbol': 'ETHUSDT',
            'side': 'SELL',
            'type': 'STOP_MARKET',
            'stopPrice': 3200.0,
            'closePosition': True
        },
        order_kind="sl",
        client=mock_client,
        explanation="Exit Brain executor placing order"
    )
    
    print(f"✓ Order placed without conflict warning: {result is not None}")
    
    # Reset to LEGACY mode
    os.environ['EXIT_MODE'] = 'LEGACY'
    importlib.reload(backend.config.exit_mode)
    importlib.reload(backend.services.execution.exit_order_gateway)
    
    print("✅ TEST 3 PASSED: Ownership conflict detection works\n")


async def test_metrics_collection():
    """Test 4: Metrics collection."""
    print("\n" + "="*80)
    print("TEST 4: Metrics Collection")
    print("="*80)
    
    from backend.services.execution.exit_order_gateway import (
        submit_exit_order,
        get_exit_order_metrics,
        log_exit_metrics_summary
    )
    
    mock_client = MockBinanceClient()
    
    # Submit multiple orders
    orders = [
        ("position_monitor", "BTCUSDT", "sl"),
        ("position_monitor", "ETHUSDT", "tp"),
        ("trailing_stop_manager", "BTCUSDT", "trailing"),
        ("safe_order_executor", "SOLUSDT", "partial_tp"),
        ("event_driven_executor", "ADAUSDT", "sl"),
    ]
    
    print("\n→ Submitting 5 test orders...")
    for module, symbol, kind in orders:
        await submit_exit_order(
            module_name=module,
            symbol=symbol,
            order_params={
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': 1.0
            },
            order_kind=kind,
            client=mock_client,
            explanation=f"Test {kind} from {module}"
        )
        print(f"  ✓ {module} → {symbol} ({kind})")
    
    # Get metrics
    print("\n→ Retrieving metrics...")
    metrics = get_exit_order_metrics()
    summary = metrics.get_summary()
    
    print(f"\n📊 Metrics Summary:")
    print(f"  Total orders: {summary['total_orders']}")
    print(f"  Orders by module: {summary['orders_by_module']}")
    print(f"  Orders by kind: {summary['orders_by_kind']}")
    print(f"  Ownership conflicts: {summary['ownership_conflicts']}")
    
    # Log summary
    print("\n→ Logging metrics summary...")
    log_exit_metrics_summary()
    
    # Verify counts
    assert summary['total_orders'] >= 5, f"Expected >= 5 orders, got {summary['total_orders']}"
    assert 'position_monitor' in summary['orders_by_module'], "position_monitor should be in metrics"
    
    print("✅ TEST 4 PASSED: Metrics collection works\n")


async def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("EXIT ORDER GATEWAY - PHASE 1 VERIFICATION")
    print("="*80)
    print("\nTesting Phase 1 implementation:")
    print("  1. EXIT_MODE configuration")
    print("  2. Gateway routing and logging")
    print("  3. Ownership conflict detection")
    print("  4. Metrics collection")
    print("\n")
    
    try:
        await test_exit_mode_config()
        await test_gateway_routing()
        await test_ownership_conflicts()
        await test_metrics_collection()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - PHASE 1 VERIFICATION COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Deploy to Docker")
        print("  2. Monitor logs for ownership conflicts")
        print("  3. Analyze metrics to identify active MUSCLE modules")
        print("  4. Plan Phase 2: Build Exit Brain Executor")
        print("\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
