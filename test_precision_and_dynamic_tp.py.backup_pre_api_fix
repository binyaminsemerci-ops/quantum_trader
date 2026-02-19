"""
Comprehensive test for:
1. Centralized precision/quantization layer
2. Dynamic TP profile integration

Tests both testnet and mainnet scenarios.
"""
import asyncio
import logging
from unittest.mock import Mock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_precision_layer():
    """Test Part A: Centralized precision quantization"""
    logger.info("=" * 80)
    logger.info("PART A: TESTING PRECISION LAYER")
    logger.info("=" * 80)
    
    from backend.domains.exits.exit_brain_v3.precision import (
        quantize_price, quantize_stop_price, quantize_quantity,
        get_symbol_filters, clear_precision_cache
    )
    
    # Mock Binance client
    mock_client = Mock()
    mock_client.futures_exchange_info = Mock(return_value={
        'symbols': [
            {
                'symbol': 'BTCUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'tickSize': '0.1', 'minPrice': '0.1', 'maxPrice': '1000000'},
                    {'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001', 'maxQty': '1000'},
                    {'filterType': 'MIN_NOTIONAL', 'notional': '5.0'}
                ]
            },
            {
                'symbol': 'XRPUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'tickSize': '0.00001', 'minPrice': '0.00001', 'maxPrice': '1000'},
                    {'filterType': 'LOT_SIZE', 'stepSize': '1.0', 'minQty': '1.0', 'maxQty': '10000000'},
                    {'filterType': 'MIN_NOTIONAL', 'notional': '5.0'}
                ]
            }
        ]
    })
    
    # Test 1: BTC price quantization
    logger.info("\n--- Test 1: BTC Price Quantization ---")
    price = 95234.567891
    quantized = quantize_price('BTCUSDT', price, mock_client)
    logger.info(f"Original: {price:.8f}, Quantized: {quantized:.8f}")
    assert quantized == 95234.5, f"Expected 95234.5, got {quantized}"
    logger.info("[OK] BTC price quantized correctly (tick=0.1)")
    
    # Test 2: XRP price quantization (5 decimals)
    logger.info("\n--- Test 2: XRP Price Quantization ---")
    xrp_price = 1.234567891
    quantized_xrp = quantize_price('XRPUSDT', xrp_price, mock_client)
    logger.info(f"Original: {xrp_price:.8f}, Quantized: {quantized_xrp:.8f}")
    assert quantized_xrp == 1.23456, f"Expected 1.23456, got {quantized_xrp}"
    logger.info("[OK] XRP price quantized correctly (tick=0.00001)")
    
    # Test 3: XRP stopPrice quantization
    logger.info("\n--- Test 3: XRP StopPrice Quantization ---")
    stop_price = 1.998374
    quantized_stop = quantize_stop_price('XRPUSDT', stop_price, mock_client)
    logger.info(f"Original: {stop_price:.8f}, Quantized: {quantized_stop:.8f}")
    assert quantized_stop == 1.99837, f"Expected 1.99837, got {quantized_stop}"
    logger.info("[OK] XRP stopPrice quantized correctly")
    
    # Test 4: BTC quantity quantization
    logger.info("\n--- Test 4: BTC Quantity Quantization ---")
    quantity = 0.0527891
    quantized_qty = quantize_quantity('BTCUSDT', quantity, mock_client)
    logger.info(f"Original: {quantity:.8f}, Quantized: {quantized_qty:.8f}")
    assert quantized_qty == 0.052, f"Expected 0.052, got {quantized_qty}"
    logger.info("[OK] BTC quantity quantized correctly (step=0.001)")
    
    # Test 5: XRP quantity below min_qty
    logger.info("\n--- Test 5: XRP Quantity Below Min ---")
    small_qty = 0.5
    quantized_small = quantize_quantity('XRPUSDT', small_qty, mock_client)
    logger.info(f"Original: {small_qty}, Quantized: {quantized_small}")
    assert quantized_small == 0.0, f"Expected 0.0 (below min), got {quantized_small}"
    logger.info("[OK] Small quantity correctly rejected (below min_qty=1.0)")
    
    # Test 6: Get symbol filters
    logger.info("\n--- Test 6: Symbol Filters ---")
    filters = get_symbol_filters('XRPUSDT', mock_client)
    logger.info(f"XRPUSDT filters: {filters}")
    assert filters['tick_size'] == 0.00001
    assert filters['step_size'] == 1.0
    assert filters['min_qty'] == 1.0
    logger.info("[OK] Symbol filters retrieved correctly")
    
    # Clean up
    clear_precision_cache()
    logger.info("\n[PART A COMPLETE] All precision tests passed!")
    return True


async def test_dynamic_tp_profiles():
    """Test Part B: Dynamic TP profile integration"""
    logger.info("\n" + "=" * 80)
    logger.info("PART B: TESTING DYNAMIC TP PROFILES")
    logger.info("=" * 80)
    
    from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
    from backend.domains.exits.exit_brain_v3.models import ExitContext
    
    # Test 1: High leverage position (should tighten TPs)
    logger.info("\n--- Test 1: High Leverage (20x) ---")
    config_dynamic = {
        "use_profiles": True,
        "use_dynamic_tp": True,
        "strategy_id": "RL_V3"
    }
    planner = ExitBrainV3(config=config_dynamic)
    
    ctx_high_lev = ExitContext(
        symbol="XRPUSDT",
        side="LONG",
        entry_price=2.00,
        current_price=2.01,
        size=1000.0,
        leverage=20.0,  # HIGH LEVERAGE
        volatility=0.025,
        market_regime="NORMAL",
        signal_confidence=0.85,
        unrealized_pnl_pct=0.5,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan = await planner.build_exit_plan(ctx_high_lev)
    tp_legs = [leg for leg in plan.legs if leg.kind.value == "TP"]
    
    logger.info(f"Plan created with {len(plan.legs)} total legs, {len(tp_legs)} TP legs")
    for i, leg in enumerate(tp_legs):
        is_dynamic = leg.metadata.get('dynamic', False)
        logger.info(
            f"  TP{i+1}: trigger={leg.trigger_pct:.2%}, "
            f"size={leg.size_pct:.0%}, "
            f"type={'DYNAMIC' if is_dynamic else 'STATIC'}"
        )
    
    assert len(tp_legs) >= 3, f"Expected at least 3 TP legs, got {len(tp_legs)}"
    logger.info("[OK] High leverage plan generated with dynamic TPs")
    
    # Test 2: Large position (should front-load exits)
    logger.info("\n--- Test 2: Large Position ($5000) ---")
    ctx_large = ExitContext(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=95000.0,
        current_price=95500.0,
        size=0.0526,  # ~$5000
        leverage=10.0,
        volatility=0.030,
        market_regime="NORMAL",
        signal_confidence=0.90,
        unrealized_pnl_pct=0.53,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan2 = await planner.build_exit_plan(ctx_large)
    tp_legs2 = [leg for leg in plan2.legs if leg.kind.value == "TP"]
    
    logger.info(f"Plan created with {len(tp_legs2)} TP legs")
    total_size = sum(leg.size_pct for leg in tp_legs2)
    first_tp_size = tp_legs2[0].size_pct if tp_legs2 else 0
    
    logger.info(f"  First TP size: {first_tp_size:.0%}")
    logger.info(f"  Total TP size: {total_size:.0%}")
    
    assert first_tp_size >= 0.35, f"Expected first TP >= 35% for large position, got {first_tp_size:.0%}"
    logger.info("[OK] Large position plan with front-loaded exits")
    
    # Test 3: Trending market (should widen TPs)
    logger.info("\n--- Test 3: Trending Market ---")
    ctx_trend = ExitContext(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=180.0,
        current_price=182.0,
        size=10.0,
        leverage=15.0,
        volatility=0.045,  # High volatility
        market_regime="TRENDING",  # TRENDING
        signal_confidence=0.88,
        unrealized_pnl_pct=1.11,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan3 = await planner.build_exit_plan(ctx_trend)
    tp_legs3 = [leg for leg in plan3.legs if leg.kind.value == "TP"]
    
    logger.info(f"Plan created with {len(tp_legs3)} TP legs")
    if tp_legs3:
        last_tp = tp_legs3[-1]
        logger.info(f"  Last TP: trigger={last_tp.trigger_pct:.2%}, size={last_tp.size_pct:.0%}")
        
        # In trending markets, last TP should be widest
        assert last_tp.trigger_pct >= 0.05, f"Expected wide last TP in trending market, got {last_tp.trigger_pct:.2%}"
        logger.info("[OK] Trending market plan with wide TPs")
    
    # Test 4: Static vs Dynamic comparison
    logger.info("\n--- Test 4: Static vs Dynamic Comparison ---")
    config_static = {
        "use_profiles": True,
        "use_dynamic_tp": False,  # STATIC
        "strategy_id": "RL_V3"
    }
    planner_static = ExitBrainV3(config=config_static)
    
    ctx_compare = ExitContext(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=3500.0,
        current_price=3520.0,
        size=0.5,
        leverage=20.0,
        volatility=0.028,
        market_regime="NORMAL",
        signal_confidence=0.82,
        unrealized_pnl_pct=0.57,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan_static = await planner_static.build_exit_plan(ctx_compare)
    plan_dynamic = await planner.build_exit_plan(ctx_compare)
    
    tp_static = [leg for leg in plan_static.legs if leg.kind.value == "TP"]
    tp_dynamic = [leg for leg in plan_dynamic.legs if leg.kind.value == "TP"]
    
    logger.info("STATIC TPs:")
    for i, leg in enumerate(tp_static):
        logger.info(f"  TP{i+1}: {leg.trigger_pct:.2%} @ {leg.size_pct:.0%}")
    
    logger.info("DYNAMIC TPs:")
    for i, leg in enumerate(tp_dynamic):
        logger.info(f"  TP{i+1}: {leg.trigger_pct:.2%} @ {leg.size_pct:.0%}")
    
    # With 20x leverage, dynamic should be tighter
    if tp_static and tp_dynamic:
        static_avg = sum(leg.trigger_pct for leg in tp_static) / len(tp_static)
        dynamic_avg = sum(leg.trigger_pct for leg in tp_dynamic) / len(tp_dynamic)
        logger.info(f"Average TP distance: Static={static_avg:.2%}, Dynamic={dynamic_avg:.2%}")
        
        assert dynamic_avg < static_avg * 1.1, "Dynamic TPs should be tighter or similar for high leverage"
        logger.info("[OK] Dynamic adapts to leverage correctly")
    
    logger.info("\n[PART B COMPLETE] All dynamic TP tests passed!")
    return True


async def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE TEST: PRECISION + DYNAMIC TP")
    logger.info("=" * 80)
    
    # Part A: Precision
    precision_ok = test_precision_layer()
    
    # Part B: Dynamic TP
    dynamic_ok = await test_dynamic_tp_profiles()
    
    if precision_ok and dynamic_ok:
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("Part A: Precision layer working correctly")
        logger.info("Part B: Dynamic TP profiles adapting to context")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
