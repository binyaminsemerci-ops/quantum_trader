#!/usr/bin/env python3
"""
P1-B: Execution Policy Test Runner
Simple validation of policy decision logic.
"""
import sys
import os

# Add current directory for direct import
sys.path.insert(0, os.path.dirname(__file__))

from execution_policy import (
    ExecutionPolicy,
    PolicyConfig,
    PolicyDecision,
    PortfolioState
)


def test_policy_decisions():
    """Test policy decision logic with various scenarios"""
    
    print("=" * 80)
    print("P1-B EXECUTION POLICY - TEST RUNNER")
    print("=" * 80)
    
    # Create policy with test config
    config = PolicyConfig(
        max_open_positions_total=5,
        max_open_positions_per_symbol=2,
        max_open_positions_per_regime=3,
        max_total_exposure_usdt=2000.0,
        max_exposure_per_symbol_usdt=500.0,
        allow_scale_in=True,
        scale_in_max_count=2,
        scale_in_confidence_delta=0.05,
        cooldown_seconds_per_symbol=60,
        cooldown_seconds_global=30,
        min_confidence=0.7
    )
    
    policy = ExecutionPolicy(config)
    
    # Empty portfolio state
    empty_portfolio = PortfolioState(
        total_positions=0,
        positions_by_symbol={},
        positions_by_regime={},
        total_exposure_usdt=0.0,
        exposure_by_symbol={},
        available_capital_usdt=5000.0,
        last_trade_time=0.0,
        last_trade_by_symbol={}
    )
    
    # Test 1: Valid new entry
    print("\n" + "=" * 80)
    print("TEST 1: Valid new entry (high confidence, no existing positions)")
    print("=" * 80)
    intent1 = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "confidence": 0.85,
        "entry_price": 50000.0,
        "price": 50000.0,
        "quantity": 0.01,
        "regime": "bull",
        "leverage": 10
    }
    decision, reason = policy.allow_new_entry(intent1, empty_portfolio)
    print(f"✅ Result: {decision.value} | {reason}")
    assert decision == PolicyDecision.ALLOW_NEW_ENTRY, "Should allow new entry"
    
    # Test 2: Low confidence block
    print("\n" + "=" * 80)
    print("TEST 2: Low confidence (0.65 < 0.70)")
    print("=" * 80)
    intent2 = {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "confidence": 0.65,
        "entry_price": 3000.0,
        "price": 3000.0,
        "quantity": 0.1,
        "regime": "bull",
        "leverage": 10
    }
    decision, reason = policy.allow_new_entry(intent2, empty_portfolio)
    print(f"🚫 Result: {decision.value} | {reason}")
    assert decision == PolicyDecision.BLOCK_LOW_CONFIDENCE, "Should block low confidence"
    
    # Test 3: Max positions reached
    print("\n" + "=" * 80)
    print("TEST 3: Max total positions reached")
    print("=" * 80)
    full_portfolio = PortfolioState(
        total_positions=5,  # At limit
        positions_by_symbol={
            "BTCUSDT": [{"side": "long", "quantity": 0.01, "entry_price": 50000.0, "confidence": 0.8}],
            "ETHUSDT": [{"side": "long", "quantity": 0.1, "entry_price": 3000.0, "confidence": 0.8}],
            "BNBUSDT": [{"side": "long", "quantity": 1.0, "entry_price": 400.0, "confidence": 0.8}],
            "SOLUSDT": [{"side": "long", "quantity": 10.0, "entry_price": 100.0, "confidence": 0.8}],
            "ADAUSDT": [{"side": "long", "quantity": 100.0, "entry_price": 0.50, "confidence": 0.8}],
        },
        positions_by_regime={"bull": 5},
        total_exposure_usdt=1500.0,
        exposure_by_symbol={"BTCUSDT": 500.0, "ETHUSDT": 300.0, "BNBUSDT": 400.0, "SOLUSDT": 200.0, "ADAUSDT": 100.0},
        available_capital_usdt=500.0,
        last_trade_time=0.0,
        last_trade_by_symbol={}
    )
    intent3 = {
        "symbol": "XRPUSDT",
        "side": "BUY",
        "confidence": 0.85,
        "entry_price": 0.60,
        "price": 0.60,
        "quantity": 100.0,
        "regime": "bull",
        "leverage": 10
    }
    decision, reason = policy.allow_new_entry(intent3, full_portfolio)
    print(f"🚫 Result: {decision.value} | {reason}")
    assert decision == PolicyDecision.BLOCK_MAX_POSITIONS, "Should block max positions"
    
    # Test 4: Scale-in allowed (same direction, higher confidence)
    print("\n" + "=" * 80)
    print("TEST 4: Scale-in allowed (higher confidence)")
    print("=" * 80)
    scale_in_portfolio = PortfolioState(
        total_positions=1,
        positions_by_symbol={
            "BTCUSDT": [{"side": "long", "quantity": 0.01, "entry_price": 50000.0, "confidence": 0.75}]
        },
        positions_by_regime={"bull": 1},
        total_exposure_usdt=250.0,  # Lower exposure to allow scale-in
        exposure_by_symbol={"BTCUSDT": 250.0},
        available_capital_usdt=4750.0,
        last_trade_time=0.0,
        last_trade_by_symbol={"BTCUSDT": 0.0}
    )
    intent4 = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "confidence": 0.85,  # Higher than existing 0.75 + delta 0.05
        "entry_price": 51000.0,
        "price": 51000.0,
        "quantity": 0.002,  # Smaller quantity to stay under exposure limit
        "regime": "bull",
        "leverage": 10
    }
    decision, reason = policy.allow_new_entry(intent4, scale_in_portfolio)
    print(f"✅ Result: {decision.value} | {reason}")
    assert decision == PolicyDecision.ALLOW_SCALE_IN, "Should allow scale-in"
    
    # Test 5: Scale-in blocked (insufficient confidence improvement)
    print("\n" + "=" * 80)
    print("TEST 5: Scale-in blocked (insufficient confidence delta)")
    print("=" * 80)
    intent5 = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "confidence": 0.77,  # Only +0.02, need +0.05
        "entry_price": 51000.0,
        "price": 51000.0,
        "quantity": 0.01,
        "regime": "bull",
        "leverage": 10
    }
    decision, reason = policy.allow_new_entry(intent5, scale_in_portfolio)
    print(f"🚫 Result: {decision.value} | {reason}")
    assert decision == PolicyDecision.BLOCK_SCALE_IN_RULE, "Should block insufficient confidence delta"
    
    # Test 6: Max exposure per symbol
    print("\n" + "=" * 80)
    print("TEST 6: Max exposure per symbol reached")
    print("=" * 80)
    high_exposure_portfolio = PortfolioState(
        total_positions=1,
        positions_by_symbol={
            "BTCUSDT": [{"side": "long", "quantity": 0.01, "entry_price": 50000.0, "confidence": 0.75}]
        },
        positions_by_regime={"bull": 1},
        total_exposure_usdt=480.0,
        exposure_by_symbol={"BTCUSDT": 480.0},  # Near 500 limit
        available_capital_usdt=4500.0,
        last_trade_time=0.0,
        last_trade_by_symbol={"BTCUSDT": 0.0}
    )
    intent6 = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "confidence": 0.85,
        "entry_price": 51000.0,
        "price": 51000.0,
        "quantity": 0.01,  # Would add ~510 USDT exposure
        "regime": "bull",
        "leverage": 1
    }
    decision, reason = policy.allow_new_entry(intent6, high_exposure_portfolio)
    print(f"🚫 Result: {decision.value} | {reason}")
    assert decision == PolicyDecision.BLOCK_MAX_EXPOSURE, "Should block max exposure per symbol"
    
    # Test 7: Compute order size
    print("\n" + "=" * 80)
    print("TEST 7: Compute order size (capital allocation)")
    print("=" * 80)
    qty = policy.compute_order_size(intent1, empty_portfolio, risk_score=1.0)
    print(f"💰 Calculated quantity: {qty:.6f} | intent_price: ${intent1['entry_price']:.2f}")
    assert qty > 0, "Should calculate positive quantity"
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print("Policy decisions validated:")
    print("  ✓ Allow new entry")
    print("  ✓ Block low confidence")
    print("  ✓ Block max positions")
    print("  ✓ Allow scale-in (higher confidence)")
    print("  ✓ Block scale-in (insufficient confidence)")
    print("  ✓ Block max exposure per symbol")
    print("  ✓ Compute order size")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_policy_decisions()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
