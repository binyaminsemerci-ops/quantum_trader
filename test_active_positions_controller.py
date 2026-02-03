#!/usr/bin/env python3
"""
Active Positions Controller - Proof Script
===========================================

Demonstrates:
1. Slots full → blocks new opens
2. New score 20% higher → rotation triggers
3. Correlation > 80% → blocks
4. Margin > 65% → blocks
5. No policy → blocks (fail-closed)

No hardcoded symbols - reads from PolicyStore.
"""

import asyncio
import json
import numpy as np
from dataclasses import dataclass
from typing import List

# Mock Redis for testing
class MockRedis:
    def __init__(self, has_policy=True, universe=None):
        self.has_policy = has_policy
        self.universe = universe or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
            "DOTUSDT", "MATICUSDT", "LINKUSDT", "AVAXUSDT", "ATOMUSDT"
        ]
    
    async def hgetall(self, key):
        if not self.has_policy:
            return {}
        
        if key == "quantum:policy:current":
            return {
                b"universe_symbols": json.dumps(self.universe).encode("utf-8")
            }
        return {}

# Mock PolicyStore
class MockPolicyStore:
    def __init__(self, has_policy=True, universe=None):
        self.redis = MockRedis(has_policy, universe)

# Import controller (adjust path if needed)
import sys
sys.path.insert(0, "/home/qt/quantum_trader")
from backend.services.risk.active_positions_controller import (
    ActivePositionsController,
    SlotConfig,
    PositionSnapshot,
    SlotDecision,
)


# ============================================================================
# TEST SCENARIOS
# ============================================================================

async def test_scenario_1_slots_full_blocks():
    """Scenario 1: open=desired_slots → blocks new opens"""
    print("\n" + "="*70)
    print("SCENARIO 1: Slots Full → Block New Opens")
    print("="*70)
    
    # Setup
    policy_store = MockPolicyStore(has_policy=True)
    controller = ActivePositionsController(policy_store)
    
    # Simulate 4 open positions (base_slots = 4)
    portfolio = [
        PositionSnapshot(
            symbol="BTCUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=50,
            entry_price=45000,
            current_price=45500,
            age_seconds=3600,
            score=75.0,
        ),
        PositionSnapshot(
            symbol="ETHUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=30,
            entry_price=3000,
            current_price=3030,
            age_seconds=3600,
            score=72.0,
        ),
        PositionSnapshot(
            symbol="SOLUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=20,
            entry_price=100,
            current_price=102,
            age_seconds=3600,
            score=70.0,
        ),
        PositionSnapshot(
            symbol="BNBUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=10,
            entry_price=500,
            current_price=505,
            age_seconds=3600,
            score=68.0,
        ),
    ]
    
    # Try to open a new position
    market_features = {
        "trend_strength": 0.5,  # TREND_MODERATE → base_slots=4
        "atr_pct": 1.5,
        "momentum_consistency": 0.5,
    }
    
    decision, record = await controller.evaluate_position_request(
        symbol="ADAUSDT",
        action="NEW_TRADE",
        candidate_score=73.0,  # Not much better than weakest (68)
        candidate_returns=None,
        market_features=market_features,
        portfolio_positions=portfolio,
        total_margin_usage_pct=40.0,
    )
    
    print(f"\n✅ Test Result:")
    print(f"   Decision: {decision.value}")
    print(f"   Expected: BLOCKED_SLOTS_FULL")
    print(f"   Match: {decision == SlotDecision.BLOCKED_SLOTS_FULL}")
    print(f"   Reason: {record.block_reason}")
    print(f"   Open positions: {record.open_positions_count}/{record.desired_slots}")
    
    assert decision == SlotDecision.BLOCKED_SLOTS_FULL, "Expected BLOCKED_SLOTS_FULL"
    print("\n✅ SCENARIO 1 PASSED: Slots full blocks new opens")


async def test_scenario_2_rotation_triggers():
    """Scenario 2: new score 20% higher → rotation triggers"""
    print("\n" + "="*70)
    print("SCENARIO 2: Score 20% Higher → Rotation Triggers")
    print("="*70)
    
    # Setup
    policy_store = MockPolicyStore(has_policy=True)
    controller = ActivePositionsController(policy_store)
    
    # Simulate 4 open positions (base_slots = 4)
    portfolio = [
        PositionSnapshot(
            symbol="BTCUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=50,
            entry_price=45000,
            current_price=45500,
            age_seconds=3600,
            score=75.0,
        ),
        PositionSnapshot(
            symbol="ETHUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=30,
            entry_price=3000,
            current_price=3030,
            age_seconds=3600,
            score=72.0,
        ),
        PositionSnapshot(
            symbol="SOLUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=20,
            entry_price=100,
            current_price=102,
            age_seconds=3600,
            score=70.0,
        ),
        PositionSnapshot(
            symbol="BNBUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=-10,
            entry_price=500,
            current_price=495,
            age_seconds=3600,
            score=60.0,  # WEAKEST
        ),
    ]
    
    # Try to open a new position with 20% better score
    market_features = {
        "trend_strength": 0.5,
        "atr_pct": 1.5,
        "momentum_consistency": 0.5,
    }
    
    decision, record = await controller.evaluate_position_request(
        symbol="ADAUSDT",
        action="NEW_TRADE",
        candidate_score=73.0,  # 21.7% better than weakest (60)
        candidate_returns=None,
        market_features=market_features,
        portfolio_positions=portfolio,
        total_margin_usage_pct=40.0,
    )
    
    print(f"\n✅ Test Result:")
    print(f"   Decision: {decision.value}")
    print(f"   Expected: ROTATION_TRIGGERED")
    print(f"   Match: {decision == SlotDecision.ROTATION_TRIGGERED}")
    print(f"   Weakest symbol: {record.weakest_symbol}")
    print(f"   Weakest score: {record.weakest_score:.2f}")
    print(f"   New symbol: {record.symbol}")
    print(f"   New score: {record.candidate_score:.2f}")
    print(f"   Improvement: {((record.candidate_score - record.weakest_score) / record.weakest_score):.1%}")
    
    assert decision == SlotDecision.ROTATION_TRIGGERED, "Expected ROTATION_TRIGGERED"
    assert record.weakest_symbol == "BNBUSDT", "Expected BNBUSDT to be weakest"
    print("\n✅ SCENARIO 2 PASSED: Rotation triggered for 20% better candidate")


async def test_scenario_3_correlation_blocks():
    """Scenario 3: correlation > 80% → blocks"""
    print("\n" + "="*70)
    print("SCENARIO 3: Correlation > 80% → Block")
    print("="*70)
    
    # Setup
    policy_store = MockPolicyStore(has_policy=True)
    config = SlotConfig(max_correlation=0.80)
    controller = ActivePositionsController(policy_store, config)
    
    # Create correlated returns
    base_returns = np.random.randn(100)
    highly_correlated_returns = base_returns + np.random.randn(100) * 0.1  # 90%+ corr
    
    # Simulate 2 open positions (slots available)
    portfolio = [
        PositionSnapshot(
            symbol="BTCUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=50,
            entry_price=45000,
            current_price=45500,
            age_seconds=3600,
            score=75.0,
            returns_series=base_returns.tolist(),
        ),
    ]
    
    market_features = {
        "trend_strength": 0.5,
        "atr_pct": 1.5,
        "momentum_consistency": 0.5,
    }
    
    decision, record = await controller.evaluate_position_request(
        symbol="ETHUSDT",
        action="NEW_TRADE",
        candidate_score=80.0,
        candidate_returns=highly_correlated_returns.tolist(),
        market_features=market_features,
        portfolio_positions=portfolio,
        total_margin_usage_pct=20.0,
    )
    
    print(f"\n✅ Test Result:")
    print(f"   Decision: {decision.value}")
    print(f"   Expected: BLOCKED_CORRELATION")
    print(f"   Match: {decision == SlotDecision.BLOCKED_CORRELATION}")
    print(f"   Correlation: {record.correlation_with_portfolio:.3f}")
    print(f"   Threshold: {config.max_correlation:.2f}")
    print(f"   Reason: {record.block_reason}")
    
    assert decision == SlotDecision.BLOCKED_CORRELATION, "Expected BLOCKED_CORRELATION"
    print("\n✅ SCENARIO 3 PASSED: High correlation blocks new position")


async def test_scenario_4_margin_blocks():
    """Scenario 4: margin > 65% → blocks"""
    print("\n" + "="*70)
    print("SCENARIO 4: Margin > 65% → Block")
    print("="*70)
    
    # Setup
    policy_store = MockPolicyStore(has_policy=True)
    controller = ActivePositionsController(policy_store)
    
    # Simulate 2 open positions (slots available)
    portfolio = [
        PositionSnapshot(
            symbol="BTCUSDT",
            size_usd=1000,
            leverage=10,
            unrealized_pnl_usd=50,
            entry_price=45000,
            current_price=45500,
            age_seconds=3600,
            score=75.0,
        ),
    ]
    
    market_features = {
        "trend_strength": 0.5,
        "atr_pct": 1.5,
        "momentum_consistency": 0.5,
    }
    
    decision, record = await controller.evaluate_position_request(
        symbol="ETHUSDT",
        action="NEW_TRADE",
        candidate_score=80.0,
        candidate_returns=None,
        market_features=market_features,
        portfolio_positions=portfolio,
        total_margin_usage_pct=70.0,  # Exceeds 65% cap
    )
    
    print(f"\n✅ Test Result:")
    print(f"   Decision: {decision.value}")
    print(f"   Expected: BLOCKED_MARGIN")
    print(f"   Match: {decision == SlotDecision.BLOCKED_MARGIN}")
    print(f"   Margin usage: {record.margin_usage_pct:.1f}%")
    print(f"   Threshold: 65.0%")
    print(f"   Reason: {record.block_reason}")
    
    assert decision == SlotDecision.BLOCKED_MARGIN, "Expected BLOCKED_MARGIN"
    print("\n✅ SCENARIO 4 PASSED: High margin usage blocks new position")


async def test_scenario_5_no_policy_blocks():
    """Scenario 5: no policy → blocks (fail-closed)"""
    print("\n" + "="*70)
    print("SCENARIO 5: No Policy → Block (Fail-Closed)")
    print("="*70)
    
    # Setup with no policy
    policy_store = MockPolicyStore(has_policy=False)
    controller = ActivePositionsController(policy_store)
    
    portfolio = []
    
    market_features = {
        "trend_strength": 0.5,
        "atr_pct": 1.5,
        "momentum_consistency": 0.5,
    }
    
    decision, record = await controller.evaluate_position_request(
        symbol="BTCUSDT",
        action="NEW_TRADE",
        candidate_score=80.0,
        candidate_returns=None,
        market_features=market_features,
        portfolio_positions=portfolio,
        total_margin_usage_pct=20.0,
    )
    
    print(f"\n✅ Test Result:")
    print(f"   Decision: {decision.value}")
    print(f"   Expected: BLOCKED_NO_POLICY")
    print(f"   Match: {decision == SlotDecision.BLOCKED_NO_POLICY}")
    print(f"   Reason: {record.block_reason}")
    
    assert decision == SlotDecision.BLOCKED_NO_POLICY, "Expected BLOCKED_NO_POLICY"
    print("\n✅ SCENARIO 5 PASSED: Missing policy blocks (fail-closed)")


async def test_scenario_6_regime_changes_slots():
    """Scenario 6: regime changes → slot count adjusts"""
    print("\n" + "="*70)
    print("SCENARIO 6: Regime Changes → Slot Count Adjusts")
    print("="*70)
    
    # Setup
    policy_store = MockPolicyStore(has_policy=True)
    controller = ActivePositionsController(policy_store)
    
    portfolio = []
    
    # Test 1: TREND_STRONG → 6 slots
    market_features_strong = {
        "trend_strength": 0.80,  # > 0.75
        "atr_pct": 1.5,
        "momentum_consistency": 0.7,  # > 0.6
    }
    
    decision1, record1 = await controller.evaluate_position_request(
        symbol="BTCUSDT",
        action="NEW_TRADE",
        candidate_score=80.0,
        candidate_returns=None,
        market_features=market_features_strong,
        portfolio_positions=portfolio,
        total_margin_usage_pct=20.0,
    )
    
    print(f"\n✅ Test 1 - TREND_STRONG:")
    print(f"   Desired slots: {record1.desired_slots}")
    print(f"   Expected: 6")
    print(f"   Match: {record1.desired_slots == 6}")
    
    # Test 2: CHOP → 3 slots
    market_features_chop = {
        "trend_strength": 0.25,  # < 0.3
        "atr_pct": 1.5,
        "momentum_consistency": 0.35,  # < 0.4
    }
    
    decision2, record2 = await controller.evaluate_position_request(
        symbol="ETHUSDT",
        action="NEW_TRADE",
        candidate_score=80.0,
        candidate_returns=None,
        market_features=market_features_chop,
        portfolio_positions=portfolio,
        total_margin_usage_pct=20.0,
    )
    
    print(f"\n✅ Test 2 - CHOP:")
    print(f"   Desired slots: {record2.desired_slots}")
    print(f"   Expected: 3")
    print(f"   Match: {record2.desired_slots == 3}")
    
    # Test 3: VOLATILITY_SPIKE → 3 slots
    market_features_spike = {
        "trend_strength": 0.50,
        "atr_pct": 3.0,  # > 2.5
        "momentum_consistency": 0.50,
    }
    
    decision3, record3 = await controller.evaluate_position_request(
        symbol="SOLUSDT",
        action="NEW_TRADE",
        candidate_score=80.0,
        candidate_returns=None,
        market_features=market_features_spike,
        portfolio_positions=portfolio,
        total_margin_usage_pct=20.0,
    )
    
    print(f"\n✅ Test 3 - VOLATILITY_SPIKE:")
    print(f"   Desired slots: {record3.desired_slots}")
    print(f"   Expected: 3")
    print(f"   Match: {record3.desired_slots == 3}")
    
    assert record1.desired_slots == 6, "Expected 6 slots for TREND_STRONG"
    assert record2.desired_slots == 3, "Expected 3 slots for CHOP"
    assert record3.desired_slots == 3, "Expected 3 slots for VOLATILITY_SPIKE"
    
    print("\n✅ SCENARIO 6 PASSED: Regime changes adjust slot count")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run all test scenarios"""
    print("\n" + "="*70)
    print("ACTIVE POSITIONS CONTROLLER - PROOF SCRIPT")
    print("="*70)
    print("\nDemonstrating:")
    print("1. Slots full → blocks new opens")
    print("2. New score 20% higher → rotation triggers")
    print("3. Correlation > 80% → blocks")
    print("4. Margin > 65% → blocks")
    print("5. No policy → blocks (fail-closed)")
    print("6. Regime changes → slot count adjusts")
    
    try:
        await test_scenario_1_slots_full_blocks()
        await test_scenario_2_rotation_triggers()
        await test_scenario_3_correlation_blocks()
        await test_scenario_4_margin_blocks()
        await test_scenario_5_no_policy_blocks()
        await test_scenario_6_regime_changes_slots()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nKey Achievements:")
        print("  ✅ No hardcoded symbols (reads from PolicyStore)")
        print("  ✅ Dynamic slot allocation (3-6 based on regime)")
        print("  ✅ Capital rotation (close weakest for better)")
        print("  ✅ Correlation caps (prevents >80% corr)")
        print("  ✅ Margin caps (prevents >65% usage)")
        print("  ✅ Fail-closed (blocks if no policy)")
        print("  ✅ Grep-friendly logs (ACTIVE_SLOTS, BLOCKED_*, ROTATION_CLOSE, CORR_BLOCK)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
