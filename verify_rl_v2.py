"""
RL v2 Verification Script
=========================

Tests all RL v2 components to verify functionality.

Usage:
    python verify_rl_v2.py
"""

import sys
import time
from typing import Dict, Any


def test_reward_engine():
    """Test Reward Engine v2."""
    print("\n" + "="*60)
    print("Testing Reward Engine v2")
    print("="*60)
    
    from backend.services.ai.rl_reward_engine_v2 import get_reward_engine
    
    reward_engine = get_reward_engine()
    
    # Test meta strategy reward
    meta_reward = reward_engine.calculate_meta_strategy_reward(
        pnl_pct=3.5,
        max_drawdown_pct=1.2,
        current_regime="TREND",
        predicted_regime="TREND",
        confidence=0.85,
        trace_id="test-meta-001"
    )
    
    print(f"✓ Meta Strategy Reward: {meta_reward:.4f}")
    assert isinstance(meta_reward, float), "Meta reward should be float"
    
    # Test position sizing reward
    size_reward = reward_engine.calculate_position_sizing_reward(
        pnl_pct=2.8,
        leverage=4,
        position_size_usd=3000.0,
        account_balance=10000.0,
        market_volatility=0.022,
        trace_id="test-size-001"
    )
    
    print(f"✓ Position Sizing Reward: {size_reward:.4f}")
    assert isinstance(size_reward, float), "Size reward should be float"
    
    print("✅ Reward Engine v2: PASSED")
    return True


def test_state_manager():
    """Test State Manager v2."""
    print("\n" + "="*60)
    print("Testing State Manager v2")
    print("="*60)
    
    from backend.services.ai.rl_state_manager_v2 import get_state_manager
    
    state_manager = get_state_manager()
    
    # Test meta strategy state
    meta_state = state_manager.build_meta_strategy_state(
        regime="TREND",
        confidence=0.75,
        market_price=52500.0,
        account_balance=10000.0,
        trace_id="test-meta-state-001"
    )
    
    print(f"✓ Meta Strategy State: {meta_state}")
    assert "regime" in meta_state, "State should have regime"
    assert "volatility" in meta_state, "State should have volatility"
    assert "market_pressure" in meta_state, "State should have market_pressure"
    
    # Test position sizing state
    size_state = state_manager.build_position_sizing_state(
        signal_confidence=0.8,
        portfolio_exposure=0.35,
        market_volatility=0.02,
        account_balance=10000.0,
        trace_id="test-size-state-001"
    )
    
    print(f"✓ Position Sizing State: {size_state}")
    assert "signal_confidence" in size_state, "State should have signal_confidence"
    assert "portfolio_exposure" in size_state, "State should have portfolio_exposure"
    
    # Test regime labeling
    regime = state_manager.label_regime(
        price_history=[50000, 51000, 52000, 53000, 54000],
        volume_history=None
    )
    
    print(f"✓ Regime Labeled: {regime}")
    assert regime in ["TREND", "RANGE", "BREAKOUT", "MEAN_REVERSION", "UNKNOWN"], "Valid regime"
    
    print("✅ State Manager v2: PASSED")
    return True


def test_action_space():
    """Test Action Space v2."""
    print("\n" + "="*60)
    print("Testing Action Space v2")
    print("="*60)
    
    from backend.services.ai.rl_action_space_v2 import get_action_space
    
    action_space = get_action_space()
    
    # Test meta strategy action selection
    q_values = {"TREND": 2.5, "RANGE": 1.8, "BREAKOUT": 2.1, "MEAN_REVERSION": 1.5}
    strategy = action_space.select_meta_strategy_action(q_values, epsilon=0.0)
    
    print(f"✓ Selected Strategy: {strategy}")
    assert strategy == "TREND", "Should select highest Q-value"
    
    # Test size multiplier selection
    q_values_list = [0.0] * 8
    q_values_list[4] = 2.0  # Best action at index 4 (multiplier 1.0)
    multiplier = action_space.select_size_multiplier(q_values_list, epsilon=0.0)
    
    print(f"✓ Selected Size Multiplier: {multiplier}")
    assert multiplier == 1.0, "Should select multiplier at index 4"
    
    # Test action encoding/decoding
    action_idx = action_space.encode_meta_action("TREND", "MODEL_XGB", "WEIGHT_UP")
    strategy_dec, model_dec, weight_dec = action_space.decode_meta_action(action_idx)
    
    print(f"✓ Action Encoding/Decoding: {action_idx} → {strategy_dec}, {model_dec}, {weight_dec}")
    assert strategy_dec == "TREND", "Strategy should decode correctly"
    
    # Test action space sizes
    meta_size = action_space.get_meta_action_space_size()
    size_size = action_space.get_size_action_space_size()
    
    print(f"✓ Meta Action Space Size: {meta_size}")
    print(f"✓ Size Action Space Size: {size_size}")
    assert meta_size == 48, "Meta action space should be 48"
    assert size_size == 56, "Size action space should be 56"
    
    print("✅ Action Space v2: PASSED")
    return True


def test_episode_tracker():
    """Test Episode Tracker v2."""
    print("\n" + "="*60)
    print("Testing Episode Tracker v2")
    print("="*60)
    
    from backend.services.ai.rl_episode_tracker_v2 import get_episode_tracker
    
    episode_tracker = get_episode_tracker()
    
    # Start episode
    trace_id = f"test-episode-{int(time.time())}"
    episode = episode_tracker.start_episode(trace_id, time.time())
    
    print(f"✓ Episode Started: {trace_id}")
    assert episode.episode_id == trace_id, "Episode ID should match"
    
    # Add steps
    state = {"regime": "TREND", "confidence": 0.8}
    episode_tracker.add_step(trace_id, state, "TREND", 2.5)
    episode_tracker.add_step(trace_id, state, "TREND", 3.0)
    
    print(f"✓ Steps Added: 2 steps")
    
    # End episode
    episode_tracker.end_episode(trace_id, time.time())
    
    print(f"✓ Episode Ended")
    
    # Get stats
    stats = episode_tracker.get_episode_stats()
    
    print(f"✓ Episode Stats: {stats}")
    assert stats["total_episodes"] >= 1, "Should have at least 1 episode"
    
    # Test TD-update
    new_q = episode_tracker.td_update_meta(
        state=state,
        action="TREND",
        reward=3.0,
        next_state=None,
        trace_id=trace_id
    )
    
    print(f"✓ TD-Update (Meta): Q-value = {new_q:.4f}")
    assert isinstance(new_q, float), "Q-value should be float"
    
    print("✅ Episode Tracker v2: PASSED")
    return True


def test_meta_agent():
    """Test Meta Strategy Agent v2."""
    print("\n" + "="*60)
    print("Testing Meta Strategy Agent v2")
    print("="*60)
    
    from backend.agents.rl_meta_strategy_agent_v2 import get_meta_agent
    
    meta_agent = get_meta_agent()
    
    # Set state
    trace_id = f"test-meta-{int(time.time())}"
    meta_agent.set_current_state(trace_id, {
        "regime": "TREND",
        "confidence": 0.8,
        "market_price": 52000.0,
        "account_balance": 10000.0
    })
    
    print(f"✓ State Set: {trace_id}")
    
    # Select action
    strategy = meta_agent.select_action(trace_id)
    
    print(f"✓ Action Selected: {strategy}")
    assert strategy in ["TREND", "RANGE", "BREAKOUT", "MEAN_REVERSION"], "Valid strategy"
    
    # Update with reward
    meta_agent.update(
        trace_id=trace_id,
        pnl_pct=3.5,
        max_drawdown_pct=1.0,
        current_regime="TREND",
        predicted_regime=strategy,
        confidence=0.8
    )
    
    print(f"✓ Agent Updated")
    
    # Get stats
    stats = meta_agent.get_stats()
    
    print(f"✓ Agent Stats: {stats}")
    assert stats["agent_type"] == "meta_strategy_v2", "Correct agent type"
    
    print("✅ Meta Strategy Agent v2: PASSED")
    return True


def test_size_agent():
    """Test Position Sizing Agent v2."""
    print("\n" + "="*60)
    print("Testing Position Sizing Agent v2")
    print("="*60)
    
    from backend.agents.rl_position_sizing_agent_v2 import get_size_agent
    
    size_agent = get_size_agent()
    
    # Set state
    trace_id = f"test-size-{int(time.time())}"
    size_agent.set_current_state(trace_id, {
        "confidence": 0.8,
        "portfolio_exposure": 0.3,
        "volatility": 0.02,
        "account_balance": 10000.0
    })
    
    print(f"✓ State Set: {trace_id}")
    
    # Select action
    multiplier, leverage = size_agent.select_action(trace_id)
    
    print(f"✓ Action Selected: multiplier={multiplier}, leverage={leverage}")
    assert 0.2 <= multiplier <= 1.8, "Valid multiplier"
    assert 1 <= leverage <= 7, "Valid leverage"
    
    # Update with reward
    size_agent.update(
        trace_id=trace_id,
        pnl_pct=2.8,
        leverage=leverage,
        position_size_usd=3000.0,
        account_balance=10000.0,
        market_volatility=0.02
    )
    
    print(f"✓ Agent Updated")
    
    # Get stats
    stats = size_agent.get_stats()
    
    print(f"✓ Agent Stats: {stats}")
    assert stats["agent_type"] == "position_sizing_v2", "Correct agent type"
    
    print("✅ Position Sizing Agent v2: PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RL v2 VERIFICATION SUITE")
    print("="*60)
    
    tests = [
        ("Reward Engine v2", test_reward_engine),
        ("State Manager v2", test_state_manager),
        ("Action Space v2", test_action_space),
        ("Episode Tracker v2", test_episode_tracker),
        ("Meta Strategy Agent v2", test_meta_agent),
        ("Position Sizing Agent v2", test_size_agent),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - RL v2 IS PRODUCTION READY!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        return 1


if __name__ == "__main__":
    sys.exit(main())
