"""
RL v2 Pipeline Integration Test
================================

Tests complete RL v2 pipeline with domain architecture.

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import pytest
from pathlib import Path
from backend.domains.learning.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
from backend.domains.learning.rl_v2.position_sizing_agent_v2 import PositionSizingAgentV2
from backend.domains.learning.rl_v2.state_builder_v2 import StateBuilderV2
from backend.domains.learning.rl_v2.action_space_v2 import ActionSpaceV2
from backend.domains.learning.rl_v2.reward_engine_v2 import RewardEngineV2
from backend.domains.learning.rl_v2.episode_tracker_v2 import EpisodeTrackerV2
from backend.domains.learning.rl_v2.q_learning_core import QLearningCore


class TestRLv2Pipeline:
    """Test RL v2 complete pipeline."""
    
    def test_meta_strategy_agent_select_and_update(self):
        """Test meta strategy agent action selection and update."""
        agent = MetaStrategyAgentV2(
            alpha=0.01,
            gamma=0.99,
            epsilon=0.1
        )
        
        # Market data
        market_data = {
            "symbol": "BTCUSDT",
            "price_history": [50000, 51000, 52000],
            "volume_history": [1000, 1100, 1200],
            "account_balance": 10000.0,
            "confidence": 0.75
        }
        
        # Select action
        action = agent.select_action(market_data)
        
        # Verify action
        assert action is not None
        assert action.strategy in ["dual_momentum", "mean_reversion", "momentum_flip"]
        assert action.model in ["lstm", "gru", "transformer", "ensemble"]
        assert 0.5 <= action.weight <= 1.5
        
        # Result data
        result_data = {
            **market_data,
            "pnl": 100.0,
            "pnl_percentage": 1.0,
            "drawdown": 0.05,
            "sharpe_ratio": 1.5,
            "regime": "TREND",
            "predicted_regime": "TREND"
        }
        
        # Update agent
        agent.update(result_data)
        
        # Verify stats
        stats = agent.get_stats()
        assert stats["q_learning_stats"]["update_count"] > 0
        
        print(f"✅ Meta strategy agent test passed: {action.strategy}/{action.model}")
    
    def test_position_sizing_agent_select_and_update(self):
        """Test position sizing agent action selection and update."""
        agent = PositionSizingAgentV2(
            alpha=0.01,
            gamma=0.99,
            epsilon=0.1
        )
        
        # Market data
        market_data = {
            "symbol": "BTCUSDT",
            "confidence": 0.75,
            "portfolio_exposure": 0.3,
            "volatility": 0.02,
            "equity_history": [10000, 10100, 10200],
            "recent_trades": [
                {"pnl": 100, "result": "win"},
                {"pnl": -50, "result": "loss"}
            ],
            "account_balance": 10200.0
        }
        
        # Select action
        action = agent.select_action(market_data)
        
        # Verify action
        assert action is not None
        assert 0.5 <= action.size_multiplier <= 2.0
        assert 5 <= action.leverage <= 50
        
        # Result data
        result_data = {
            **market_data,
            "pnl": 150.0,
            "pnl_percentage": 1.5,
            "leverage": action.leverage,
            "position_size_usd": 1000.0,
            "risk_penalty": 0.1
        }
        
        # Update agent
        agent.update(result_data)
        
        # Verify stats
        stats = agent.get_stats()
        assert stats["q_learning_stats"]["update_count"] > 0
        
        print(f"✅ Position sizing agent test passed: {action.size_multiplier}x @ {action.leverage}x")
    
    def test_complete_rl_v2_pipeline(self):
        """Test complete RL v2 pipeline with both agents."""
        # Initialize components
        state_builder = StateBuilderV2()
        action_space = ActionSpaceV2()
        reward_engine = RewardEngineV2()
        episode_tracker = EpisodeTrackerV2(gamma=0.99)
        q_learning = QLearningCore(alpha=0.01, gamma=0.99, epsilon=0.1)
        
        meta_agent = MetaStrategyAgentV2(alpha=0.01, gamma=0.99, epsilon=0.1)
        sizing_agent = PositionSizingAgentV2(alpha=0.01, gamma=0.99, epsilon=0.1)
        
        # Simulate trading episode
        market_data = {
            "symbol": "BTCUSDT",
            "price_history": [50000, 51000, 52000, 51500, 52500],
            "volume_history": [1000, 1100, 1200, 1150, 1250],
            "account_balance": 10000.0,
            "equity_history": [10000, 10100, 10200, 10150, 10300],
            "recent_trades": [
                {"pnl": 100, "result": "win"},
                {"pnl": 80, "result": "win"},
                {"pnl": -50, "result": "loss"}
            ],
            "confidence": 0.75,
            "portfolio_exposure": 0.3,
            "volatility": 0.02
        }
        
        # Start episode
        episode_id = episode_tracker.start_episode()
        assert episode_id > 0
        
        # Meta strategy decision
        meta_action = meta_agent.select_action(market_data)
        assert meta_action is not None
        
        # Position sizing decision
        sizing_action = sizing_agent.select_action(market_data)
        assert sizing_action is not None
        
        # Simulate trade result
        result_data = {
            **market_data,
            "pnl": 200.0,
            "pnl_percentage": 2.0,
            "drawdown": 0.03,
            "sharpe_ratio": 1.8,
            "regime": "TREND",
            "predicted_regime": "TREND",
            "leverage": sizing_action.leverage,
            "position_size_usd": 1500.0,
            "risk_penalty": 0.08
        }
        
        # Calculate rewards
        meta_reward = reward_engine.calculate_meta_strategy_reward(result_data)
        sizing_reward = reward_engine.calculate_position_sizing_reward(result_data)
        
        assert meta_reward != 0.0
        assert sizing_reward != 0.0
        
        # Update agents
        meta_agent.update(result_data)
        sizing_agent.update(result_data)
        
        # Record episode step
        state = state_builder.build_meta_strategy_state(market_data)
        action = action_space.action_to_dict(meta_action)
        
        episode_tracker.record_step(
            state=state,
            action=action,
            reward=meta_reward
        )
        
        # End episode
        discounted_return = episode_tracker.end_episode()
        assert discounted_return != 0.0
        
        # Verify stats
        meta_stats = meta_agent.get_stats()
        sizing_stats = sizing_agent.get_stats()
        episode_stats = episode_tracker.get_episode_stats()
        
        assert meta_stats["q_learning_stats"]["update_count"] > 0
        assert sizing_stats["q_learning_stats"]["update_count"] > 0
        assert episode_stats["total_episodes"] > 0
        
        print("✅ Complete RL v2 pipeline test passed")
        print(f"   Meta strategy: {meta_action.strategy}/{meta_action.model}")
        print(f"   Position sizing: {sizing_action.size_multiplier}x @ {sizing_action.leverage}x")
        print(f"   Meta reward: {meta_reward:.4f}")
        print(f"   Sizing reward: {sizing_reward:.4f}")
        print(f"   Discounted return: {discounted_return:.4f}")


if __name__ == "__main__":
    # Run tests
    test_suite = TestRLv2Pipeline()
    
    print("=" * 60)
    print("RL v2 Pipeline Integration Tests")
    print("=" * 60)
    
    test_suite.test_meta_strategy_agent_select_and_update()
    test_suite.test_position_sizing_agent_select_and_update()
    test_suite.test_complete_rl_v2_pipeline()
    
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
