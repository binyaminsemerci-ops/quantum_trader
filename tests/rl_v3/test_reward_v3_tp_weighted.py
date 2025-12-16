"""
Tests for RL v3 reward function with configurable TP weight.

Validates that:
1. tp_reward_weight parameter works correctly
2. Default behavior (weight=1.0) matches previous baseline
3. Higher weights increase TP accuracy reward impact
4. Lower weights decrease TP accuracy reward impact
"""

import pytest
from backend.domains.learning.rl_v3.reward_v3 import compute_reward


class TestTPRewardWeightDefault:
    """Test default behavior (weight=1.0)."""
    
    def test_default_weight_matches_baseline(self):
        """Default tp_reward_weight=1.0 maintains previous behavior."""
        # Scenario: Good TP accuracy
        reward_default = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.75,
            tp_reward_weight=1.0
        )
        
        # Same call without explicit weight (defaults to 1.0)
        reward_implicit = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.75
        )
        
        assert reward_default == reward_implicit
    
    def test_zero_tp_accuracy_no_bonus(self):
        """No TP accuracy → no TP bonus regardless of weight."""
        reward = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.0,
            tp_reward_weight=2.0  # High weight but no accuracy
        )
        
        # Should not include TP bonus (no if branch entered)
        # Expected: base PnL (1.0) - drawdown (0.125) + regime (1.6) + survival (0.1) = 2.575
        assert pytest.approx(reward, rel=0.1) == 2.575


class TestTPRewardWeightHigher:
    """Test increased tp_reward_weight (e.g. 2.0 for low hit rate)."""
    
    def test_double_weight_doubles_tp_bonus(self):
        """tp_reward_weight=2.0 doubles TP accuracy bonus."""
        # Baseline with default weight
        reward_baseline = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.80,
            tp_reward_weight=1.0
        )
        
        # With doubled weight
        reward_doubled = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.80,
            tp_reward_weight=2.0
        )
        
        # TP bonus calculation: tp_zone_accuracy * 5.0 * tp_reward_weight
        # Baseline: 0.80 * 5.0 * 1.0 = 4.0
        # Doubled:  0.80 * 5.0 * 2.0 = 8.0
        # Difference: 4.0
        assert pytest.approx(reward_doubled - reward_baseline, abs=0.01) == 4.0
    
    def test_high_weight_encourages_tp_accuracy(self):
        """High tp_reward_weight makes TP accuracy more valuable."""
        # Scenario: Agent learning TP prediction with high weight
        # Good TP accuracy should yield much higher reward
        
        reward_poor_tp = compute_reward(
            pnl_delta=0.005,
            drawdown=0.03,
            position_size=0.4,
            regime_alignment=0.5,
            volatility=0.02,
            tp_zone_accuracy=0.20,  # Poor TP prediction
            tp_reward_weight=2.5  # High weight to encourage improvement
        )
        
        reward_good_tp = compute_reward(
            pnl_delta=0.005,
            drawdown=0.03,
            position_size=0.4,
            regime_alignment=0.5,
            volatility=0.02,
            tp_zone_accuracy=0.90,  # Excellent TP prediction
            tp_reward_weight=2.5
        )
        
        # TP bonus difference: (0.90 - 0.20) * 5.0 * 2.5 = 0.70 * 12.5 = 8.75
        assert reward_good_tp > reward_poor_tp
        assert pytest.approx(reward_good_tp - reward_poor_tp, abs=0.1) == 8.75


class TestTPRewardWeightLower:
    """Test decreased tp_reward_weight (e.g. 0.5 for high hit rate but low R)."""
    
    def test_half_weight_halves_tp_bonus(self):
        """tp_reward_weight=0.5 halves TP accuracy bonus."""
        reward_baseline = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.70,
            tp_reward_weight=1.0
        )
        
        reward_halved = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.70,
            tp_reward_weight=0.5
        )
        
        # TP bonus difference: 0.70 * 5.0 * (1.0 - 0.5) = 1.75
        assert pytest.approx(reward_baseline - reward_halved, abs=0.01) == 1.75
    
    def test_low_weight_deprioritizes_tp_accuracy(self):
        """Low tp_reward_weight makes TP accuracy less important."""
        # Scenario: High hit rate but low R → deprioritize TP accuracy
        # Focus agent on other aspects (PnL, regime alignment)
        
        reward_poor_tp = compute_reward(
            pnl_delta=0.01,
            drawdown=0.02,
            position_size=0.3,
            regime_alignment=0.9,
            volatility=0.02,
            tp_zone_accuracy=0.30,
            tp_reward_weight=0.3  # Low weight - TP accuracy not critical
        )
        
        reward_good_tp = compute_reward(
            pnl_delta=0.01,
            drawdown=0.02,
            position_size=0.3,
            regime_alignment=0.9,
            volatility=0.02,
            tp_zone_accuracy=0.85,
            tp_reward_weight=0.3
        )
        
        # TP bonus difference: (0.85 - 0.30) * 5.0 * 0.3 = 0.55 * 1.5 = 0.825
        # Much smaller difference than with default weight
        assert reward_good_tp > reward_poor_tp
        assert pytest.approx(reward_good_tp - reward_poor_tp, abs=0.1) == 0.825


class TestTPRewardWeightScenarios:
    """Test realistic CLM v3 feedback scenarios."""
    
    def test_low_hit_rate_scenario_high_weight(self):
        """
        CLM detects low hit rate (30%) + good R (3.3).
        Sets tp_reward_weight=2.0 to encourage better TP prediction.
        """
        # Agent with poor TP accuracy
        reward_poor = compute_reward(
            pnl_delta=0.008,
            drawdown=0.04,
            position_size=0.45,
            regime_alignment=0.7,
            volatility=0.025,
            tp_zone_accuracy=0.25,
            tp_reward_weight=2.0  # CLM increased weight
        )
        
        # Agent improves TP accuracy
        reward_improved = compute_reward(
            pnl_delta=0.008,
            drawdown=0.04,
            position_size=0.45,
            regime_alignment=0.7,
            volatility=0.025,
            tp_zone_accuracy=0.65,  # Better prediction
            tp_reward_weight=2.0
        )
        
        # Improvement should be rewarded strongly
        improvement = reward_improved - reward_poor
        # TP bonus delta: (0.65 - 0.25) * 5.0 * 2.0 = 4.0
        assert improvement > 3.5
        assert pytest.approx(improvement, abs=0.5) == 4.0
    
    def test_high_hit_rate_low_r_scenario_low_weight(self):
        """
        CLM detects high hit rate (85%) + low R (1.18).
        Sets tp_reward_weight=0.5 to deprioritize TP accuracy.
        """
        # Focus on PnL and regime alignment instead of TP
        reward = compute_reward(
            pnl_delta=0.012,  # Good PnL more important
            drawdown=0.03,
            position_size=0.4,
            regime_alignment=0.95,  # Good regime alignment rewarded
            volatility=0.02,
            tp_zone_accuracy=0.70,  # TP accuracy less critical
            tp_reward_weight=0.5  # CLM decreased weight
        )
        
        # TP bonus should be moderate: 0.70 * 5.0 * 0.5 = 1.75
        # Expected total: PnL (1.2) - drawdown (0.045) + regime (1.9) + survival (0.1) + TP (1.75)
        # ≈ 4.905
        assert pytest.approx(reward, rel=0.1) == 4.9
    
    def test_optimal_performance_default_weight(self):
        """
        CLM detects optimal metrics (55% hit rate, 1.82R).
        Keeps tp_reward_weight=1.0 (default behavior).
        """
        reward = compute_reward(
            pnl_delta=0.01,
            drawdown=0.04,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.75,
            tp_reward_weight=1.0  # CLM keeps default
        )
        
        # TP bonus: 0.75 * 5.0 * 1.0 = 3.75
        # Expected total: 1.0 - 0.08 + 1.6 + 0.1 + 3.75 = 6.37
        assert pytest.approx(reward, rel=0.1) == 6.4


class TestTPRewardWeightEdgeCases:
    """Test edge cases for tp_reward_weight."""
    
    def test_zero_weight_eliminates_tp_bonus(self):
        """tp_reward_weight=0.0 completely eliminates TP bonus."""
        reward = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=1.0,  # Perfect TP accuracy
            tp_reward_weight=0.0  # Zero weight
        )
        
        # TP bonus: 1.0 * 5.0 * 0.0 = 0.0
        # Should be same as no TP accuracy
        reward_no_tp = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.0,
            tp_reward_weight=0.0
        )
        
        assert reward == reward_no_tp
    
    def test_very_high_weight(self):
        """tp_reward_weight=5.0 makes TP accuracy dominant."""
        reward = compute_reward(
            pnl_delta=0.002,  # Small PnL
            drawdown=0.01,
            position_size=0.3,
            regime_alignment=0.3,
            volatility=0.02,
            tp_zone_accuracy=0.95,  # Excellent TP
            tp_reward_weight=5.0  # Very high weight
        )
        
        # TP bonus dominates: 0.95 * 5.0 * 5.0 = 23.75
        assert reward > 20  # TP bonus should be dominant component


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_omitting_tp_reward_weight_uses_default(self):
        """Omitting tp_reward_weight uses default 1.0."""
        # This is how existing code calls compute_reward
        reward = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8,
            volatility=0.02,
            tp_zone_accuracy=0.70
            # tp_reward_weight not provided
        )
        
        # Should use default weight=1.0
        # TP bonus: 0.70 * 5.0 * 1.0 = 3.5
        assert reward > 0  # Should compute successfully
    
    def test_all_parameters_default(self):
        """Test with minimal parameters (backward compatibility)."""
        reward = compute_reward(
            pnl_delta=0.01,
            drawdown=0.05,
            position_size=0.5,
            regime_alignment=0.8
            # Optional params use defaults
        )
        
        assert reward > 0  # Should work with defaults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
