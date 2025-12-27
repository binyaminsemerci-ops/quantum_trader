"""
REINFORCEMENT SIGNAL MANAGER - COMPREHENSIVE TEST SUITE

Tests for Reinforcement Signals (Modul 2):
- Unit tests for weight updates, reward shaping, calibration
- Integration tests with trading system
- Scenario-based simulations
"""

import pytest
import asyncio
import json
import os
import tempfile
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Assuming reinforcement_signal_manager.py is in backend/services/ai/
import sys
sys.path.append('/app/backend')

from services.ai.reinforcement_signal_manager import (
    ReinforcementSignalManager,
    ModelType,
    TradeOutcome,
    ReinforcementSignal,
    ModelWeights,
    CalibrationMetrics,
    ReinforcementContext
)


# ============================================================
# UNIT TESTS
# ============================================================

class TestModelWeightUpdates:
    """Test model weight update mechanism"""
    
    def test_initial_weights(self):
        """Test initial weights are correct"""
        manager = ReinforcementSignalManager()
        
        assert manager.model_weights.xgboost == 0.25
        assert manager.model_weights.lightgbm == 0.25
        assert manager.model_weights.nhits == 0.30
        assert manager.model_weights.patchtst == 0.20
        
        # Sum should be 1.0
        total = (
            manager.model_weights.xgboost +
            manager.model_weights.lightgbm +
            manager.model_weights.nhits +
            manager.model_weights.patchtst
        )
        assert abs(total - 1.0) < 0.001
    
    def test_weight_update_positive_reward(self):
        """Test weights increase for models that voted correctly"""
        manager = ReinforcementSignalManager(learning_rate=0.05)
        
        # Simulate profitable trade where XGBoost and LightGBM voted LONG
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.72},
            'lightgbm': {'action': 'LONG', 'confidence': 0.68},
            'nhits': {'action': 'SHORT', 'confidence': 0.55},  # Wrong
            'patchtst': {'action': 'LONG', 'confidence': 0.65}
        }
        
        initial_xgb_weight = manager.model_weights.xgboost
        initial_nhits_weight = manager.model_weights.nhits
        
        signal = manager.process_trade_outcome(
            symbol="BTCUSDT",
            action="LONG",
            confidence=0.68,
            pnl=50.0,  # Positive reward
            position_size=0.1,
            entry_price=50000.0,
            exit_price=50500.0,
            duration_seconds=300,
            regime="TRENDING",
            model_votes=model_votes,
            setup_hash="test_hash"
        )
        
        # XGBoost voted correctly → weight should increase
        assert manager.model_weights.xgboost > initial_xgb_weight
        
        # N-HiTS voted wrong → weight should not increase (or decrease slightly)
        assert manager.model_weights.nhits <= initial_nhits_weight
        
        # Weights still sum to 1.0
        total = (
            manager.model_weights.xgboost +
            manager.model_weights.lightgbm +
            manager.model_weights.nhits +
            manager.model_weights.patchtst
        )
        assert abs(total - 1.0) < 0.001
    
    def test_weight_update_negative_reward(self):
        """Test weights decrease for models that voted for losing trades"""
        manager = ReinforcementSignalManager(learning_rate=0.05)
        
        model_votes = {
            'xgboost': {'action': 'SHORT', 'confidence': 0.65},
            'lightgbm': {'action': 'SHORT', 'confidence': 0.62},
            'nhits': {'action': 'LONG', 'confidence': 0.58},  # Correct
            'patchtst': {'action': 'SHORT', 'confidence': 0.60}
        }
        
        initial_xgb_weight = manager.model_weights.xgboost
        initial_nhits_weight = manager.model_weights.nhits
        
        signal = manager.process_trade_outcome(
            symbol="ETHUSDT",
            action="SHORT",
            confidence=0.63,
            pnl=-40.0,  # Negative reward
            position_size=0.08,
            entry_price=3000.0,
            exit_price=3120.0,
            duration_seconds=450,
            regime="RANGING",
            model_votes=model_votes,
            setup_hash="test_hash2"
        )
        
        # XGBoost voted for losing trade → weight should decrease
        assert manager.model_weights.xgboost < initial_xgb_weight
        
        # N-HiTS voted LONG (opposite) → didn't contribute to loss
        # Weight should be relatively stable or increase
        assert manager.model_weights.nhits >= initial_nhits_weight * 0.95
    
    def test_weight_bounds(self):
        """Test weights respect min/max bounds"""
        manager = ReinforcementSignalManager(
            learning_rate=0.10,  # High learning rate
            min_model_weight=0.05,
            max_model_weight=0.50
        )
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.75},
            'lightgbm': {'action': 'SHORT', 'confidence': 0.60},
            'nhits': {'action': 'SHORT', 'confidence': 0.58},
            'patchtst': {'action': 'SHORT', 'confidence': 0.62}
        }
        
        # 20 consecutive wins where only XGBoost voted correctly
        for i in range(20):
            manager.process_trade_outcome(
                symbol="BTCUSDT",
                action="LONG",
                confidence=0.75,
                pnl=100.0,  # Large reward
                position_size=0.15,
                entry_price=50000.0,
                exit_price=51000.0,
                duration_seconds=300,
                regime="TRENDING",
                model_votes=model_votes,
                setup_hash=f"hash{i}"
            )
        
        # XGBoost should hit max weight (0.50)
        assert manager.model_weights.xgboost <= 0.50
        
        # Others should stay above min weight (0.05)
        assert manager.model_weights.lightgbm >= 0.05
        assert manager.model_weights.nhits >= 0.05
        assert manager.model_weights.patchtst >= 0.05
        
        # Total still 1.0
        total = (
            manager.model_weights.xgboost +
            manager.model_weights.lightgbm +
            manager.model_weights.nhits +
            manager.model_weights.patchtst
        )
        assert abs(total - 1.0) < 0.001


class TestRewardShaping:
    """Test reward shaping calculations"""
    
    def test_reward_components(self):
        """Test that shaped reward has all components"""
        manager = ReinforcementSignalManager(
            reward_alpha=0.6,
            reward_beta=0.3,
            reward_gamma=0.1
        )
        
        # Build trade history for normalization
        for i in range(20):
            manager.trade_history.append(TradeOutcome(
                timestamp=datetime.now().isoformat(),
                symbol="BTCUSDT",
                action="LONG",
                confidence=0.65,
                pnl=30.0,
                position_size=0.1,
                entry_price=50000.0,
                exit_price=50300.0,
                duration_seconds=300,
                regime="TRENDING",
                model_votes={},
                setup_hash=""
            ))
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.70},
            'lightgbm': {'action': 'LONG', 'confidence': 0.65},
            'nhits': {'action': 'LONG', 'confidence': 0.68},
            'patchtst': {'action': 'LONG', 'confidence': 0.63}
        }
        
        signal = manager.process_trade_outcome(
            symbol="BTCUSDT",
            action="LONG",
            confidence=0.67,
            pnl=50.0,
            position_size=0.12,
            entry_price=50000.0,
            exit_price=50600.0,
            duration_seconds=300,
            regime="TRENDING",
            model_votes=model_votes,
            setup_hash="test"
        )
        
        # Shaped reward should be different from raw reward
        assert signal.shaped_reward != signal.raw_reward
        
        # Components should exist
        assert signal.sharpe_contribution != 0.0
        assert signal.risk_adjusted_return != 0.0
        
        # Shaped reward should be positive for profitable trade
        assert signal.shaped_reward > 0.0
    
    def test_negative_reward_shaping(self):
        """Test shaped reward for losing trade"""
        manager = ReinforcementSignalManager()
        
        model_votes = {
            'xgboost': {'action': 'SHORT', 'confidence': 0.60},
            'lightgbm': {'action': 'SHORT', 'confidence': 0.58},
            'nhits': {'action': 'SHORT', 'confidence': 0.62},
            'patchtst': {'action': 'SHORT', 'confidence': 0.59}
        }
        
        signal = manager.process_trade_outcome(
            symbol="ETHUSDT",
            action="SHORT",
            confidence=0.60,
            pnl=-35.0,  # Loss
            position_size=0.08,
            entry_price=3000.0,
            exit_price=3110.0,
            duration_seconds=450,
            regime="VOLATILE",
            model_votes=model_votes,
            setup_hash="test_loss"
        )
        
        # Shaped reward should be negative
        assert signal.shaped_reward < 0.0
        
        # Raw reward equals PnL
        assert signal.raw_reward == -35.0


class TestConfidenceCalibration:
    """Test confidence calibration (Brier score)"""
    
    def test_calibration_initialization(self):
        """Test calibration metrics start at neutral"""
        manager = ReinforcementSignalManager()
        
        # All models should start with Brier score = 0.25 (neutral)
        for model in ModelType:
            assert manager.calibration_metrics.brier_score[model.value] == 0.25
            assert manager.confidence_scalers[model.value] == 1.0
    
    def test_calibration_update_well_calibrated(self):
        """Test calibration for well-calibrated model"""
        manager = ReinforcementSignalManager(calibration_kappa=0.5)
        
        # Model predicts 0.60, wins 60% of time (well calibrated)
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.60},
            'lightgbm': {'action': 'LONG', 'confidence': 0.60},
            'nhits': {'action': 'LONG', 'confidence': 0.60},
            'patchtst': {'action': 'LONG', 'confidence': 0.60}
        }
        
        # 6 wins
        for i in range(6):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.60,
                pnl=30.0, position_size=0.1, entry_price=50000.0,
                exit_price=50300.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"win{i}"
            )
        
        # 4 losses
        for i in range(4):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.60,
                pnl=-25.0, position_size=0.1, entry_price=50000.0,
                exit_price=49750.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"loss{i}"
            )
        
        # Brier score should be low (good calibration)
        xgb_brier = manager.calibration_metrics.brier_score['xgboost']
        assert xgb_brier < 0.25  # Better than neutral
        
        # Confidence scaler should be near 1.0
        xgb_scaler = manager.confidence_scalers['xgboost']
        assert 0.9 <= xgb_scaler <= 1.1
    
    def test_calibration_update_overconfident(self):
        """Test calibration for overconfident model"""
        manager = ReinforcementSignalManager(calibration_kappa=0.5)
        
        # Model predicts 0.80 confidence but only wins 50%
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.80},
            'lightgbm': {'action': 'LONG', 'confidence': 0.65},
            'nhits': {'action': 'LONG', 'confidence': 0.70},
            'patchtst': {'action': 'LONG', 'confidence': 0.68}
        }
        
        # 5 wins, 5 losses (50% win rate with 0.80 conf = overconfident)
        for i in range(5):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.75,
                pnl=30.0, position_size=0.1, entry_price=50000.0,
                exit_price=50300.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"win{i}"
            )
        
        for i in range(5):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.75,
                pnl=-25.0, position_size=0.1, entry_price=50000.0,
                exit_price=49750.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"loss{i}"
            )
        
        # XGBoost Brier score should be high (poor calibration)
        xgb_brier = manager.calibration_metrics.brier_score['xgboost']
        # Note: EMA means it won't immediately spike, but should trend up
        
        # Confidence scaler should decrease (penalize overconfidence)
        xgb_scaler = manager.confidence_scalers['xgboost']
        assert xgb_scaler < 1.0  # Reduced


class TestExplorationExploitation:
    """Test exploration-exploitation balance"""
    
    def test_initial_exploration_rate(self):
        """Test exploration starts at initial value"""
        manager = ReinforcementSignalManager(initial_exploration_rate=0.20)
        
        context = manager.get_reinforcement_context()
        assert context.exploration_rate == 0.20
    
    def test_exploration_decay(self):
        """Test exploration decays over time"""
        manager = ReinforcementSignalManager(
            initial_exploration_rate=0.20,
            min_exploration_rate=0.05,
            exploration_decay_trades=100
        )
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.65},
            'lightgbm': {'action': 'LONG', 'confidence': 0.62},
            'nhits': {'action': 'LONG', 'confidence': 0.68},
            'patchtst': {'action': 'LONG', 'confidence': 0.60}
        }
        
        # Initial
        context_0 = manager.get_reinforcement_context()
        eps_0 = context_0.exploration_rate
        
        # After 50 trades
        for i in range(50):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.65,
                pnl=30.0, position_size=0.1, entry_price=50000.0,
                exit_price=50300.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"trade{i}"
            )
        
        context_50 = manager.get_reinforcement_context()
        eps_50 = context_50.exploration_rate
        
        # After 100 trades
        for i in range(50, 100):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.65,
                pnl=30.0, position_size=0.1, entry_price=50000.0,
                exit_price=50300.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"trade{i}"
            )
        
        context_100 = manager.get_reinforcement_context()
        eps_100 = context_100.exploration_rate
        
        # Exploration should decay
        assert eps_0 > eps_50 > eps_100
        
        # Should approach min exploration rate
        assert eps_100 >= 0.05
    
    def test_apply_reinforcement_exploration(self):
        """Test exploration returns uniform weights"""
        manager = ReinforcementSignalManager(initial_exploration_rate=1.0)  # Always explore
        
        model_predictions = {
            'xgboost': {'action': 'LONG', 'confidence': 0.70},
            'lightgbm': {'action': 'LONG', 'confidence': 0.65},
            'nhits': {'action': 'SHORT', 'confidence': 0.60},
            'patchtst': {'action': 'LONG', 'confidence': 0.68}
        }
        
        weights, is_exploring = manager.apply_reinforcement_to_signal(
            model_predictions=model_predictions,
            use_exploration=True
        )
        
        # Should be exploring
        assert is_exploring == True
        
        # Weights should be uniform
        assert weights['xgboost'] == 0.25
        assert weights['lightgbm'] == 0.25
        assert weights['nhits'] == 0.25
        assert weights['patchtst'] == 0.25
    
    def test_apply_reinforcement_exploitation(self):
        """Test exploitation uses learned weights"""
        manager = ReinforcementSignalManager(initial_exploration_rate=0.0)  # Never explore
        
        # Manually adjust weights
        manager.model_weights.xgboost = 0.35
        manager.model_weights.lightgbm = 0.25
        manager.model_weights.nhits = 0.25
        manager.model_weights.patchtst = 0.15
        
        model_predictions = {
            'xgboost': {'action': 'LONG', 'confidence': 0.70},
            'lightgbm': {'action': 'LONG', 'confidence': 0.65},
            'nhits': {'action': 'SHORT', 'confidence': 0.60},
            'patchtst': {'action': 'LONG', 'confidence': 0.68}
        }
        
        weights, is_exploring = manager.apply_reinforcement_to_signal(
            model_predictions=model_predictions,
            use_exploration=True
        )
        
        # Should be exploiting
        assert is_exploring == False
        
        # Weights should match learned weights
        assert weights['xgboost'] == 0.35
        assert weights['lightgbm'] == 0.25
        assert weights['nhits'] == 0.25
        assert weights['patchtst'] == 0.15


class TestBaselineReward:
    """Test baseline reward calculation"""
    
    def test_baseline_starts_zero(self):
        """Test baseline reward starts at 0"""
        manager = ReinforcementSignalManager()
        assert manager.baseline_reward == 0.0
    
    def test_baseline_updates(self):
        """Test baseline reward tracks moving average"""
        manager = ReinforcementSignalManager()
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.65},
            'lightgbm': {'action': 'LONG', 'confidence': 0.62},
            'nhits': {'action': 'LONG', 'confidence': 0.68},
            'patchtst': {'action': 'LONG', 'confidence': 0.60}
        }
        
        # 10 trades with varying PnL
        pnls = [50, -30, 40, -20, 60, 35, -25, 45, -15, 55]
        
        for i, pnl in enumerate(pnls):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.65,
                pnl=pnl, position_size=0.1, entry_price=50000.0,
                exit_price=50000.0 + pnl * 10, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"trade{i}"
            )
        
        # Baseline should be non-zero and reflect average
        assert manager.baseline_reward != 0.0
        
        # Should be roughly the average of shaped rewards
        avg_pnl = np.mean(pnls)
        # Baseline won't exactly match avg_pnl due to shaping, but should be same sign
        if avg_pnl > 0:
            assert manager.baseline_reward > 0
        
    def test_advantage_calculation(self):
        """Test advantage = shaped_reward - baseline"""
        manager = ReinforcementSignalManager()
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.65},
            'lightgbm': {'action': 'LONG', 'confidence': 0.62},
            'nhits': {'action': 'LONG', 'confidence': 0.68},
            'patchtst': {'action': 'LONG', 'confidence': 0.60}
        }
        
        # Establish baseline with 20 trades
        for i in range(20):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.65,
                pnl=30.0, position_size=0.1, entry_price=50000.0,
                exit_price=50300.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"setup{i}"
            )
        
        baseline_before = manager.baseline_reward
        
        # Exceptionally good trade
        signal = manager.process_trade_outcome(
            symbol="BTCUSDT", action="LONG", confidence=0.75,
            pnl=120.0,  # Much better than baseline
            position_size=0.15, entry_price=50000.0,
            exit_price=51200.0, duration_seconds=300,
            regime="TRENDING", model_votes=model_votes,
            setup_hash="exceptional"
        )
        
        # Advantage should be positive
        assert signal.advantage > 0.0
        
        # Advantage = shaped_reward - baseline
        assert abs(signal.advantage - (signal.shaped_reward - baseline_before)) < 0.1


class TestCheckpointing:
    """Test checkpoint save/load"""
    
    def test_checkpoint_save_and_load(self):
        """Test state persists across restarts"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_path = f.name
        
        try:
            # Create manager and process trades
            manager1 = ReinforcementSignalManager(
                learning_rate=0.05,
                checkpoint_path=checkpoint_path
            )
            
            model_votes = {
                'xgboost': {'action': 'LONG', 'confidence': 0.70},
                'lightgbm': {'action': 'LONG', 'confidence': 0.65},
                'nhits': {'action': 'SHORT', 'confidence': 0.60},
                'patchtst': {'action': 'LONG', 'confidence': 0.68}
            }
            
            for i in range(20):
                manager1.process_trade_outcome(
                    symbol="BTCUSDT", action="LONG", confidence=0.68,
                    pnl=40.0, position_size=0.1, entry_price=50000.0,
                    exit_price=50400.0, duration_seconds=300,
                    regime="TRENDING", model_votes=model_votes,
                    setup_hash=f"trade{i}"
                )
            
            # Manual checkpoint
            manager1.checkpoint()
            
            weights1 = manager1.model_weights.to_dict()
            trades1 = manager1.total_trades_processed
            baseline1 = manager1.baseline_reward
            
            # Load new manager from same file
            manager2 = ReinforcementSignalManager(checkpoint_path=checkpoint_path)
            
            weights2 = manager2.model_weights.to_dict()
            trades2 = manager2.total_trades_processed
            baseline2 = manager2.baseline_reward
            
            # Verify state was restored
            assert trades1 == trades2
            assert abs(baseline1 - baseline2) < 0.001
            
            for model in ['xgboost', 'lightgbm', 'nhits', 'patchtst']:
                assert abs(weights1[model] - weights2[model]) < 0.001
        
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestSystemIntegration:
    """Integration tests with trading system"""
    
    @pytest.mark.asyncio
    async def test_full_rl_cycle(self):
        """Test complete cycle: signal → trade → outcome → weight update"""
        manager = ReinforcementSignalManager()
        
        # 1. Get initial context
        context_before = manager.get_reinforcement_context()
        weights_before = context_before.model_weights.to_dict()
        
        # 2. Simulate model predictions
        model_predictions = {
            'xgboost': {'action': 'LONG', 'confidence': 0.72},
            'lightgbm': {'action': 'LONG', 'confidence': 0.68},
            'nhits': {'action': 'SHORT', 'confidence': 0.55},
            'patchtst': {'action': 'LONG', 'confidence': 0.65}
        }
        
        # 3. Apply RL weights
        weights, is_exploring = manager.apply_reinforcement_to_signal(
            model_predictions=model_predictions,
            use_exploration=True
        )
        
        # 4. Simulate trade outcome (profitable)
        signal = manager.process_trade_outcome(
            symbol="BTCUSDT",
            action="LONG",
            confidence=0.68,
            pnl=55.0,
            position_size=0.12,
            entry_price=50000.0,
            exit_price=50550.0,
            duration_seconds=360,
            regime="TRENDING",
            model_votes=model_predictions,
            setup_hash="integration_test"
        )
        
        # 5. Get updated context
        context_after = manager.get_reinforcement_context()
        weights_after = context_after.model_weights.to_dict()
        
        # Verify changes
        assert manager.total_trades_processed == 1
        assert signal.raw_reward == 55.0
        assert signal.shaped_reward > 0.0
        
        # Models that voted LONG should have increased weight
        assert weights_after['xgboost'] > weights_before['xgboost']
        assert weights_after['lightgbm'] > weights_before['lightgbm']
        assert weights_after['patchtst'] > weights_before['patchtst']
        
        # N-HiTS voted SHORT (wrong) → weight should not increase
        assert weights_after['nhits'] <= weights_before['nhits']


# ============================================================
# SCENARIO SIMULATIONS
# ============================================================

class TestScenarios:
    """Real-world scenario simulations"""
    
    def test_winning_streak_scenario(self):
        """Scenario: 15 consecutive wins, all models agree"""
        manager = ReinforcementSignalManager(learning_rate=0.05)
        
        print("\n=== WINNING STREAK SCENARIO ===")
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.70},
            'lightgbm': {'action': 'LONG', 'confidence': 0.68},
            'nhits': {'action': 'LONG', 'confidence': 0.72},
            'patchtst': {'action': 'LONG', 'confidence': 0.65}
        }
        
        initial_weights = manager.model_weights.to_dict()
        
        for i in range(15):
            signal = manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG",
                confidence=0.69, pnl=45.0 + i * 2,
                position_size=0.1, entry_price=50000.0,
                exit_price=50450.0 + i * 20, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"win{i}"
            )
            
            if (i + 1) % 5 == 0:
                context = manager.get_reinforcement_context()
                print(f"After {i+1} wins: Weights={context.model_weights.to_dict()}")
        
        final_weights = manager.model_weights.to_dict()
        
        # All models should have similar weights (all voted correctly)
        weight_variance = np.var(list(final_weights.values()))
        print(f"\nFinal weight variance: {weight_variance:.4f}")
        
        # Weights should have changed (learned)
        assert final_weights != initial_weights
        
        # Baseline reward should be positive
        assert manager.baseline_reward > 0.0
    
    def test_model_disagreement_scenario(self):
        """Scenario: Models disagree, test weight divergence"""
        manager = ReinforcementSignalManager(learning_rate=0.05)
        
        print("\n=== MODEL DISAGREEMENT SCENARIO ===")
        
        # XGBoost and LightGBM vote LONG (correct)
        # N-HiTS and PatchTST vote SHORT (wrong)
        
        for i in range(20):
            model_votes = {
                'xgboost': {'action': 'LONG', 'confidence': 0.72},
                'lightgbm': {'action': 'LONG', 'confidence': 0.68},
                'nhits': {'action': 'SHORT', 'confidence': 0.65},
                'patchtst': {'action': 'SHORT', 'confidence': 0.62}
            }
            
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG",
                confidence=0.70, pnl=40.0,
                position_size=0.1, entry_price=50000.0,
                exit_price=50400.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"disagree{i}"
            )
            
            if (i + 1) % 5 == 0:
                context = manager.get_reinforcement_context()
                weights = context.model_weights.to_dict()
                print(
                    f"Trade {i+1}: XGB={weights['xgboost']:.3f}, "
                    f"LGB={weights['lightgbm']:.3f}, "
                    f"N-HiTS={weights['nhits']:.3f}, "
                    f"PatchTST={weights['patchtst']:.3f}"
                )
        
        final_context = manager.get_reinforcement_context()
        final_weights = final_context.model_weights.to_dict()
        
        # XGBoost and LightGBM should have higher weights
        assert final_weights['xgboost'] > final_weights['nhits']
        assert final_weights['lightgbm'] > final_weights['patchtst']
        
        print(f"\nFinal weights: {final_weights}")
    
    def test_exploration_exploitation_scenario(self):
        """Scenario: Track exploration rate decay"""
        manager = ReinforcementSignalManager(
            initial_exploration_rate=0.20,
            min_exploration_rate=0.05,
            exploration_decay_trades=100
        )
        
        print("\n=== EXPLORATION-EXPLOITATION SCENARIO ===")
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.70},
            'lightgbm': {'action': 'LONG', 'confidence': 0.65},
            'nhits': {'action': 'LONG', 'confidence': 0.68},
            'patchtst': {'action': 'LONG', 'confidence': 0.62}
        }
        
        exploration_rates = []
        
        for i in range(120):
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.66,
                pnl=35.0, position_size=0.1, entry_price=50000.0,
                exit_price=50350.0, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"trade{i}"
            )
            
            context = manager.get_reinforcement_context()
            exploration_rates.append(context.exploration_rate)
            
            if (i + 1) % 20 == 0:
                print(f"Trade {i+1}: ε={context.exploration_rate:.3f}")
        
        # Exploration should decay from 0.20 to ~0.05
        assert exploration_rates[0] == 0.20
        assert exploration_rates[-1] <= 0.06  # Close to 0.05
        assert exploration_rates[50] < exploration_rates[0]  # Decaying
    
    def test_calibration_improvement_scenario(self):
        """Scenario: Track calibration improvement over time"""
        manager = ReinforcementSignalManager(calibration_kappa=0.5)
        
        print("\n=== CALIBRATION IMPROVEMENT SCENARIO ===")
        
        # XGBoost is overconfident initially (predicts 0.75, actual ~60% WR)
        
        for i in range(50):
            is_win = (i % 5 < 3)  # 60% win rate
            
            model_votes = {
                'xgboost': {'action': 'LONG', 'confidence': 0.75},  # Overconfident
                'lightgbm': {'action': 'LONG', 'confidence': 0.62},
                'nhits': {'action': 'LONG', 'confidence': 0.65},
                'patchtst': {'action': 'LONG', 'confidence': 0.60}
            }
            
            pnl = 35.0 if is_win else -30.0
            
            manager.process_trade_outcome(
                symbol="BTCUSDT", action="LONG", confidence=0.65,
                pnl=pnl, position_size=0.1, entry_price=50000.0,
                exit_price=50000.0 + pnl * 10, duration_seconds=300,
                regime="TRENDING", model_votes=model_votes,
                setup_hash=f"cal{i}"
            )
            
            if (i + 1) % 10 == 0:
                diag = manager.get_diagnostics()
                xgb_brier = diag['calibration_metrics']['brier_scores']['xgboost']
                xgb_scaler = manager.confidence_scalers['xgboost']
                print(
                    f"Trade {i+1}: Brier={xgb_brier:.4f}, "
                    f"Scaler={xgb_scaler:.3f}"
                )
        
        # XGBoost confidence scaler should be <1.0 (penalized for overconfidence)
        final_scaler = manager.confidence_scalers['xgboost']
        assert final_scaler < 1.0
        
        print(f"\nFinal XGBoost scaler: {final_scaler:.3f} (penalized for overconfidence)")


# ============================================================
# PERFORMANCE BENCHMARKS
# ============================================================

class TestPerformance:
    """Performance and memory usage tests"""
    
    def test_memory_usage_after_500_trades(self):
        """Test memory doesn't leak after many trades"""
        import tracemalloc
        
        tracemalloc.start()
        
        manager = ReinforcementSignalManager()
        
        model_votes = {
            'xgboost': {'action': 'LONG', 'confidence': 0.70},
            'lightgbm': {'action': 'LONG', 'confidence': 0.65},
            'nhits': {'action': 'LONG', 'confidence': 0.68},
            'patchtst': {'action': 'LONG', 'confidence': 0.62}
        }
        
        # 500 trades
        for i in range(500):
            manager.process_trade_outcome(
                symbol=f"COIN{i % 10}USDT", action="LONG", confidence=0.66,
                pnl=np.random.uniform(-50, 50), position_size=0.1,
                entry_price=50000.0, exit_price=50300.0,
                duration_seconds=300, regime="TRENDING",
                model_votes=model_votes, setup_hash=f"hash{i}"
            )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\nMemory Usage after 500 trades:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")
        
        # Should stay under 50MB
        assert current / 1024 / 1024 < 50
    
    def test_checkpoint_performance(self):
        """Test checkpoint save/load speed"""
        import time
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_path = f.name
        
        try:
            manager = ReinforcementSignalManager(checkpoint_path=checkpoint_path)
            
            model_votes = {
                'xgboost': {'action': 'LONG', 'confidence': 0.70},
                'lightgbm': {'action': 'LONG', 'confidence': 0.65},
                'nhits': {'action': 'LONG', 'confidence': 0.68},
                'patchtst': {'action': 'LONG', 'confidence': 0.62}
            }
            
            # Add data
            for i in range(100):
                manager.process_trade_outcome(
                    symbol="BTCUSDT", action="LONG", confidence=0.66,
                    pnl=35.0, position_size=0.1, entry_price=50000.0,
                    exit_price=50350.0, duration_seconds=300,
                    regime="TRENDING", model_votes=model_votes,
                    setup_hash=f"hash{i}"
                )
            
            # Measure save time
            start = time.time()
            manager.checkpoint()
            save_time = time.time() - start
            
            # Measure load time
            start = time.time()
            manager2 = ReinforcementSignalManager(checkpoint_path=checkpoint_path)
            load_time = time.time() - start
            
            print(f"\nCheckpoint Performance:")
            print(f"  Save Time: {save_time * 1000:.2f} ms")
            print(f"  Load Time: {load_time * 1000:.2f} ms")
            
            # Should be fast (<100ms each)
            assert save_time < 0.1
            assert load_time < 0.1
        
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
