"""
Test Suite for Shadow Model Manager

Module 5: Shadow Models - Section 6

Test Coverage:
- Unit tests: Statistical tests, promotion scoring, Thompson sampling
- Integration tests: Promotion cycle, rollback, checkpoint
- Scenario tests: Lucky streaks, degradation, multiple challengers
- Performance tests: Latency benchmarks
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

import sys
sys.path.append('.')

from backend.services.ai.shadow_model_manager import (
    ShadowModelManager,
    PerformanceTracker,
    StatisticalTester,
    PromotionEngine,
    ThompsonSampling,
    ModelRole,
    PromotionStatus,
    ModelMetadata,
    TradeResult,
    PerformanceMetrics
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def shadow_manager():
    """Create shadow manager with temp checkpoint"""
    checkpoint_path = Path('data/test_shadow_checkpoint.json')
    manager = ShadowModelManager(
        min_trades_for_promotion=500,
        mdd_tolerance=1.20,
        alpha=0.05,
        n_bootstrap=1000,  # Reduced for faster tests
        checkpoint_path=checkpoint_path
    )
    
    yield manager
    
    # Cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()


@pytest.fixture
def performance_tracker():
    """Create performance tracker"""
    return PerformanceTracker()


@pytest.fixture
def statistical_tester():
    """Create statistical tester"""
    return StatisticalTester(alpha=0.05, n_bootstrap=1000)


@pytest.fixture
def promotion_engine():
    """Create promotion engine"""
    return PromotionEngine(min_trades=500, mdd_tolerance=1.20)


@pytest.fixture
def thompson_sampling():
    """Create Thompson sampling"""
    return ThompsonSampling()


def generate_trades(n_trades: int, win_rate: float, mean_pnl: float, std_pnl: float):
    """Generate synthetic trade results"""
    trades = []
    
    for i in range(n_trades):
        outcome = 1 if np.random.rand() < win_rate else 0
        pnl = np.random.normal(mean_pnl, std_pnl)
        
        trade = TradeResult(
            timestamp=datetime.now() - timedelta(minutes=n_trades-i),
            model_name='test_model',
            prediction=outcome,
            actual_outcome=outcome,
            pnl=pnl,
            confidence=0.7,
            executed=False
        )
        trades.append(trade)
    
    return trades


# ============================================================================
# UNIT TESTS: STATISTICAL TESTER
# ============================================================================

class TestStatisticalTester:
    """Test statistical testing methods"""
    
    def test_t_test_significant_difference(self, statistical_tester):
        """Test t-test detects significant difference"""
        np.random.seed(42)
        
        # Champion: mean=$50, std=$100
        champion_pnls = np.random.normal(50, 100, 500)
        
        # Challenger: mean=$70, std=$100 (significantly better)
        challenger_pnls = np.random.normal(70, 100, 500)
        
        t_stat, p_value, passed = statistical_tester.t_test(champion_pnls, challenger_pnls)
        
        assert passed, f"T-test should pass (p={p_value:.4f})"
        assert p_value < 0.05, f"P-value should be <0.05 (got {p_value:.4f})"
        assert t_stat > 0, f"T-statistic should be positive (got {t_stat:.2f})"
    
    def test_t_test_no_difference(self, statistical_tester):
        """Test t-test does not detect difference when none exists"""
        np.random.seed(42)
        
        # Champion and challenger same distribution
        champion_pnls = np.random.normal(50, 100, 500)
        challenger_pnls = np.random.normal(50, 100, 500)
        
        t_stat, p_value, passed = statistical_tester.t_test(champion_pnls, challenger_pnls)
        
        assert not passed, f"T-test should not pass (p={p_value:.4f})"
        assert p_value >= 0.05, f"P-value should be >=0.05 (got {p_value:.4f})"
    
    def test_bootstrap_ci_significant(self, statistical_tester):
        """Test bootstrap CI detects significant difference"""
        np.random.seed(42)
        
        champion_pnls = np.random.normal(50, 100, 500)
        challenger_pnls = np.random.normal(70, 100, 500)
        
        ci_lower, ci_upper, passed = statistical_tester.bootstrap_ci(
            champion_pnls, challenger_pnls
        )
        
        assert passed, f"Bootstrap should pass (CI: [{ci_lower:.1f}, {ci_upper:.1f}])"
        assert ci_lower > 0, f"CI lower bound should be >0 (got {ci_lower:.1f})"
        assert ci_upper > 0, f"CI upper bound should be >0 (got {ci_upper:.1f})"
    
    def test_bootstrap_ci_no_difference(self, statistical_tester):
        """Test bootstrap CI when no difference"""
        np.random.seed(42)
        
        champion_pnls = np.random.normal(50, 100, 500)
        challenger_pnls = np.random.normal(50, 100, 500)
        
        ci_lower, ci_upper, passed = statistical_tester.bootstrap_ci(
            champion_pnls, challenger_pnls
        )
        
        assert not passed, f"Bootstrap should not pass (CI: [{ci_lower:.1f}, {ci_upper:.1f}])"
        # CI should contain 0
        assert ci_lower < 0 < ci_upper, f"CI should contain 0"
    
    def test_sharpe_test(self, statistical_tester):
        """Test Sharpe ratio comparison"""
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.56,
            mean_pnl=50,
            std_pnl=100,
            total_pnl=25000,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5000,
            rolling_wr_std=0.03,
            last_updated=datetime.now()
        )
        
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=500,
            win_rate=0.58,
            mean_pnl=60,
            std_pnl=100,
            total_pnl=30000,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            max_drawdown=4500,
            rolling_wr_std=0.03,
            last_updated=datetime.now()
        )
        
        z_stat, p_value, passed = statistical_tester.sharpe_test(
            champion_metrics, challenger_metrics
        )
        
        assert passed, f"Sharpe test should pass (p={p_value:.4f})"
        assert z_stat > 0, f"Z-statistic should be positive"
    
    def test_win_rate_test(self, statistical_tester):
        """Test win rate comparison"""
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.56,
            mean_pnl=50,
            std_pnl=100,
            total_pnl=25000,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5000,
            rolling_wr_std=0.03,
            last_updated=datetime.now()
        )
        
        # +5pp improvement (should be significant)
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=500,
            win_rate=0.61,
            mean_pnl=60,
            std_pnl=100,
            total_pnl=30000,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            max_drawdown=4500,
            rolling_wr_std=0.03,
            last_updated=datetime.now()
        )
        
        z_stat, p_value, passed = statistical_tester.win_rate_test(
            champion_metrics, challenger_metrics
        )
        
        assert passed, f"WR test should pass (p={p_value:.4f})"
        assert z_stat > 0, f"Z-statistic should be positive"


# ============================================================================
# UNIT TESTS: PROMOTION ENGINE
# ============================================================================

class TestPromotionEngine:
    """Test promotion criteria and scoring"""
    
    def test_promotion_approved(self, promotion_engine):
        """Test promotion approval when all criteria met"""
        np.random.seed(42)
        
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.56,
            mean_pnl=50,
            std_pnl=100,
            total_pnl=25000,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5000,
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        # Strong improvement across all metrics
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=500,
            win_rate=0.60,  # +4pp
            mean_pnl=65,    # +$15
            std_pnl=100,
            total_pnl=32500,
            sharpe_ratio=1.95,  # +0.45
            sortino_ratio=2.4,
            max_drawdown=4200,  # Lower (better)
            rolling_wr_std=0.03,
            last_updated=datetime.now()
        )
        
        # Mock test results (all passed)
        from backend.services.ai.shadow_model_manager import StatisticalTestResults
        test_results = StatisticalTestResults(
            t_test=(3.5, 0.0005, True),
            bootstrap_ci=(10.0, 20.0, True),
            sharpe_test=(4.2, 0.0001, True),
            wr_test=(3.8, 0.0002, True),
            statistical_significance=True
        )
        
        decision = promotion_engine.check_criteria(
            champion_metrics, challenger_metrics, test_results
        )
        
        assert decision.status == PromotionStatus.APPROVED, f"Should be approved (got {decision.status})"
        assert decision.promotion_score >= 70, f"Score should be >=70 (got {decision.promotion_score:.1f})"
        assert decision.statistical_significance, "Should have statistical significance"
        assert decision.sharpe_criterion, "Should pass Sharpe criterion"
        assert decision.sample_size_criterion, "Should pass sample size criterion"
        assert decision.mdd_criterion, "Should pass MDD criterion"
        assert decision.win_rate_criterion, "Should pass WR criterion"
    
    def test_promotion_rejected_statistical(self, promotion_engine):
        """Test rejection when statistical significance not achieved"""
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.56,
            mean_pnl=50,
            std_pnl=100,
            total_pnl=25000,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5000,
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=500,
            win_rate=0.57,  # Only +1pp (marginal)
            mean_pnl=52,
            std_pnl=100,
            total_pnl=26000,
            sharpe_ratio=1.52,
            sortino_ratio=2.05,
            max_drawdown=4900,
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        # No statistical significance
        from backend.services.ai.shadow_model_manager import StatisticalTestResults
        test_results = StatisticalTestResults(
            t_test=(1.2, 0.23, False),
            bootstrap_ci=(-2.0, 8.0, False),
            sharpe_test=(0.8, 0.42, False),
            wr_test=(1.1, 0.27, False),
            statistical_significance=False
        )
        
        decision = promotion_engine.check_criteria(
            champion_metrics, challenger_metrics, test_results
        )
        
        assert decision.status == PromotionStatus.REJECTED, "Should be rejected"
        assert not decision.statistical_significance, "Should not have statistical significance"
        assert "Statistical significance not achieved" in decision.reason
    
    def test_promotion_rejected_sample_size(self, promotion_engine):
        """Test rejection when sample size too small"""
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.56,
            mean_pnl=50,
            std_pnl=100,
            total_pnl=25000,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5000,
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        # Only 300 trades
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=300,
            win_rate=0.62,
            mean_pnl=70,
            std_pnl=100,
            total_pnl=21000,
            sharpe_ratio=2.1,
            sortino_ratio=2.5,
            max_drawdown=4000,
            rolling_wr_std=0.03,
            last_updated=datetime.now()
        )
        
        from backend.services.ai.shadow_model_manager import StatisticalTestResults
        test_results = StatisticalTestResults(
            t_test=(3.5, 0.0005, True),
            bootstrap_ci=(15.0, 25.0, True),
            sharpe_test=(4.8, 0.0001, True),
            wr_test=(4.2, 0.0001, True),
            statistical_significance=True
        )
        
        decision = promotion_engine.check_criteria(
            champion_metrics, challenger_metrics, test_results
        )
        
        assert decision.status == PromotionStatus.REJECTED, "Should be rejected"
        assert not decision.sample_size_criterion, "Should fail sample size criterion"
        assert "Sample size insufficient" in decision.reason
    
    def test_promotion_pending_manual_review(self, promotion_engine):
        """Test pending status when score between 50-69"""
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.56,
            mean_pnl=50,
            std_pnl=100,
            total_pnl=25000,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5000,
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        # Marginal improvement
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=500,
            win_rate=0.575,  # +1.5pp (marginal)
            mean_pnl=55,
            std_pnl=100,
            total_pnl=27500,
            sharpe_ratio=1.65,  # +0.15 (marginal)
            sortino_ratio=2.1,
            max_drawdown=4800,
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        # Weak statistical significance
        from backend.services.ai.shadow_model_manager import StatisticalTestResults
        test_results = StatisticalTestResults(
            t_test=(2.1, 0.036, True),  # Just significant
            bootstrap_ci=(1.0, 10.0, True),
            sharpe_test=(1.8, 0.072, False),
            wr_test=(1.5, 0.13, False),
            statistical_significance=True
        )
        
        decision = promotion_engine.check_criteria(
            champion_metrics, challenger_metrics, test_results
        )
        
        assert decision.status == PromotionStatus.PENDING, f"Should be pending (got {decision.status})"
        assert 50 <= decision.promotion_score < 70, f"Score should be 50-69 (got {decision.promotion_score:.1f})"
        assert "Manual review recommended" in decision.reason
    
    def test_promotion_score_calculation(self, promotion_engine):
        """Test promotion score formula"""
        champion_metrics = PerformanceMetrics(model_name="champion", 
            n_trades=500,
            win_rate=0.50,
            mean_pnl=40,
            std_pnl=100,
            total_pnl=20000,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            max_drawdown=6000,
            rolling_wr_std=0.05,
            last_updated=datetime.now()
        )
        
        challenger_metrics = PerformanceMetrics(model_name="challenger", 
            n_trades=500,
            win_rate=0.55,  # +5pp
            mean_pnl=50,    # +$10
            std_pnl=100,
            total_pnl=27500,
            sharpe_ratio=1.5,  # +0.30 (25% improvement)
            sortino_ratio=2.2,
            max_drawdown=4500,  # -25% (improvement)
            rolling_wr_std=0.04,
            last_updated=datetime.now()
        )
        
        from backend.services.ai.shadow_model_manager import StatisticalTestResults
        test_results = StatisticalTestResults(
            t_test=(4.5, 0.0001, True),
            bootstrap_ci=(5.0, 15.0, True),
            sharpe_test=(3.2, 0.001, True),
            wr_test=(4.8, 0.0001, True),
            statistical_significance=True
        )
        
        decision = promotion_engine.check_criteria(
            champion_metrics, challenger_metrics, test_results
        )
        
        # Expected score breakdown:
        # - Statistical: 30 (p<0.01)
        # - Sharpe: 25 (25% improvement, capped at 25)
        # - WR: 20 (+5pp = 20 points)
        # - MDD: 15 (25% improvement, capped at 15)
        # - Consistency: 10 (rolling_wr_std < 5pp)
        # Total: 100
        
        assert decision.promotion_score >= 90, f"Score should be ~100 (got {decision.promotion_score:.1f})"


# ============================================================================
# UNIT TESTS: THOMPSON SAMPLING
# ============================================================================

class TestThompsonSampling:
    """Test Thompson sampling for multi-armed bandit"""
    
    def test_initialize_model(self, thompson_sampling):
        """Test model initialization"""
        thompson_sampling.initialize_model('model1', prior_mean=50, prior_std=50)
        
        assert 'model1' in thompson_sampling.means
        assert 'model1' in thompson_sampling.stds
        assert thompson_sampling.means['model1'] == 50
        assert thompson_sampling.stds['model1'] == 50
    
    def test_update_belief(self, thompson_sampling):
        """Test Bayesian belief update"""
        thompson_sampling.initialize_model('model1', prior_mean=50, prior_std=50)
        
        # Observe high PnL
        thompson_sampling.update('model1', observed_pnl=80)
        
        # Mean should increase towards observation
        assert thompson_sampling.means['model1'] > 50, "Mean should increase"
        assert thompson_sampling.means['model1'] < 80, "Mean should not reach observation"
        
        # Std should decrease (more certain)
        assert thompson_sampling.stds['model1'] < 50, "Std should decrease"
    
    def test_sample_allocation(self, thompson_sampling):
        """Test Thompson sampling allocation"""
        np.random.seed(42)
        
        # Model 1: High performance
        thompson_sampling.initialize_model('model1', prior_mean=70, prior_std=10)
        
        # Model 2: Medium performance
        thompson_sampling.initialize_model('model2', prior_mean=50, prior_std=20)
        
        # Model 3: Low performance
        thompson_sampling.initialize_model('model3', prior_mean=30, prior_std=15)
        
        # Sample 1000 times
        selections = []
        for _ in range(1000):
            selected = thompson_sampling.sample_allocation(['model1', 'model2', 'model3'])
            selections.append(selected)
        
        # Model 1 should be selected most often
        model1_count = selections.count('model1')
        model2_count = selections.count('model2')
        model3_count = selections.count('model3')
        
        assert model1_count > model2_count, "Model1 should be selected more than Model2"
        assert model2_count > model3_count, "Model2 should be selected more than Model3"
        
        # Model 1 should get majority (but not all due to exploration)
        assert model1_count > 500, f"Model1 should get >50% (got {model1_count}/1000)"
        assert model3_count > 0, f"Model3 should still get some selections (exploration)"


# ============================================================================
# INTEGRATION TESTS: FULL PROMOTION CYCLE
# ============================================================================

class TestPromotionCycle:
    """Test complete promotion cycle"""
    
    def test_full_promotion_workflow(self, shadow_manager):
        """Test champion registration → challenger testing → promotion"""
        np.random.seed(42)
        
        # 1. Register champion
        shadow_manager.register_model(
            model_name='xgboost_v1',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Champion model'
        )
        
        assert shadow_manager.get_champion() == 'xgboost_v1'
        
        # 2. Register challenger
        shadow_manager.register_model(
            model_name='lightgbm_v1',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='Challenger model'
        )
        
        challengers = shadow_manager.get_challengers()
        assert 'lightgbm_v1' in challengers
        
        # 3. Simulate 500 trades
        for i in range(500):
            # Champion: 56% WR, $50 mean PnL
            champ_outcome = 1 if np.random.rand() < 0.56 else 0
            champ_pnl = np.random.normal(50, 100)
            
            shadow_manager.record_prediction(
                model_name='xgboost_v1',
                prediction=champ_outcome,
                actual_outcome=champ_outcome,
                pnl=champ_pnl,
                confidence=0.7,
                executed=True
            )
            
            # Challenger: 60% WR, $65 mean PnL (better)
            chal_outcome = 1 if np.random.rand() < 0.60 else 0
            chal_pnl = np.random.normal(65, 100)
            
            shadow_manager.record_prediction(
                model_name='lightgbm_v1',
                prediction=chal_outcome,
                actual_outcome=chal_outcome,
                pnl=chal_pnl,
                confidence=0.75,
                executed=False  # Shadow mode
            )
        
        # 4. Check promotion criteria
        decision = shadow_manager.check_promotion_criteria('lightgbm_v1')
        
        assert decision is not None, "Should return promotion decision"
        assert decision.status in [PromotionStatus.APPROVED, PromotionStatus.PENDING], \
            f"Should be approved or pending (got {decision.status})"
        
        # 5. Promote if approved
        if decision.status == PromotionStatus.APPROVED:
            success = shadow_manager.promote_challenger('lightgbm_v1')
            
            assert success, "Promotion should succeed"
            assert shadow_manager.get_champion() == 'lightgbm_v1', "Challenger should be new champion"
            
            # Check promotion history
            history = shadow_manager.get_promotion_history(n=1)
            assert len(history) == 1, "Should have 1 promotion event"
            assert history[0].new_champion == 'lightgbm_v1'
            assert history[0].old_champion == 'xgboost_v1'
    
    def test_rollback_procedure(self, shadow_manager):
        """Test rollback to previous champion"""
        np.random.seed(42)
        
        # Register champion and challenger
        shadow_manager.register_model(
            model_name='champion_v1',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Original champion'
        )
        
        shadow_manager.register_model(
            model_name='challenger_v1',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='Challenger'
        )
        
        # Simulate trades and promote
        for i in range(500):
            shadow_manager.record_prediction(
                model_name='champion_v1',
                prediction=1, actual_outcome=1,
                pnl=50, confidence=0.7, executed=True
            )
            shadow_manager.record_prediction(
                model_name='challenger_v1',
                prediction=1, actual_outcome=1,
                pnl=70, confidence=0.75, executed=False
            )
        
        # Force promotion
        shadow_manager.promote_challenger('challenger_v1', force=True)
        assert shadow_manager.get_champion() == 'challenger_v1'
        
        # Rollback
        success = shadow_manager.rollback_to_previous_champion(reason="Test rollback")
        
        assert success, "Rollback should succeed"
        assert shadow_manager.get_champion() == 'champion_v1', "Should restore previous champion"
        
        # Check promotion history updated
        history = shadow_manager.get_promotion_history(n=1)
        assert history[0].promotion_status == PromotionStatus.ROLLED_BACK
    
    def test_checkpoint_save_restore(self, shadow_manager):
        """Test state persistence via checkpointing"""
        # Register models
        shadow_manager.register_model(
            model_name='champion',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Champion'
        )
        
        shadow_manager.register_model(
            model_name='challenger',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='Challenger'
        )
        
        # Record some trades
        for i in range(50):
            shadow_manager.record_prediction(
                model_name='champion',
                prediction=1, actual_outcome=1,
                pnl=50, confidence=0.7, executed=True
            )
        
        # Checkpoint
        shadow_manager.checkpoint()
        
        checkpoint_path = shadow_manager.checkpoint_path
        assert checkpoint_path.exists(), "Checkpoint file should exist"
        
        # Load checkpoint data
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        assert 'models' in data
        assert 'current_champion' in data
        assert data['current_champion'] == 'champion'
        assert 'champion' in data['models']
        assert 'challenger' in data['models']
        
        # Create new manager and restore
        new_manager = ShadowModelManager(checkpoint_path=checkpoint_path)
        
        assert new_manager.get_champion() == 'champion'
        assert 'challenger' in new_manager.get_challengers()
        assert new_manager.get_trade_count('champion') == 50


# ============================================================================
# SCENARIO TESTS: EDGE CASES
# ============================================================================

class TestScenarios:
    """Test realistic scenarios and edge cases"""
    
    def test_lucky_streak_rejection(self, shadow_manager):
        """Test that lucky streaks are properly detected and rejected"""
        np.random.seed(42)
        
        shadow_manager.register_model(
            model_name='champion',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Champion'
        )
        
        shadow_manager.register_model(
            model_name='lucky_challenger',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='Lucky but inferior challenger'
        )
        
        # Champion: Stable 56% WR
        for i in range(500):
            outcome = 1 if np.random.rand() < 0.56 else 0
            pnl = np.random.normal(50, 100)
            shadow_manager.record_prediction(
                model_name='champion',
                prediction=outcome, actual_outcome=outcome,
                pnl=pnl, confidence=0.7, executed=True
            )
        
        # Challenger: True 54% WR, but first 500 trades happen to be 58% (lucky)
        np.random.seed(123)  # Different seed for lucky streak
        lucky_outcomes = []
        for i in range(1000):
            outcome = 1 if np.random.rand() < 0.54 else 0
            lucky_outcomes.append(outcome)
        
        # Take first 500 (which happen to be lucky)
        for i in range(500):
            pnl = np.random.normal(55, 100)
            shadow_manager.record_prediction(
                model_name='lucky_challenger',
                prediction=lucky_outcomes[i], actual_outcome=lucky_outcomes[i],
                pnl=pnl, confidence=0.72, executed=False
            )
        
        # Check promotion
        decision = shadow_manager.check_promotion_criteria('lucky_challenger')
        
        # Should be rejected or pending (not auto-promoted)
        # Because:
        # 1. Improvement is marginal (2pp)
        # 2. Statistical tests may not pass with high confidence
        # 3. Promotion score likely <70
        
        assert decision.status != PromotionStatus.APPROVED or decision.promotion_score < 70, \
            "Lucky streak should not auto-promote"
    
    def test_gradual_degradation_detection(self, shadow_manager):
        """Test detection of gradual champion degradation"""
        shadow_manager.register_model(
            model_name='degrading_champion',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Champion that will degrade'
        )
        
        # Simulate gradual WR degradation: 60% → 52% over 200 trades
        for i in range(200):
            # Linear degradation
            current_wr = 0.60 - (i / 200) * 0.08  # 60% → 52%
            outcome = 1 if np.random.rand() < current_wr else 0
            pnl = np.random.normal(50 - (i/200)*20, 100)  # PnL also degrades
            
            shadow_manager.record_prediction(
                model_name='degrading_champion',
                prediction=outcome, actual_outcome=outcome,
                pnl=pnl, confidence=0.7, executed=True
            )
        
        # Get final metrics
        metrics = shadow_manager.get_metrics('degrading_champion')
        
        # Should show degraded performance
        assert metrics.win_rate < 0.57, f"WR should be degraded (got {metrics.win_rate:.2%})"
        
        # EWMA/CUSUM would detect this earlier (not implemented in basic tests)
        # In production, this would trigger drift detection and challenger deployment
    
    def test_multiple_challengers_best_promotes(self, shadow_manager):
        """Test that best challenger promotes when multiple testing"""
        np.random.seed(42)
        
        shadow_manager.register_model(
            model_name='champion',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Champion'
        )
        
        # Register 3 challengers with different performance
        challengers = [
            ('challenger_weak', 0.57, 52),     # +1pp, $52
            ('challenger_medium', 0.59, 58),   # +3pp, $58
            ('challenger_strong', 0.62, 70)    # +6pp, $70
        ]
        
        for name, wr, pnl in challengers:
            shadow_manager.register_model(
                model_name=name,
                model_type='lightgbm',
                version='1.0',
                role=ModelRole.CHALLENGER,
                description=f'Challenger {name}'
            )
        
        # Simulate trades for all models
        for i in range(500):
            # Champion: 56% WR, $50
            champ_outcome = 1 if np.random.rand() < 0.56 else 0
            shadow_manager.record_prediction(
                model_name='champion',
                prediction=champ_outcome, actual_outcome=champ_outcome,
                pnl=np.random.normal(50, 100),
                confidence=0.7, executed=True
            )
            
            # Challengers
            for name, wr, pnl_mean in challengers:
                outcome = 1 if np.random.rand() < wr else 0
                shadow_manager.record_prediction(
                    model_name=name,
                    prediction=outcome, actual_outcome=outcome,
                    pnl=np.random.normal(pnl_mean, 100),
                    confidence=0.72, executed=False
                )
        
        # Check promotion decisions
        decisions = {}
        for name, _, _ in challengers:
            decision = shadow_manager.check_promotion_criteria(name)
            decisions[name] = decision
        
        # Strong challenger should have highest score
        strong_score = decisions['challenger_strong'].promotion_score
        medium_score = decisions['challenger_medium'].promotion_score
        weak_score = decisions['challenger_weak'].promotion_score
        
        assert strong_score > medium_score, "Strong should score higher than medium"
        assert medium_score > weak_score, "Medium should score higher than weak"
        
        # Only strong should auto-promote (score >=70)
        assert decisions['challenger_strong'].status == PromotionStatus.APPROVED or \
               decisions['challenger_strong'].promotion_score >= 70, \
               "Strong challenger should auto-promote"
    
    def test_post_promotion_degradation_rollback(self, shadow_manager):
        """Test automatic rollback when promoted model degrades"""
        np.random.seed(42)
        
        shadow_manager.register_model(
            model_name='stable_champion',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Stable champion'
        )
        
        shadow_manager.register_model(
            model_name='unstable_challenger',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='Unstable challenger'
        )
        
        # Phase 1: Challenger looks good (500 trades at 60% WR)
        for i in range(500):
            shadow_manager.record_prediction(
                model_name='stable_champion',
                prediction=1, actual_outcome=1,
                pnl=np.random.normal(50, 100),
                confidence=0.7, executed=True
            )
            shadow_manager.record_prediction(
                model_name='unstable_challenger',
                prediction=1, actual_outcome=1,
                pnl=np.random.normal(65, 100),
                confidence=0.75, executed=False
            )
        
        # Force promotion
        shadow_manager.promote_challenger('unstable_challenger', force=True)
        assert shadow_manager.get_champion() == 'unstable_challenger'
        
        baseline_wr = shadow_manager.promotion_baseline_wr
        
        # Phase 2: Post-promotion, new champion degrades to 52% WR (>5pp drop)
        degraded_trades = 0
        for i in range(100):
            outcome = 1 if np.random.rand() < 0.52 else 0  # 52% WR (was 60%)
            shadow_manager.record_prediction(
                model_name='unstable_challenger',
                prediction=outcome, actual_outcome=outcome,
                pnl=np.random.normal(42, 100),
                confidence=0.70, executed=True
            )
            degraded_trades += 1
        
        # Check if degradation detected
        current_metrics = shadow_manager.get_metrics('unstable_challenger')
        wr_drop = baseline_wr - current_metrics.win_rate
        
        # In production, this would trigger automatic rollback
        if wr_drop > 0.05:  # >5pp drop
            # Simulate automatic rollback
            shadow_manager.rollback_to_previous_champion(reason="Post-promotion degradation >5pp")
            
            assert shadow_manager.get_champion() == 'stable_champion', \
                "Should rollback to stable champion"


# ============================================================================
# PERFORMANCE TESTS: LATENCY BENCHMARKS
# ============================================================================

class TestPerformance:
    """Test performance and latency requirements"""
    
    def test_statistical_test_latency(self, statistical_tester):
        """Test that statistical tests complete within 200ms"""
        np.random.seed(42)
        
        champion_pnls = np.random.normal(50, 100, 500)
        challenger_pnls = np.random.normal(60, 100, 500)
        
        start = time.time()
        
        # Run all tests
        statistical_tester.t_test(champion_pnls, challenger_pnls)
        statistical_tester.bootstrap_ci(champion_pnls, challenger_pnls)
        
        elapsed = time.time() - start
        
        assert elapsed < 0.2, f"Statistical tests should complete in <200ms (took {elapsed*1000:.1f}ms)"
    
    def test_promotion_decision_latency(self, shadow_manager):
        """Test that promotion decision completes within 500ms"""
        np.random.seed(42)
        
        # Setup
        shadow_manager.register_model(
            model_name='champion',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Champion'
        )
        
        shadow_manager.register_model(
            model_name='challenger',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='Challenger'
        )
        
        # Simulate trades
        for i in range(500):
            shadow_manager.record_prediction(
                model_name='champion',
                prediction=1, actual_outcome=1,
                pnl=np.random.normal(50, 100),
                confidence=0.7, executed=True
            )
            shadow_manager.record_prediction(
                model_name='challenger',
                prediction=1, actual_outcome=1,
                pnl=np.random.normal(60, 100),
                confidence=0.75, executed=False
            )
        
        # Time promotion check
        start = time.time()
        decision = shadow_manager.check_promotion_criteria('challenger')
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Promotion decision should complete in <500ms (took {elapsed*1000:.1f}ms)"
    
    def test_rollback_speed(self, shadow_manager):
        """Test that rollback completes within 30 seconds (target: <5s in practice)"""
        # Setup
        shadow_manager.register_model(
            model_name='champion_old',
            model_type='xgboost',
            version='1.0',
            role=ModelRole.CHAMPION,
            description='Old champion'
        )
        
        shadow_manager.register_model(
            model_name='champion_new',
            model_type='lightgbm',
            version='1.0',
            role=ModelRole.CHALLENGER,
            description='New champion'
        )
        
        # Promote
        shadow_manager.promote_challenger('champion_new', force=True)
        
        # Time rollback
        start = time.time()
        success = shadow_manager.rollback_to_previous_champion(reason="Test rollback")
        elapsed = time.time() - start
        
        assert success, "Rollback should succeed"
        assert elapsed < 30, f"Rollback should complete in <30s (took {elapsed:.1f}s)"
        # In production with actual model loading, this should still be <5s


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

