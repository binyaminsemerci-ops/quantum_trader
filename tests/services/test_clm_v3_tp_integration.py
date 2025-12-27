"""
Tests for CLM v3 TP feedback integration.

Validates that CLM orchestrator correctly:
1. Fetches TP performance metrics
2. Computes appropriate tp_reward_weight
3. Enriches RL training jobs with TP feedback
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from backend.services.clm_v3.models import (
    ModelType,
    TrainingJob,
    TriggerReason,
)
from backend.services.clm_v3.orchestrator import ClmOrchestrator
from backend.services.monitoring.tp_performance_tracker import TPMetrics


@pytest.fixture
def mock_tp_tracker():
    """Mock TPPerformanceTracker."""
    tracker = Mock()
    return tracker


@pytest.fixture
def mock_registry():
    """Mock ModelRegistryV3."""
    registry = Mock()
    registry.update_training_job = Mock()
    return registry


@pytest.fixture
def mock_training_adapter():
    """Mock ModelTrainingAdapter."""
    adapter = AsyncMock()
    return adapter


@pytest.fixture
def mock_backtest_adapter():
    """Mock BacktestAdapter."""
    adapter = AsyncMock()
    return adapter


@pytest.fixture
def orchestrator(mock_registry, mock_training_adapter, mock_backtest_adapter):
    """Create ClmOrchestrator with mocked dependencies."""
    config = {
        'enable_tp_optimizer_logging': False,  # Disable for most tests
        'promotion_criteria': {
            'min_sharpe_ratio': 1.0,
            'min_win_rate': 0.52,
        }
    }
    return ClmOrchestrator(
        registry=mock_registry,
        training_adapter=mock_training_adapter,
        backtest_adapter=mock_backtest_adapter,
        config=config
    )


class TestTPFeedbackComputation:
    """Test TP feedback weight computation logic."""
    
    @pytest.mark.asyncio
    async def test_low_hit_rate_high_r_increases_weight(self, orchestrator, mock_tp_tracker):
        """Low hit rate + good R → increase tp_reward_weight to 2.0."""
        # Create RL training job
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="BTCUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback: 30% hit rate → ~3.33R avg
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.30,
            'avg_r_multiple': 3.33,
            'total_attempts': 20,
            'total_hits': 6,
            'total_misses': 14,
            'premature_exit_rate': 0.10
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify tp_reward_weight increased
        assert job.training_params['tp_reward_weight'] == 2.0
        assert job.training_params['tp_hit_rate'] == 0.30
        assert job.training_params['avg_tp_r_multiple'] == 3.33
        assert 'Low hit rate' in job.training_params['tp_feedback_reason']
    
    @pytest.mark.asyncio
    async def test_high_hit_rate_low_r_decreases_weight(self, orchestrator, mock_tp_tracker):
        """High hit rate + low R → decrease tp_reward_weight to 0.5."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="ETHUSDT",
            trigger_reason=TriggerReason.DRIFT_DETECTED,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback: 85% hit rate → ~1.18R avg (low)
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.85,
            'avg_r_multiple': 1.18,
            'total_attempts': 30,
            'total_hits': 25,
            'total_misses': 5,
            'premature_exit_rate': 0.05
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify tp_reward_weight decreased
        assert job.training_params['tp_reward_weight'] == 0.5
        assert job.training_params['tp_hit_rate'] == 0.85
        assert 'High hit rate' in job.training_params['tp_feedback_reason']
    
    @pytest.mark.asyncio
    async def test_optimal_metrics_default_weight(self, orchestrator, mock_tp_tracker):
        """Optimal hit rate + good R → default tp_reward_weight = 1.0."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="ADAUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback: 55% hit rate → ~1.82R avg (optimal)
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.55,
            'avg_r_multiple': 1.82,
            'total_attempts': 50,
            'total_hits': 27,
            'total_misses': 23,
            'premature_exit_rate': 0.08
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify default weight
        assert job.training_params['tp_reward_weight'] == 1.0
        assert 'optimal' in job.training_params['tp_feedback_reason'].lower()
    
    @pytest.mark.asyncio
    async def test_acceptable_hit_rate_low_r_moderate_increase(self, orchestrator, mock_tp_tracker):
        """Hit rate ok but R low → moderate increase to 1.2."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="SOLUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback: 60% hit rate → ~1.67R but let's say actual is 1.1R
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.60,
            'avg_r_multiple': 1.1,  # Below min 1.2
            'total_attempts': 40,
            'total_hits': 24,
            'total_misses': 16,
            'premature_exit_rate': 0.12
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify moderate increase
        assert job.training_params['tp_reward_weight'] == 1.2
        assert 'Hit rate ok' in job.training_params['tp_feedback_reason']
    
    @pytest.mark.asyncio
    async def test_insufficient_data_uses_default(self, orchestrator, mock_tp_tracker):
        """Insufficient TP data → use default weight 1.0."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="DOTUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock: No TP feedback (insufficient data)
        mock_tp_tracker.get_strategy_tp_feedback.return_value = None
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify default weight used
        assert job.training_params['tp_reward_weight'] == 1.0
        assert 'tp_hit_rate' not in job.training_params
    
    @pytest.mark.asyncio
    async def test_non_rl_model_skips_enrichment(self, orchestrator, mock_tp_tracker):
        """Non-RL models (XGBoost, LGBM) skip TP feedback enrichment."""
        job = TrainingJob(
            model_type=ModelType.XGBOOST,
            symbol="BTCUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={}
        )
        
        await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify no TP feedback added
        assert 'tp_reward_weight' not in job.training_params
        assert not mock_tp_tracker.get_strategy_tp_feedback.called


class TestTPFeedbackIntegration:
    """Test integration of TP feedback into training pipeline."""
    
    @pytest.mark.asyncio
    async def test_rl_training_job_receives_tp_feedback(self, orchestrator, mock_tp_tracker):
        """Test that RL training job gets enriched with TP feedback."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="BTCUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.35,
            'avg_r_multiple': 2.85,
            'total_attempts': 25,
            'total_hits': 9,
            'total_misses': 16,
            'premature_exit_rate': 0.15
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify job contains TP feedback
        assert 'tp_reward_weight' in job.training_params
        assert 'tp_hit_rate' in job.training_params
        assert 'avg_tp_r_multiple' in job.training_params
        assert 'tp_feedback_reason' in job.training_params
        
        # Verify values
        assert job.training_params['tp_reward_weight'] == 2.0  # Low hit rate scenario
        assert job.training_params['tp_hit_rate'] == 0.35
        assert job.training_params['avg_tp_r_multiple'] == 2.85
    
    @pytest.mark.asyncio
    async def test_multi_symbol_training_aggregates_tp_metrics(self, orchestrator, mock_tp_tracker):
        """Multi-symbol training aggregates TP metrics across all symbols."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol=None,  # Multi-symbol
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock aggregated TP feedback
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.52,
            'avg_r_multiple': 1.92,
            'total_attempts': 150,
            'total_hits': 78,
            'total_misses': 72,
            'premature_exit_rate': 0.10
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify get_strategy_tp_feedback called with symbol=None
        mock_tp_tracker.get_strategy_tp_feedback.assert_called_once_with(
            strategy_id='RL_V3',
            symbol=None,
            min_attempts=10
        )
        
        # Verify optimal weight (hit rate and R both good)
        assert job.training_params['tp_reward_weight'] == 1.0


class TestTPOptimizerLogging:
    """Test optional TPOptimizer recommendations logging."""
    
    @pytest.mark.asyncio
    async def test_tp_optimizer_recommendations_logged(self, orchestrator, mock_tp_tracker):
        """Test TPOptimizer recommendations are logged when enabled."""
        # Enable optimizer logging
        orchestrator.config['enable_tp_optimizer_logging'] = True
        
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="BTCUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.30,
            'avg_r_multiple': 3.33,
            'total_attempts': 20
        }
        
        # Mock TPOptimizer
        mock_optimizer = Mock()
        mock_rec = Mock()
        mock_rec.symbol = "BTCUSDT"
        mock_rec.direction.value = "CLOSER"
        mock_rec.suggested_scale_factor = 0.95
        mock_rec.confidence = 0.75
        mock_rec.reason = "Low hit rate - bring TPs closer"
        mock_optimizer.evaluate_profile.return_value = mock_rec
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker), \
             patch('backend.services.monitoring.tp_optimizer_v3.get_tp_optimizer', return_value=mock_optimizer):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify recommendations logged
        assert 'tp_optimizer_recommendations' in job.training_params
        recommendations = job.training_params['tp_optimizer_recommendations']
        assert len(recommendations) == 1
        assert recommendations[0]['symbol'] == "BTCUSDT"
        assert recommendations[0]['direction'] == "CLOSER"
        assert recommendations[0]['scale_factor'] == 0.95
    
    @pytest.mark.asyncio
    async def test_tp_optimizer_disabled_by_default(self, orchestrator, mock_tp_tracker):
        """Test TPOptimizer logging is disabled by default."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="ETHUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock TP feedback
        mock_tp_tracker.get_strategy_tp_feedback.return_value = {
            'tp_hit_rate': 0.55,
            'avg_r_multiple': 1.82,
            'total_attempts': 30
        }
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify no optimizer recommendations
        assert 'tp_optimizer_recommendations' not in job.training_params


class TestErrorHandling:
    """Test error handling in TP feedback integration."""
    
    @pytest.mark.asyncio
    async def test_tp_tracker_error_falls_back_to_default(self, orchestrator, mock_tp_tracker):
        """TP tracker error → fallback to default weight."""
        job = TrainingJob(
            model_type=ModelType.RL_V3,
            symbol="BTCUSDT",
            trigger_reason=TriggerReason.PERIODIC,
            training_params={'strategy_id': 'RL_V3'}
        )
        
        # Mock error in TP tracker
        mock_tp_tracker.get_strategy_tp_feedback.side_effect = Exception("Database error")
        
        with patch('backend.services.monitoring.tp_performance_tracker.get_tp_tracker', return_value=mock_tp_tracker):
            await orchestrator._enrich_rl_training_with_tp_feedback(job)
        
        # Verify fallback to default
        assert job.training_params['tp_reward_weight'] == 1.0
        assert 'tp_hit_rate' not in job.training_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


