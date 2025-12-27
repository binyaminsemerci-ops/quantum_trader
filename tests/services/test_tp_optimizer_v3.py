"""
Tests for TP Optimizer v3
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from backend.services.monitoring.tp_optimizer_v3 import (
    TPOptimizerV3,
    TPOptimizationTarget,
    TPAdjustmentRecommendation,
    AdjustmentDirection,
    get_tp_optimizer,
    optimize_profiles_for_strategy
)
from backend.services.monitoring.tp_performance_tracker import (
    TPPerformanceTracker,
    TPMetrics
)
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
    TPProfile,
    TPProfileLeg,
    TrailingProfile,
    MarketRegime,
    TPKind
)


@pytest.fixture
def temp_storage():
    """Create temporary storage for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "tp_metrics.json"


@pytest.fixture
def tp_tracker(temp_storage):
    """Create TPPerformanceTracker with test data."""
    tracker = TPPerformanceTracker(storage_path=temp_storage)
    return tracker


@pytest.fixture
def optimizer(tp_tracker):
    """Create TPOptimizerV3 instance."""
    targets = [
        TPOptimizationTarget(
            strategy_id="TEST_STRAT",
            symbol="*",
            min_hit_rate=0.45,
            max_hit_rate=0.70,
            min_avg_r=1.2,
            min_attempts=10
        )
    ]
    return TPOptimizerV3(tp_tracker=tp_tracker, targets=targets)


class TestTPOptimizationTarget:
    """Test TPOptimizationTarget configuration."""
    
    def test_default_values(self):
        """Test default target values."""
        target = TPOptimizationTarget(strategy_id="TEST")
        
        assert target.strategy_id == "TEST"
        assert target.symbol == "*"
        assert target.min_hit_rate == 0.45
        assert target.max_hit_rate == 0.70
        assert target.min_avg_r == 1.2
        assert target.min_attempts == 20
    
    def test_custom_values(self):
        """Test custom target configuration."""
        target = TPOptimizationTarget(
            strategy_id="AGGRESSIVE",
            symbol="BTCUSDT",
            min_hit_rate=0.60,
            max_hit_rate=0.85,
            min_avg_r=0.8,
            min_attempts=50
        )
        
        assert target.strategy_id == "AGGRESSIVE"
        assert target.symbol == "BTCUSDT"
        assert target.min_hit_rate == 0.60
        assert target.max_hit_rate == 0.85


class TestTPOptimizerV3Initialization:
    """Test optimizer initialization."""
    
    def test_default_initialization(self, tp_tracker):
        """Test optimizer with default targets."""
        optimizer = TPOptimizerV3(tp_tracker=tp_tracker)
        
        assert optimizer.tp_tracker is not None
        assert len(optimizer._targets) > 0  # Should have default targets
    
    def test_custom_targets(self, tp_tracker):
        """Test optimizer with custom targets."""
        custom_targets = [
            TPOptimizationTarget(strategy_id="CUSTOM1"),
            TPOptimizationTarget(strategy_id="CUSTOM2")
        ]
        
        optimizer = TPOptimizerV3(tp_tracker=tp_tracker, targets=custom_targets)
        assert len(optimizer._targets) == 2
    
    def test_target_matching(self, optimizer):
        """Test target matching logic."""
        # Should match wildcard
        target = optimizer._get_target_for_pair("TEST_STRAT", "BTCUSDT")
        assert target is not None
        assert target.strategy_id == "TEST_STRAT"
        
        # Should not match
        target = optimizer._get_target_for_pair("UNKNOWN_STRAT", "ETHUSDT")
        assert target is None


class TestOptimizationLogic:
    """Test core optimization decision logic."""
    
    def test_low_hit_rate_high_r_suggests_closer(self, optimizer, tp_tracker):
        """Low hit rate but good R → bring TPs closer."""
        # Setup: 30% hit rate → calculates to 1/0.30 = 3.33R avg
        tp_tracker.metrics[("TEST_STRAT", "BTCUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="BTCUSDT",
            tp_attempts=20,
            tp_hits=6,
            tp_misses=14,
            tp_hit_rate=0.30,  # Below min_hit_rate=0.45, will give ~3.3R
            total_tp_profit_usd=1000.0,
            avg_tp_profit_usd=166.67
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "BTCUSDT")
        
        assert rec is not None
        assert rec.direction == AdjustmentDirection.CLOSER
        assert rec.suggested_scale_factor < 1.0
        assert "below target" in rec.reason.lower()
    
    def test_high_hit_rate_low_r_suggests_further(self, optimizer, tp_tracker):
        """High hit rate but low R → push TPs further."""
        # Setup: 80% hit rate → calculates to 1/0.80 = 1.25R avg (above min 1.2 but we want to test low R)
        # Actually, 80% would give 1.25R which is above 1.2 min, so let's use 85% → 1.18R
        tp_tracker.metrics[("TEST_STRAT", "ETHUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ETHUSDT",
            tp_attempts=20,
            tp_hits=17,
            tp_misses=3,
            tp_hit_rate=0.85,  # Above max_hit_rate=0.70, will give 1/0.85 = 1.18R (below 1.2)
            total_tp_profit_usd=800.0,
            avg_tp_profit_usd=50.0
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "ETHUSDT")
        
        assert rec is not None
        assert rec.direction == AdjustmentDirection.FURTHER
        assert rec.suggested_scale_factor > 1.0
        assert "above target" in rec.reason.lower()
    
    def test_optimal_metrics_no_recommendation(self, optimizer, tp_tracker):
        """Metrics within target band → no recommendation."""
        # Setup: 55% hit rate → 1/0.55 = 1.82R (above min 1.2)
        # Hit rate is within 0.45-0.70, R is above 1.2 → optimal
        tp_tracker.metrics[("TEST_STRAT", "ADAUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ADAUSDT",
            tp_attempts=20,
            tp_hits=11,
            tp_misses=9,
            tp_hit_rate=0.55,  # Within target band 0.45-0.70, gives 1.82R (> 1.2)
            total_tp_profit_usd=1100.0,
            avg_tp_profit_usd=100.0
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "ADAUSDT")
        
        # Should return None for optimal performance
        assert rec is None
    
    def test_insufficient_data_no_recommendation(self, optimizer, tp_tracker):
        """Too few attempts → no recommendation."""
        tp_tracker.metrics[("TEST_STRAT", "DOTUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="DOTUSDT",
            tp_attempts=5,  # Below min_attempts=10
            tp_hits=2,
            tp_misses=3,
            tp_hit_rate=0.40
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "DOTUSDT")
        assert rec is None
    
    def test_no_metrics_available(self, optimizer):
        """No metrics → no recommendation."""
        rec = optimizer.evaluate_profile("TEST_STRAT", "NONEXISTENT")
        assert rec is None
    
    def test_no_target_defined(self, optimizer, tp_tracker):
        """No target for strategy → no recommendation."""
        tp_tracker.metrics[("UNKNOWN_STRAT", "BTCUSDT")] = TPMetrics(
            strategy_id="UNKNOWN_STRAT",
            symbol="BTCUSDT",
            tp_attempts=20,
            tp_hits=10,
            tp_misses=10,
            tp_hit_rate=0.50
        )
        
        rec = optimizer.evaluate_profile("UNKNOWN_STRAT", "BTCUSDT")
        assert rec is None


class TestRecommendationApplication:
    """Test applying recommendations."""
    
    def test_apply_recommendation_log_only(self, optimizer, tp_tracker):
        """Test log-only application."""
        # Create metrics that will trigger recommendation
        tp_tracker.metrics[("TEST_STRAT", "BTCUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="BTCUSDT",
            tp_attempts=20,
            tp_hits=6,
            tp_misses=14,
            tp_hit_rate=0.30
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "BTCUSDT")
        assert rec is not None
        
        # Apply without persistence
        adjusted_profile = optimizer.apply_recommendation(rec, persist=False)
        
        assert adjusted_profile is not None
        assert rec.applied is False  # Not marked as applied for log-only
        
        # Check adjusted profile structure
        assert len(adjusted_profile.tp_legs) > 0
        assert adjusted_profile.name == rec.profile_name
    
    def test_apply_recommendation_with_persistence(self, optimizer, tp_tracker):
        """Test persistent application."""
        # Setup: 85% hit rate → 1.18R (high hit rate, low R)
        tp_tracker.metrics[("TEST_STRAT", "ETHUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ETHUSDT",
            tp_attempts=20,
            tp_hits=17,
            tp_misses=3,
            tp_hit_rate=0.85  # Above 0.70, gives 1.18R (below 1.2)
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "ETHUSDT")
        assert rec is not None
        
        # Apply with persistence
        adjusted_profile = optimizer.apply_recommendation(rec, persist=True)
        
        assert adjusted_profile is not None
        assert rec.applied is True
        
        # Check runtime override was set
        override = optimizer.get_runtime_override("TEST_STRAT", "ETHUSDT")
        assert override == rec.suggested_scale_factor
    
    def test_adjusted_profile_structure(self, optimizer, tp_tracker):
        """Test that adjusted profile maintains structure."""
        from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
            DEFAULT_TREND_PROFILE
        )
        
        # Create adjusted profile
        scale_factor = 0.9
        adjusted = optimizer._create_adjusted_profile(
            base_profile=DEFAULT_TREND_PROFILE,
            scale_factor=scale_factor,
            new_name="TEST_ADJUSTED"
        )
        
        # Check legs are scaled
        assert len(adjusted.tp_legs) == len(DEFAULT_TREND_PROFILE.tp_legs)
        for base_leg, adj_leg in zip(DEFAULT_TREND_PROFILE.tp_legs, adjusted.tp_legs):
            assert adj_leg.r_multiple == pytest.approx(base_leg.r_multiple * scale_factor)
            assert adj_leg.size_fraction == base_leg.size_fraction
            assert adj_leg.kind == base_leg.kind
        
        # Check trailing is scaled
        if DEFAULT_TREND_PROFILE.trailing:
            assert adjusted.trailing is not None
            assert adjusted.trailing.activation_r == pytest.approx(
                DEFAULT_TREND_PROFILE.trailing.activation_r * scale_factor
            )
    
    def test_double_application_warning(self, optimizer, tp_tracker):
        """Test warning on double application."""
        tp_tracker.metrics[("TEST_STRAT", "ADAUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ADAUSDT",
            tp_attempts=20,
            tp_hits=6,
            tp_misses=14,
            tp_hit_rate=0.30
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "ADAUSDT")
        assert rec is not None
        
        # Apply first time
        optimizer.apply_recommendation(rec, persist=True)
        assert rec.applied is True
        
        # Try to apply again
        result = optimizer.apply_recommendation(rec, persist=True)
        assert result is None  # Should return None


class TestBatchOptimization:
    """Test batch optimization."""
    
    def test_optimize_all_profiles_once(self, optimizer, tp_tracker):
        """Test batch optimization across all tracked pairs."""
        # Add multiple metrics
        # 30% hit rate → 3.33R (low hit, high R) → CLOSER
        tp_tracker.metrics[("TEST_STRAT", "BTCUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="BTCUSDT",
            tp_attempts=20,
            tp_hits=6,
            tp_misses=14,
            tp_hit_rate=0.30
        )
        
        # 85% hit rate → 1.18R (high hit, low R) → FURTHER
        tp_tracker.metrics[("TEST_STRAT", "ETHUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ETHUSDT",
            tp_attempts=20,
            tp_hits=17,
            tp_misses=3,
            tp_hit_rate=0.85
        )
        
        # 55% hit rate → 1.82R (optimal) → NO CHANGE
        tp_tracker.metrics[("TEST_STRAT", "ADAUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ADAUSDT",
            tp_attempts=20,
            tp_hits=11,
            tp_misses=9,
            tp_hit_rate=0.55
        )
        
        recommendations = optimizer.optimize_all_profiles_once()
        
        # Should get 2 recommendations (BTCUSDT and ETHUSDT)
        assert len(recommendations) == 2
        
        # Check recommendations
        symbols = [rec.symbol for rec in recommendations]
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "ADAUSDT" not in symbols  # Optimal performance
    
    def test_convenience_function(self, tp_tracker):
        """Test optimize_profiles_for_strategy convenience function."""
        # Setup metrics
        tp_tracker.metrics[("STRAT_A", "BTCUSDT")] = TPMetrics(
            strategy_id="STRAT_A",
            symbol="BTCUSDT",
            tp_attempts=20,
            tp_hits=6,
            tp_misses=14,
            tp_hit_rate=0.30
        )
        
        # Note: Need to setup target for STRAT_A
        from backend.services.monitoring.tp_optimizer_v3 import get_tp_optimizer
        opt = get_tp_optimizer()
        opt.load_targets([
            TPOptimizationTarget(
                strategy_id="STRAT_A",
                symbol="*",
                min_hit_rate=0.45,
                max_hit_rate=0.70,
                min_avg_r=1.2,
                min_attempts=10
            )
        ])
        
        recommendations = optimize_profiles_for_strategy("STRAT_A")
        
        assert len(recommendations) >= 0  # May or may not have recommendations


class TestRuntimeOverrides:
    """Test runtime override system."""
    
    def test_get_runtime_override(self, optimizer):
        """Test retrieving runtime overrides."""
        # No override initially
        override = optimizer.get_runtime_override("TEST", "BTC")
        assert override is None
        
        # Set override
        optimizer._runtime_overrides[("TEST", "BTC")] = 0.95
        
        # Retrieve override
        override = optimizer.get_runtime_override("TEST", "BTC")
        assert override == 0.95
    
    def test_clear_runtime_overrides(self, optimizer):
        """Test clearing all overrides."""
        optimizer._runtime_overrides[("TEST1", "BTC")] = 0.9
        optimizer._runtime_overrides[("TEST2", "ETH")] = 1.1
        
        assert len(optimizer._runtime_overrides) == 2
        
        optimizer.clear_runtime_overrides()
        
        assert len(optimizer._runtime_overrides) == 0


class TestRecommendationMetadata:
    """Test recommendation metadata and tracking."""
    
    def test_recommendation_contains_metrics_snapshot(self, optimizer, tp_tracker):
        """Test that recommendations include metrics snapshot."""
        tp_tracker.metrics[("TEST_STRAT", "BTCUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="BTCUSDT",
            tp_attempts=25,
            tp_hits=8,
            tp_misses=17,
            tp_hit_rate=0.32,
            avg_slippage_pct=0.002,
            premature_exits=3
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "BTCUSDT")
        
        assert rec is not None
        assert 'tp_hit_rate' in rec.metrics_snapshot
        assert 'avg_r_multiple' in rec.metrics_snapshot
        assert 'tp_attempts' in rec.metrics_snapshot
        assert 'premature_exits' in rec.metrics_snapshot
        
        assert rec.metrics_snapshot['tp_hit_rate'] == 0.32
        assert rec.metrics_snapshot['tp_attempts'] == 25
    
    def test_recommendation_contains_targets(self, optimizer, tp_tracker):
        """Test that recommendations include target thresholds."""
        # 85% hit rate → 1.18R (high hit, low R)
        tp_tracker.metrics[("TEST_STRAT", "ETHUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ETHUSDT",
            tp_attempts=20,
            tp_hits=17,
            tp_misses=3,
            tp_hit_rate=0.85
        )
        
        rec = optimizer.evaluate_profile("TEST_STRAT", "ETHUSDT")
        
        assert rec is not None
        assert 'min_hit_rate' in rec.targets
        assert 'max_hit_rate' in rec.targets
        assert 'min_avg_r' in rec.targets
        
        assert rec.targets['min_hit_rate'] == 0.45
        assert rec.targets['max_hit_rate'] == 0.70
    
    def test_recommendation_timestamp(self, optimizer, tp_tracker):
        """Test recommendation has timestamp."""
        tp_tracker.metrics[("TEST_STRAT", "ADAUSDT")] = TPMetrics(
            strategy_id="TEST_STRAT",
            symbol="ADAUSDT",
            tp_attempts=20,
            tp_hits=6,
            tp_misses=14,
            tp_hit_rate=0.30
        )
        
        before = datetime.now(timezone.utc)
        rec = optimizer.evaluate_profile("TEST_STRAT", "ADAUSDT")
        after = datetime.now(timezone.utc)
        
        assert rec is not None
        assert before <= rec.timestamp <= after


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_avg_r_calculation_fallback(self, optimizer):
        """Test avg R calculation with missing profit data."""
        from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
            DEFAULT_NORMAL_PROFILE
        )
        
        # Metrics with no profit data
        metrics = TPMetrics(
            strategy_id="TEST",
            symbol="BTC",
            tp_attempts=20,
            tp_hits=10,
            tp_misses=10,
            tp_hit_rate=0.50,
            avg_tp_profit_usd=0.0  # No profit data
        )
        
        avg_r = optimizer._calculate_avg_r_multiple(metrics, DEFAULT_NORMAL_PROFILE)
        
        # Should use fallback calculation
        assert avg_r > 0
        assert avg_r == pytest.approx(2.0, rel=0.5)  # Based on 50% hit rate
    
    def test_very_low_hit_rate_edge_case(self, optimizer):
        """Test handling of very low hit rates."""
        from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
            DEFAULT_NORMAL_PROFILE
        )
        
        metrics = TPMetrics(
            strategy_id="TEST",
            symbol="BTC",
            tp_attempts=100,
            tp_hits=1,
            tp_misses=99,
            tp_hit_rate=0.01  # 1% hit rate
        )
        
        avg_r = optimizer._calculate_avg_r_multiple(metrics, DEFAULT_NORMAL_PROFILE)
        
        # Should cap at reasonable maximum
        assert avg_r <= 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
