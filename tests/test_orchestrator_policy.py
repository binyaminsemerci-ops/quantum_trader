"""
Tests for Orchestrator Policy Engine

Verifies that the Orchestrator correctly unifies subsystem outputs
into authoritative trading policies.
"""

import pytest
from datetime import datetime, timezone, timedelta
from backend.services.governance.orchestrator_policy import (
    OrchestratorPolicy,
    OrchestratorConfig,
    RiskState,
    SymbolPerformanceData,
    CostMetrics,
    TradingPolicy,
    create_risk_state,
    create_symbol_performance,
    create_cost_metrics
)


@pytest.fixture
def config():
    """Standard test configuration."""
    return OrchestratorConfig(
        base_confidence=0.50,
        base_risk_pct=1.0,
        daily_dd_limit=3.0,
        losing_streak_limit=5,
        max_open_positions=8,
        total_exposure_limit=15.0,
        policy_update_interval_sec=1  # Fast updates for testing
    )


@pytest.fixture
def orchestrator(config):
    """Create orchestrator instance."""
    return OrchestratorPolicy(config=config)


@pytest.fixture
def normal_risk_state():
    """Normal risk state."""
    return create_risk_state(
        daily_pnl_pct=0.5,
        current_drawdown_pct=-0.8,
        losing_streak=1,
        open_trades_count=3,
        total_exposure_pct=6.0
    )


@pytest.fixture
def good_symbol_performance():
    """Good symbol performance."""
    return [
        create_symbol_performance("BTCUSDT", 0.60, 1.5, 1000.0, "GOOD"),
        create_symbol_performance("ETHUSDT", 0.55, 1.2, 500.0, "NEUTRAL"),
        create_symbol_performance("SOLUSDT", 0.25, -0.5, -300.0, "BAD")
    ]


@pytest.fixture
def normal_costs():
    """Normal cost metrics."""
    return create_cost_metrics("NORMAL", "NORMAL")


class TestOrchestratorConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.base_confidence == 0.50
        assert config.base_risk_pct == 1.0
        assert config.daily_dd_limit == 3.0
        assert config.losing_streak_limit == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            base_confidence=0.55,
            daily_dd_limit=5.0
        )
        assert config.base_confidence == 0.55
        assert config.daily_dd_limit == 5.0


class TestDataClasses:
    """Test data class creation."""
    
    def test_risk_state_creation(self):
        """Test RiskState creation."""
        rs = create_risk_state(1.5, -2.0, 3, 5, 10.0)
        assert rs.daily_pnl_pct == 1.5
        assert rs.current_drawdown_pct == -2.0
        assert rs.losing_streak == 3
    
    def test_symbol_performance_creation(self):
        """Test SymbolPerformanceData creation."""
        sp = create_symbol_performance("BTCUSDT", 0.60, 1.5, 1000.0, "GOOD")
        assert sp.symbol == "BTCUSDT"
        assert sp.winrate == 0.60
        assert sp.performance_tag == "GOOD"
    
    def test_cost_metrics_creation(self):
        """Test CostMetrics creation."""
        cm = create_cost_metrics("HIGH", "NORMAL", 0.001)
        assert cm.spread_level == "HIGH"
        assert cm.slippage_level == "NORMAL"
        assert cm.funding_cost_estimate == 0.001


class TestPolicySimilarity:
    """Test policy similarity scoring."""
    
    def test_identical_policies(self):
        """Identical policies should have similarity=1.0."""
        p1 = TradingPolicy(
            allow_new_trades=True,
            risk_profile="NORMAL",
            max_risk_pct=1.0,
            min_confidence=0.50,
            entry_mode="NORMAL",
            exit_mode="TREND_FOLLOW",
            allowed_symbols=["BTCUSDT"],
            disallowed_symbols=[],
            note="Test",
            timestamp="2025-01-01T00:00:00Z"
        )
        p2 = TradingPolicy(
            allow_new_trades=True,
            risk_profile="NORMAL",
            max_risk_pct=1.0,
            min_confidence=0.50,
            entry_mode="NORMAL",
            exit_mode="TREND_FOLLOW",
            allowed_symbols=["BTCUSDT"],
            disallowed_symbols=[],
            note="Test2",
            timestamp="2025-01-01T00:01:00Z"
        )
        
        similarity = p1.similarity_score(p2)
        assert similarity == 1.0
    
    def test_completely_different_policies(self):
        """Completely different policies should have low similarity."""
        p1 = TradingPolicy(
            allow_new_trades=True,
            risk_profile="NORMAL",
            max_risk_pct=1.0,
            min_confidence=0.50,
            entry_mode="NORMAL",
            exit_mode="TREND_FOLLOW",
            allowed_symbols=["BTCUSDT"],
            disallowed_symbols=[],
            note="Test",
            timestamp="2025-01-01T00:00:00Z"
        )
        p2 = TradingPolicy(
            allow_new_trades=False,
            risk_profile="NO_NEW_TRADES",
            max_risk_pct=0.3,
            min_confidence=0.65,
            entry_mode="DEFENSIVE",
            exit_mode="FAST_TP",
            allowed_symbols=["ETHUSDT"],
            disallowed_symbols=["BTCUSDT"],
            note="Different",
            timestamp="2025-01-01T00:01:00Z"
        )
        
        similarity = p1.similarity_score(p2)
        assert similarity < 0.3
    
    def test_minor_numeric_differences(self):
        """Policies with minor numeric differences should be similar."""
        p1 = TradingPolicy(
            allow_new_trades=True,
            risk_profile="NORMAL",
            max_risk_pct=1.0,
            min_confidence=0.50,
            entry_mode="NORMAL",
            exit_mode="TREND_FOLLOW",
            allowed_symbols=[],
            disallowed_symbols=[],
            note="Test",
            timestamp="2025-01-01T00:00:00Z"
        )
        p2 = TradingPolicy(
            allow_new_trades=True,
            risk_profile="NORMAL",
            max_risk_pct=1.05,  # 5% difference
            min_confidence=0.51,  # 2% difference
            entry_mode="NORMAL",
            exit_mode="TREND_FOLLOW",
            allowed_symbols=[],
            disallowed_symbols=[],
            note="Test2",
            timestamp="2025-01-01T00:01:00Z"
        )
        
        similarity = p1.similarity_score(p2)
        assert similarity >= 0.90


class TestExtremeVolatility:
    """Test EXTREME_VOL scenario."""
    
    def test_extreme_vol_blocks_trades(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """EXTREME volatility should block all new trades."""
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="EXTREME",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.allow_new_trades is False
        assert policy.risk_profile == "NO_NEW_TRADES"
        assert "EXTREME" in policy.note


class TestDailyDrawdownProtection:
    """Test daily drawdown limit."""
    
    def test_dd_limit_hit(self, orchestrator, good_symbol_performance, normal_costs):
        """Daily DD limit should block trades."""
        risk_state = create_risk_state(
            daily_pnl_pct=-4.0,
            current_drawdown_pct=-3.5,  # Exceeds -3.0 limit
            losing_streak=2,
            open_trades_count=2,
            total_exposure_pct=5.0
        )
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.allow_new_trades is False
        assert policy.risk_profile == "NO_NEW_TRADES"
        assert "DD limit" in policy.note


class TestLosingStreakProtection:
    """Test losing streak protection."""
    
    def test_losing_streak_reduces_risk(self, orchestrator, good_symbol_performance, normal_costs):
        """Losing streak should reduce risk to 30%."""
        risk_state = create_risk_state(
            daily_pnl_pct=-1.5,
            current_drawdown_pct=-1.8,
            losing_streak=6,  # Exceeds limit of 5
            open_trades_count=2,
            total_exposure_pct=4.0
        )
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        # Should multiply base 1.0 by 0.3 = 0.3
        assert policy.max_risk_pct <= 0.4  # Allow some tolerance
        assert policy.entry_mode == "DEFENSIVE"
        assert policy.min_confidence > 0.50
        assert "streak" in policy.note.lower()


class TestRangingMarket:
    """Test RANGING market behavior."""
    
    def test_ranging_defensive_scalping(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """RANGING market should trigger defensive scalping."""
        policy = orchestrator.update_policy(
            regime_tag="RANGING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.entry_mode == "DEFENSIVE"
        assert policy.exit_mode == "FAST_TP"
        # Base 1.0 * 0.7 = 0.7
        assert 0.6 <= policy.max_risk_pct <= 0.8
        # Base 0.50 + 0.05 = 0.55
        assert policy.min_confidence >= 0.54
        assert "RANGING" in policy.note


class TestTrendingMarket:
    """Test TRENDING market behavior."""
    
    def test_trending_aggressive(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """TRENDING + NORMAL_VOL should be aggressive."""
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.entry_mode == "AGGRESSIVE"
        assert policy.exit_mode == "TREND_FOLLOW"
        # Base 0.50 - 0.03 = 0.47
        assert policy.min_confidence <= 0.48
        assert "TRENDING" in policy.note


class TestSymbolPerformanceFiltering:
    """Test symbol filtering based on performance."""
    
    def test_bad_symbol_excluded(self, orchestrator, normal_risk_state, normal_costs):
        """BAD performers should be disallowed."""
        symbol_perf = [
            create_symbol_performance("BTCUSDT", 0.60, 1.5, 1000.0, "GOOD"),
            create_symbol_performance("ETHUSDT", 0.55, 1.2, 500.0, "NEUTRAL"),
            create_symbol_performance("SOLUSDT", 0.25, -0.5, -300.0, "BAD"),
            create_symbol_performance("APTUSDT", 0.20, -1.0, -500.0, "BAD")
        ]
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=symbol_perf,
            cost_metrics=normal_costs
        )
        
        assert "SOLUSDT" in policy.disallowed_symbols
        assert "APTUSDT" in policy.disallowed_symbols
        assert "BTCUSDT" in policy.allowed_symbols
        assert "ETHUSDT" in policy.allowed_symbols
        assert len(policy.disallowed_symbols) == 2


class TestHighCosts:
    """Test high cost scenario."""
    
    def test_high_spread_stricter_confidence(self, orchestrator, normal_risk_state, good_symbol_performance):
        """High spread should increase confidence threshold."""
        high_costs = create_cost_metrics("HIGH", "HIGH")
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=high_costs
        )
        
        assert policy.entry_mode == "DEFENSIVE"
        # Base 0.50 - 0.03 (trending) + 0.03 (high costs) = 0.50
        assert policy.min_confidence >= 0.48
        assert "cost" in policy.note.lower()


class TestHighVolatility:
    """Test HIGH volatility scenario."""
    
    def test_high_vol_reduced_risk(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """HIGH volatility should reduce risk by 50%."""
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="HIGH",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.risk_profile == "REDUCED"
        # Base 1.0 * 0.5 = 0.5
        assert 0.45 <= policy.max_risk_pct <= 0.55
        assert policy.min_confidence > 0.50


class TestPositionLimits:
    """Test position count and exposure limits."""
    
    def test_max_positions_reached(self, orchestrator, good_symbol_performance, normal_costs):
        """Max positions should block new trades."""
        risk_state = create_risk_state(
            daily_pnl_pct=1.0,
            current_drawdown_pct=-0.5,
            losing_streak=0,
            open_trades_count=8,  # At limit
            total_exposure_pct=12.0
        )
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.allow_new_trades is False
        assert "positions" in policy.note.lower()
    
    def test_max_exposure_reached(self, orchestrator, good_symbol_performance, normal_costs):
        """Max exposure should block new trades."""
        risk_state = create_risk_state(
            daily_pnl_pct=2.0,
            current_drawdown_pct=-0.3,
            losing_streak=0,
            open_trades_count=5,
            total_exposure_pct=16.0  # Exceeds 15.0 limit
        )
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.allow_new_trades is False
        assert "exposure" in policy.note.lower()


class TestPolicyStability:
    """Test policy stability mechanism."""
    
    def test_no_oscillation_on_similar_inputs(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """Similar inputs should not trigger policy update."""
        # First update
        policy1 = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        # Immediate second update with same inputs
        policy2 = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        # Should return same policy (not updated)
        assert policy1.timestamp == policy2.timestamp
    
    def test_update_after_interval(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """Policy should update after interval elapsed."""
        import time
        
        # First update
        policy1 = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        # Wait for interval
        time.sleep(1.1)  # config.policy_update_interval_sec = 1
        
        # Second update with different inputs
        policy2 = orchestrator.update_policy(
            regime_tag="RANGING",
            vol_level="HIGH",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        # Should get new policy
        assert policy1.timestamp != policy2.timestamp
        assert policy1.entry_mode != policy2.entry_mode


class TestLowEnsembleQuality:
    """Test ensemble quality adjustment."""
    
    def test_low_ensemble_quality_raises_confidence(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """Low ensemble quality should raise confidence requirement."""
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            ensemble_quality=0.35,  # Low quality
            cost_metrics=normal_costs
        )
        
        # Base 0.50 - 0.03 (trending) + 0.05 (low ensemble) = 0.52
        assert policy.min_confidence >= 0.51
        assert "ensemble" in policy.note.lower()


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_policy(self, orchestrator):
        """Test get_policy returns current policy."""
        policy = orchestrator.get_policy()
        assert policy is not None
        assert isinstance(policy, TradingPolicy)
    
    def test_reset_daily(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """Test daily reset."""
        # Update policy
        policy1 = orchestrator.update_policy(
            regime_tag="RANGING",
            vol_level="HIGH",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        # Reset
        orchestrator.reset_daily()
        
        # Should be back to default
        policy2 = orchestrator.get_policy()
        assert policy2.risk_profile == "NORMAL"
        assert policy2.entry_mode == "NORMAL"
    
    def test_policy_history(self, orchestrator, normal_risk_state, good_symbol_performance, normal_costs):
        """Test policy history tracking."""
        import time
        
        # Create multiple policies with different inputs to trigger updates
        vol_levels = ["NORMAL", "HIGH", "LOW"]
        for i, vol in enumerate(vol_levels):
            orchestrator.update_policy(
                regime_tag="TRENDING",
                vol_level=vol,
                risk_state=normal_risk_state,
                symbol_performance=good_symbol_performance,
                cost_metrics=normal_costs
            )
            time.sleep(1.1)
        
        history = orchestrator.get_policy_history(limit=5)
        assert len(history) >= 3


class TestComplexScenarios:
    """Test complex multi-factor scenarios."""
    
    def test_multiple_constraints_compound(self, orchestrator, good_symbol_performance):
        """Multiple constraints should compound properly."""
        # HIGH vol + RANGING + losing streak
        risk_state = create_risk_state(
            daily_pnl_pct=-2.0,
            current_drawdown_pct=-2.5,
            losing_streak=6,
            open_trades_count=4,
            total_exposure_pct=8.0
        )
        
        high_costs = create_cost_metrics("HIGH", "HIGH")
        
        policy = orchestrator.update_policy(
            regime_tag="RANGING",
            vol_level="HIGH",
            risk_state=risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=high_costs
        )
        
        # Risk should be heavily reduced
        # Base 1.0 * 0.5 (high vol) * 0.7 (ranging) * 0.3 (streak) = 0.105
        assert policy.max_risk_pct < 0.2
        assert policy.entry_mode == "DEFENSIVE"
        assert policy.min_confidence > 0.55
        assert policy.risk_profile == "REDUCED"


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_symbol_list(self, orchestrator, normal_risk_state, normal_costs):
        """Empty symbol list should not crash."""
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=[],
            cost_metrics=normal_costs
        )
        
        assert len(policy.allowed_symbols) == 0
        assert len(policy.disallowed_symbols) == 0
    
    def test_no_cost_metrics(self, orchestrator, normal_risk_state, good_symbol_performance):
        """None cost metrics should not crash."""
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=normal_risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=None
        )
        
        assert policy is not None
    
    def test_zero_risk_state(self, orchestrator, good_symbol_performance, normal_costs):
        """All-zero risk state should work."""
        risk_state = create_risk_state(0, 0, 0, 0, 0)
        
        policy = orchestrator.update_policy(
            regime_tag="TRENDING",
            vol_level="NORMAL",
            risk_state=risk_state,
            symbol_performance=good_symbol_performance,
            cost_metrics=normal_costs
        )
        
        assert policy.allow_new_trades is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
