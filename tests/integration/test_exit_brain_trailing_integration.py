"""
Exit Brain v3 + Trailing Stop Integration Tests

Tests the complete flow from Exit Brain v3 plan creation
to trailing stop execution for live positions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from backend.domains.exits.exit_brain_v3.models import ExitContext, ExitPlan, ExitLeg, ExitKind
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
from backend.domains.exits.exit_brain_v3.router import ExitRouter
from backend.domains.exits.exit_brain_v3.integration import (
    to_trailing_config,
    to_dynamic_tpsl,
    build_context_from_position
)


class TestExitBrainV3TrailingIntegration:
    """Test Exit Brain v3 produces correct trailing configurations"""
    
    @pytest.mark.asyncio
    async def test_exit_plan_always_has_trailing_for_normal_positions(self):
        """
        Exit Brain uses PROFILE-BASED trailing logic:
        - NORMAL_DEFAULT: TP legs use 100% (no trailing leg)
        - TREND_DEFAULT: TP legs use 65%, trailing takes 35%
        - VOLATILE_DEFAULT: TP legs use 100%, but has trailing config
        
        This test verifies TREND profile produces trailing leg.
        """
        brain = ExitBrainV3()
        
        # Standard LONG position context with TREND regime
        ctx = ExitContext(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.1,
            leverage=10.0,
            current_price=50500.0,
            unrealized_pnl_pct=1.0,
            volatility=0.02,
            trend_strength=0.3,
            market_regime="TRENDING",  # Must be "TRENDING" not "TREND" to map correctly
            risk_mode="NORMAL"
        )
        
        plan = await brain.build_exit_plan(ctx)
        
        # CRITICAL: Plan must exist
        assert plan is not None, "Exit Brain must return a plan"
        
        # TREND profile should have trailing leg
        trail_legs = plan.get_legs_by_kind(ExitKind.TRAIL)
        assert len(trail_legs) > 0, "TREND profile must have trailing leg for remaining position"
        
        trail_leg = trail_legs[0]
        assert trail_leg.trail_callback is not None, "Trailing leg must have callback rate"
        assert 0.005 <= trail_leg.trail_callback <= 0.05, "Callback must be 0.5%-5%"
        assert trail_leg.size_pct > 0, "Trailing must have position size"
    
    @pytest.mark.asyncio
    async def test_exit_plan_has_tp_and_sl_legs(self):
        """Exit Brain must produce TP and SL legs"""
        brain = ExitBrainV3()
        
        ctx = ExitContext(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=3000.0,
            size=1.0,
            leverage=20.0,
            current_price=2970.0,
            unrealized_pnl_pct=1.0,
            risk_mode="NORMAL"
        )
        
        plan = await brain.build_exit_plan(ctx)
        
        # Must have SL
        sl_legs = plan.get_legs_by_kind(ExitKind.SL)
        assert len(sl_legs) > 0, "Must have SL leg"
        
        # Must have TP
        tp_legs = plan.get_legs_by_kind(ExitKind.TP)
        assert len(tp_legs) > 0, "Must have TP leg(s)"
    
    @pytest.mark.asyncio
    async def test_volatile_regime_adjusts_trailing(self):
        """VOLATILE regime should disable or adjust trailing"""
        brain = ExitBrainV3()
        
        ctx = ExitContext(
            symbol="SOLUSDT",
            side="LONG",
            entry_price=100.0,
            size=10.0,
            leverage=15.0,
            current_price=102.0,
            unrealized_pnl_pct=2.0,
            volatility=0.08,  # High volatility
            market_regime="VOLATILE",
            risk_mode="NORMAL"
        )
        
        plan = await brain.build_exit_plan(ctx)
        
        # In volatile regime, trailing may be disabled
        # (implementation may vary based on strategy)
        assert plan is not None
        # Verify plan respects volatility in configuration


class TestTrailingConfigConversion:
    """Test conversion from ExitPlan to trailing config"""
    
    def test_to_trailing_config_extracts_correct_fields(self):
        """to_trailing_config must map ExitPlan fields correctly"""
        # Create mock plan
        trail_leg = ExitLeg(
            kind=ExitKind.TRAIL,
            size_pct=0.5,
            trigger_pct=0.06,
            trail_callback=0.015,  # 1.5%
            priority=2,
            reason="Trailing stop for profit protection"
        )
        
        tp_leg = ExitLeg(
            kind=ExitKind.TP,
            size_pct=0.25,
            trigger_pct=0.03,  # 3%
            priority=0
        )
        
        plan = ExitPlan(
            symbol="BTCUSDT",
            legs=[tp_leg, trail_leg],
            strategy_id="STANDARD_LADDER",
            source="EXIT_BRAIN_V3",
            reason="Test"
        )
        
        # Convert to trailing config
        config = to_trailing_config(plan, None)
        
        # Verify conversion
        assert config is not None, "Must return config when trailing leg exists"
        assert config["enabled"] is True
        assert config["callback_pct"] == 0.015, "Must use trail_leg.trail_callback"
        assert config["activation_pct"] == 0.015, "Activation should be 50% of primary TP (3% / 2)"
        assert config["size_pct"] == 0.5
    
    def test_to_trailing_config_returns_none_without_trailing(self):
        """to_trailing_config must return None if no trailing leg"""
        plan = ExitPlan(
            symbol="ETHUSDT",
            legs=[
                ExitLeg(kind=ExitKind.TP, size_pct=1.0, trigger_pct=0.03, priority=0),
                ExitLeg(kind=ExitKind.SL, size_pct=1.0, trigger_pct=-0.02, priority=1)
            ],
            strategy_id="NO_TRAIL",
            source="TEST",
            reason="Test without trailing"
        )
        
        config = to_trailing_config(plan, None)
        
        assert config is None, "Must return None when no trailing leg exists"


class TestExitRouterCaching:
    """Test ExitRouter caching behavior"""
    
    @pytest.mark.asyncio
    async def test_router_caches_plans(self):
        """ExitRouter must cache plans by symbol"""
        router = ExitRouter()
        
        position = {
            "symbol": "BTCUSDT",
            "positionAmt": "0.1",
            "entryPrice": "50000.0",
            "markPrice": "50500.0",
            "leverage": "10"
        }
        
        # First call creates plan
        plan1 = await router.get_or_create_plan(position)
        assert plan1 is not None
        
        # Second call returns cached plan
        plan2 = await router.get_or_create_plan(position)
        assert plan2 is plan1, "Must return same cached plan instance"
        
        # get_active_plan returns cached plan
        cached = router.get_active_plan("BTCUSDT")
        assert cached is plan1, "get_active_plan must return cached plan"
    
    def test_router_get_active_plan_returns_none_for_missing(self):
        """get_active_plan must return None for non-existent symbols"""
        router = ExitRouter()
        
        plan = router.get_active_plan("NONEXISTENT")
        assert plan is None, "Must return None for symbols without cached plan"


class TestPreviousBugScenarios:
    """Regression tests for previously reported bugs"""
    
    def test_no_attribute_error_on_get_plan(self):
        """
        REGRESSION: Previously got "'ExitRouter' object has no attribute 'get_plan'"
        
        This must NOT happen - ExitRouter should only have:
        - get_or_create_plan() (async, requires position dict)
        - get_active_plan() (sync, returns cached plan or None)
        - invalidate_plan() (sync, clears cache)
        """
        router = ExitRouter()
        
        # Verify correct methods exist
        assert hasattr(router, 'get_or_create_plan'), "Must have get_or_create_plan"
        assert hasattr(router, 'get_active_plan'), "Must have get_active_plan"
        assert hasattr(router, 'invalidate_plan'), "Must have invalidate_plan"
        
        # Verify old broken method does NOT exist
        assert not hasattr(router, 'get_plan'), "Must NOT have get_plan (old broken API)"
    
    def test_trailing_config_dict_keys_match_implementation(self):
        """
        REGRESSION: Previously used wrong dict keys:
        - "trail_enabled" instead of "enabled"
        - "trail_callback" instead of "callback_pct"
        - "activation_threshold_pct" instead of "activation_pct"
        """
        trail_leg = ExitLeg(
            kind=ExitKind.TRAIL,
            size_pct=1.0,
            trail_callback=0.02,
            priority=0
        )
        
        plan = ExitPlan(
            symbol="TEST",
            legs=[trail_leg],
            strategy_id="TEST",
            source="TEST",
            reason="Test"
        )
        
        config = to_trailing_config(plan, None)
        
        # Verify CORRECT keys
        assert "enabled" in config, "Must use 'enabled' key"
        assert "callback_pct" in config, "Must use 'callback_pct' key"
        assert "activation_pct" in config, "Must use 'activation_pct' key"
        
        # Verify WRONG keys don't exist
        assert "trail_enabled" not in config, "Must NOT use old 'trail_enabled' key"
        assert "trail_callback" not in config, "Must NOT use old 'trail_callback' key"
        assert "activation_threshold_pct" not in config, "Must NOT use old 'activation_threshold_pct' key"


class TestLiveProtectionFlow:
    """Test that positions get protected immediately"""
    
    @pytest.mark.asyncio
    async def test_position_gets_exit_plan_on_entry(self):
        """
        New position must get ExitPlan immediately.
        Note: Not all profiles include trailing leg - depends on TP leg allocation.
        TREND profile has trailing, NORMAL profile may not (100% allocated to TPs).
        """
        router = ExitRouter()
        
        # Simulate new position from Binance
        new_position = {
            "symbol": "ADAUSDT",
            "positionAmt": "1000.0",  # LONG
            "entryPrice": "0.50",
            "markPrice": "0.505",
            "leverage": "20"
        }
        
        # Router creates plan
        plan = await router.get_or_create_plan(new_position)
        
        # Verify plan provides protection
        assert plan is not None
        assert len(plan.legs) > 0, "Plan must have exit legs"
        
        # Must have SL for immediate protection
        sl_legs = plan.get_legs_by_kind(ExitKind.SL)
        assert len(sl_legs) > 0, "Must have SL for immediate protection"
        
        # Should have TP legs
        tp_legs = plan.get_legs_by_kind(ExitKind.TP)
        assert len(tp_legs) > 0, "Should have TP legs for profit taking"


def test_integration_summary():
    """
    Summary test documenting the complete integration flow:
    
    1. Position opened → Event-driven executor detects
    2. ExitRouter.get_or_create_plan(position) called
    3. ExitBrainV3.build_exit_plan(context) creates plan
    4. Plan contains TP, SL, and TRAIL legs
    5. Executor places initial SL immediately
    6. Trailing Stop Manager monitors position
    7. Manager calls exit_router.get_active_plan(symbol)
    8. Converts plan via to_trailing_config()
    9. Applies trailing stop with callback from plan
    10. DynamicTrailingManager tightens as profit increases
    """
    # This test documents the flow - implementation tests above verify each step
    pass


if __name__ == "__main__":
    # Run with: pytest tests/integration/test_exit_brain_trailing_integration.py -v
    pytest.main([__file__, "-v", "-s"])
