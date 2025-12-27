"""
Tests for Exit Brain v3 TP Profile System
"""

import pytest
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
    TPProfile,
    TPProfileLeg,
    TrailingProfile,
    TPProfileMapping,
    MarketRegime,
    TPKind,
    get_tp_and_trailing_profile,
    register_custom_profile,
    calculate_tp_price,
    get_trailing_callback_for_profit,
    DEFAULT_TREND_PROFILE,
    DEFAULT_RANGE_PROFILE,
    DEFAULT_VOLATILE_PROFILE,
    DEFAULT_CHOP_PROFILE,
    DEFAULT_NORMAL_PROFILE
)
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
from backend.domains.exits.exit_brain_v3.models import ExitContext, ExitKind


class TestTPProfileStructure:
    """Test TPProfile data models"""
    
    def test_tp_profile_leg_validation(self):
        """Test TPProfileLeg validation"""
        # Valid leg
        leg = TPProfileLeg(r_multiple=1.0, size_fraction=0.25, kind=TPKind.HARD)
        assert leg.r_multiple == 1.0
        assert leg.size_fraction == 0.25
        
        # Invalid size fraction
        with pytest.raises(ValueError):
            TPProfileLeg(r_multiple=1.0, size_fraction=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            TPProfileLeg(r_multiple=1.0, size_fraction=0.0)  # = 0
        
        # Invalid r_multiple
        with pytest.raises(ValueError):
            TPProfileLeg(r_multiple=-1.0, size_fraction=0.5)
    
    def test_trailing_profile_tightening_curve(self):
        """Test trailing profile with tightening curve"""
        profile = TrailingProfile(
            callback_pct=0.020,
            activation_r=1.0,
            tightening_curve=[
                (3.0, 0.015),
                (5.0, 0.010),
                (2.0, 0.018)  # Unsorted
            ]
        )
        
        # Should be sorted after __post_init__
        assert profile.tightening_curve[0][0] == 2.0
        assert profile.tightening_curve[1][0] == 3.0
        assert profile.tightening_curve[2][0] == 5.0
    
    def test_tp_profile_size_validation(self):
        """Test that TPProfile validates total size"""
        # Valid: sizes sum to 1.0
        profile = TPProfile(
            name="TEST",
            tp_legs=[
                TPProfileLeg(r_multiple=0.5, size_fraction=0.3),
                TPProfileLeg(r_multiple=1.0, size_fraction=0.4),
                TPProfileLeg(r_multiple=2.0, size_fraction=0.3),
            ]
        )
        assert profile.total_tp_size == 1.0
        
        # Valid: sizes sum to < 1.0 (remaining for trailing)
        profile2 = TPProfile(
            name="TEST2",
            tp_legs=[
                TPProfileLeg(r_multiple=0.5, size_fraction=0.3),
                TPProfileLeg(r_multiple=1.0, size_fraction=0.4),
            ],
            trailing=TrailingProfile(callback_pct=0.015, activation_r=1.0)
        )
        assert profile2.total_tp_size == 0.7
        
        # Invalid: sizes sum to > 1.0
        with pytest.raises(ValueError):
            TPProfile(
                name="INVALID",
                tp_legs=[
                    TPProfileLeg(r_multiple=0.5, size_fraction=0.6),
                    TPProfileLeg(r_multiple=1.0, size_fraction=0.6),
                ]
            )


class TestProfileSelection:
    """Test profile selection and fallback logic"""
    
    def test_default_regime_profiles(self):
        """Test that default profiles exist for all regimes"""
        # TREND
        profile, trailing = get_tp_and_trailing_profile("BTCUSDT", "RL_V3", MarketRegime.TREND)
        assert profile.name == "TREND_DEFAULT"
        assert trailing is not None  # Trend should have trailing
        
        # RANGE
        profile, trailing = get_tp_and_trailing_profile("ETHUSDT", "RL_V3", MarketRegime.RANGE)
        assert profile.name == "RANGE_DEFAULT"
        assert trailing is None  # Range should NOT have trailing
        
        # VOLATILE
        profile, trailing = get_tp_and_trailing_profile("ADAUSDT", "RL_V3", MarketRegime.VOLATILE)
        assert profile.name == "VOLATILE_DEFAULT"
        
        # CHOP
        profile, trailing = get_tp_and_trailing_profile("DOTUSDT", "RL_V3", MarketRegime.CHOP)
        assert profile.name == "CHOP_DEFAULT"
        
        # NORMAL
        profile, trailing = get_tp_and_trailing_profile("AVAXUSDT", "RL_V3", MarketRegime.NORMAL)
        assert profile.name == "NORMAL_DEFAULT"
    
    def test_profile_specificity_ordering(self):
        """Test that more specific profiles take precedence"""
        # Register custom profile for specific symbol+strategy+regime
        custom_profile = TPProfile(
            name="CUSTOM_BTC_TREND",
            tp_legs=[
                TPProfileLeg(r_multiple=1.0, size_fraction=0.5),
                TPProfileLeg(r_multiple=3.0, size_fraction=0.5),
            ],
            description="Custom BTC trend profile"
        )
        register_custom_profile(
            custom_profile,
            symbol="BTCUSDT",
            strategy_id="RL_V3",
            regime=MarketRegime.TREND
        )
        
        # Should get custom profile for exact match
        profile, _ = get_tp_and_trailing_profile("BTCUSDT", "RL_V3", MarketRegime.TREND)
        assert profile.name == "CUSTOM_BTC_TREND"
        
        # Should still get default for other symbols
        profile, _ = get_tp_and_trailing_profile("ETHUSDT", "RL_V3", MarketRegime.TREND)
        assert profile.name == "TREND_DEFAULT"
        
        # Should still get default for other strategies on same symbol
        profile, _ = get_tp_and_trailing_profile("BTCUSDT", "OTHER_STRAT", MarketRegime.TREND)
        assert profile.name == "TREND_DEFAULT"
    
    def test_profile_mapping_specificity_score(self):
        """Test specificity scoring"""
        # (symbol, strategy, regime) = most specific
        mapping1 = TPProfileMapping("BTCUSDT", "RL_V3", MarketRegime.TREND)
        assert mapping1.specificity() == 111  # 100 + 10 + 1
        
        # (symbol, strategy, NORMAL) = less specific
        mapping2 = TPProfileMapping("BTCUSDT", "RL_V3", MarketRegime.NORMAL)
        assert mapping2.specificity() == 110  # 100 + 10 + 0
        
        # (*, strategy, regime) = even less specific
        mapping3 = TPProfileMapping("*", "RL_V3", MarketRegime.TREND)
        assert mapping3.specificity() == 11  # 0 + 10 + 1
        
        # (*, *, regime) = least specific (but still not default)
        mapping4 = TPProfileMapping("*", "*", MarketRegime.TREND)
        assert mapping4.specificity() == 1  # 0 + 0 + 1
        
        # (*, *, NORMAL) = default
        mapping5 = TPProfileMapping("*", "*", MarketRegime.NORMAL)
        assert mapping5.specificity() == 0  # 0 + 0 + 0


class TestProfileIntegrationWithPlanner:
    """Test profile system integration with ExitBrainV3"""
    
    @pytest.mark.asyncio
    async def test_profile_based_plan_creation(self):
        """Test that planner uses profiles when use_profiles=True"""
        # Create planner with profiles enabled
        brain = ExitBrainV3(config={"use_profiles": True, "strategy_id": "TEST_STRAT"})
        
        # Create context for TREND regime - use ETHUSDT to avoid custom profile
        ctx = ExitContext(
            symbol="ETHUSDT",
            side="LONG",
            entry_price=3000.0,
            size=1.0,
            leverage=10.0,
            current_price=3030.0,
            market_regime="TRENDING",
            risk_mode="NORMAL"
        )
        
        # Build plan
        plan = await brain.build_exit_plan(ctx)
        
        # Should have profile name
        assert plan.profile_name == "TREND_DEFAULT"
        assert plan.market_regime == "TRENDING"
        
        # Should have TP legs from profile
        tp_legs = plan.get_legs_by_kind(ExitKind.TP)
        assert len(tp_legs) == 3  # TREND profile has 3 TP legs
        
        # Check R multiples are tracked
        assert tp_legs[0].r_multiple == 0.5
        assert tp_legs[1].r_multiple == 1.0
        assert tp_legs[2].r_multiple == 2.0
    
    @pytest.mark.asyncio
    async def test_legacy_plan_creation(self):
        """Test backward compatibility when use_profiles=False"""
        # Create planner with profiles disabled
        brain = ExitBrainV3(config={"use_profiles": False})
        
        ctx = ExitContext(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.1,
            market_regime="NORMAL"
        )
        
        # Build plan
        plan = await brain.build_exit_plan(ctx)
        
        # Should NOT have profile name (legacy mode)
        assert plan.profile_name is None
        
        # Should still have legs (from legacy partial_template)
        assert len(plan.legs) > 0
    
    @pytest.mark.asyncio
    async def test_regime_specific_profiles(self):
        """Test that different regimes use appropriate profiles"""
        brain = ExitBrainV3(config={"use_profiles": True, "strategy_id": "TEST_STRAT2"})
        
        # TREND regime - use SOLUSDT to avoid custom profiles
        ctx_trend = ExitContext(
            symbol="SOLUSDT",
            side="LONG",
            entry_price=100.0,
            size=10.0,
            market_regime="TRENDING"
        )
        plan_trend = await brain.build_exit_plan(ctx_trend)
        assert plan_trend.profile_name == "TREND_DEFAULT"
        assert plan_trend.has_trailing  # Trend should have trailing
        
        # RANGE regime
        ctx_range = ExitContext(
            symbol="SOLUSDT",
            side="LONG",
            entry_price=100.0,
            size=10.0,
            market_regime="RANGE_BOUND"
        )
        plan_range = await brain.build_exit_plan(ctx_range)
        assert plan_range.profile_name == "RANGE_DEFAULT"
        # RANGE profile has no trailing in default config


class TestHelperFunctions:
    """Test helper utility functions"""
    
    def test_calculate_tp_price_long(self):
        """Test TP price calculation for LONG positions"""
        entry = 50000.0
        sl_distance = 0.02  # 2%
        
        # 1R (1 * 2% = 2% profit)
        tp_1r = calculate_tp_price(entry, "LONG", 1.0, sl_distance)
        assert tp_1r == 51000.0  # 50000 * 1.02
        
        # 2R (2 * 2% = 4% profit)
        tp_2r = calculate_tp_price(entry, "LONG", 2.0, sl_distance)
        assert tp_2r == 52000.0  # 50000 * 1.04
    
    def test_calculate_tp_price_short(self):
        """Test TP price calculation for SHORT positions"""
        entry = 50000.0
        sl_distance = 0.02  # 2%
        
        # 1R (1 * 2% = 2% profit)
        tp_1r = calculate_tp_price(entry, "SHORT", 1.0, sl_distance)
        assert tp_1r == 49000.0  # 50000 * 0.98
        
        # 2R (2 * 2% = 4% profit)
        tp_2r = calculate_tp_price(entry, "SHORT", 2.0, sl_distance)
        assert tp_2r == 48000.0  # 50000 * 0.96
    
    def test_trailing_callback_tightening(self):
        """Test trailing callback tightening based on profit"""
        profile = TrailingProfile(
            callback_pct=0.020,  # Start at 2%
            activation_r=1.0,
            tightening_curve=[
                (2.0, 0.015),  # At 2R, tighten to 1.5%
                (4.0, 0.010),  # At 4R, tighten to 1.0%
                (6.0, 0.005),  # At 6R, tighten to 0.5%
            ]
        )
        
        # Below first threshold
        callback = get_trailing_callback_for_profit(profile, 1.5)
        assert callback == 0.020  # Still at initial 2%
        
        # At first threshold
        callback = get_trailing_callback_for_profit(profile, 2.0)
        assert callback == 0.015  # Tightened to 1.5%
        
        # Between thresholds
        callback = get_trailing_callback_for_profit(profile, 3.5)
        assert callback == 0.015  # Still at 1.5% (not reached 4R)
        
        # At second threshold
        callback = get_trailing_callback_for_profit(profile, 4.5)
        assert callback == 0.010  # Tightened to 1.0%
        
        # Beyond last threshold
        callback = get_trailing_callback_for_profit(profile, 10.0)
        assert callback == 0.005  # Tightened to tightest (0.5%)


class TestDefaultProfiles:
    """Test default profile configurations"""
    
    def test_trend_profile_structure(self):
        """Test TREND profile has appropriate structure"""
        profile = DEFAULT_TREND_PROFILE
        
        assert profile.name == "TREND_DEFAULT"
        assert len(profile.tp_legs) == 3
        assert profile.trailing is not None  # Should have trailing
        
        # Check R multiples are ascending
        r_mults = [leg.r_multiple for leg in profile.tp_legs]
        assert r_mults == sorted(r_mults)  # Ascending order
        
        # Should let profits run (higher R multiples)
        assert max(r_mults) >= 2.0
    
    def test_range_profile_structure(self):
        """Test RANGE profile has quick exits"""
        profile = DEFAULT_RANGE_PROFILE
        
        assert profile.name == "RANGE_DEFAULT"
        assert profile.trailing is None  # No trailing in range mode
        
        # Should have lower R multiples (quick exits)
        r_mults = [leg.r_multiple for leg in profile.tp_legs]
        assert max(r_mults) <= 1.0  # Quick exits
    
    def test_volatile_profile_structure(self):
        """Test VOLATILE profile has wider stops"""
        profile = DEFAULT_VOLATILE_PROFILE
        
        assert profile.trailing is not None
        # Wider callback for volatile markets
        assert profile.trailing.callback_pct >= 0.020
    
    def test_all_profiles_sum_to_valid_size(self):
        """Test all default profiles have valid size allocations"""
        profiles = [
            DEFAULT_TREND_PROFILE,
            DEFAULT_RANGE_PROFILE,
            DEFAULT_VOLATILE_PROFILE,
            DEFAULT_CHOP_PROFILE,
            DEFAULT_NORMAL_PROFILE
        ]
        
        for profile in profiles:
            total_size = profile.total_tp_size
            assert 0.0 < total_size <= 1.0, f"{profile.name} has invalid size: {total_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
