"""
Tests for Profile Guard

EPIC-P10-TESTS-001: Test suite for profile limit enforcement
"""

import pytest
from backend.services.risk.profile_guard import (
    check_profile_allows_strategy,
    check_leverage_limit,
    check_position_count_limit,
    check_single_trade_risk,
    check_daily_drawdown_limit,
    check_weekly_drawdown_limit,
    check_all_profile_limits,
    ProfileLimitViolationError,
)
from backend.policies.strategy_profile_policy import (
    StrategyNotAllowedError,
    STRATEGIES_BLACKLIST,
    STRATEGIES_WHITELIST,
)


class TestProfileGuard:
    """Test profile guard enforcement."""
    
    def setup_method(self):
        """Reset policies before each test."""
        STRATEGIES_WHITELIST.clear()
        STRATEGIES_BLACKLIST.clear()
    
    def test_check_profile_allows_strategy_passes(self):
        """Test strategy check passes when allowed."""
        # No whitelist/blacklist = allow all
        check_profile_allows_strategy("micro", "any_strategy")
    
    def test_check_profile_allows_strategy_fails(self):
        """Test strategy check fails when blocked."""
        STRATEGIES_BLACKLIST["micro"] = {"blocked_strategy"}
        
        with pytest.raises(StrategyNotAllowedError):
            check_profile_allows_strategy("micro", "blocked_strategy")
    
    def test_check_leverage_limit_passes(self):
        """Test leverage check passes when within limit."""
        # Micro profile: max 1x leverage
        check_leverage_limit("micro", 1)
    
    def test_check_leverage_limit_fails(self):
        """Test leverage check fails when exceeds limit."""
        # Micro profile: max 1x leverage
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_leverage_limit("micro", 2)
        
        assert exc_info.value.profile == "micro"
        assert exc_info.value.limit_type == "leverage"
        assert "2x" in str(exc_info.value)
    
    def test_check_leverage_limit_normal_profile(self):
        """Test leverage limits for normal profile."""
        # Normal profile: max 3x leverage
        check_leverage_limit("normal", 1)
        check_leverage_limit("normal", 2)
        check_leverage_limit("normal", 3)
        
        with pytest.raises(ProfileLimitViolationError):
            check_leverage_limit("normal", 4)
    
    def test_check_position_count_limit_passes(self):
        """Test position count check passes when within limit."""
        # Micro profile: max 2 positions
        check_position_count_limit("micro", current_positions=1, new_positions=1)
    
    def test_check_position_count_limit_fails(self):
        """Test position count check fails when exceeds limit."""
        # Micro profile: max 2 positions
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_position_count_limit("micro", current_positions=2, new_positions=1)
        
        assert exc_info.value.profile == "micro"
        assert exc_info.value.limit_type == "position_count"
    
    def test_check_position_count_limit_normal_profile(self):
        """Test position count limits for normal profile."""
        # Normal profile: max 5 positions
        check_position_count_limit("normal", current_positions=4, new_positions=1)
        
        with pytest.raises(ProfileLimitViolationError):
            check_position_count_limit("normal", current_positions=5, new_positions=1)
    
    def test_check_single_trade_risk_passes(self):
        """Test single trade risk check passes when within limit."""
        # Micro profile: max 0.2% risk
        check_single_trade_risk("micro", trade_risk_pct=0.15)
    
    def test_check_single_trade_risk_fails(self):
        """Test single trade risk check fails when exceeds limit."""
        # Micro profile: max 0.2% risk
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_single_trade_risk("micro", trade_risk_pct=0.3)
        
        assert exc_info.value.profile == "micro"
        assert exc_info.value.limit_type == "single_trade_risk"
    
    def test_check_daily_drawdown_limit_passes(self):
        """Test daily DD check passes when within limit."""
        # Micro profile: max -0.5% daily loss
        check_daily_drawdown_limit("micro", current_daily_pnl_pct=-0.3)
    
    def test_check_daily_drawdown_limit_fails(self):
        """Test daily DD check fails when breached."""
        # Micro profile: max -0.5% daily loss
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_daily_drawdown_limit("micro", current_daily_pnl_pct=-0.6)
        
        assert exc_info.value.profile == "micro"
        assert exc_info.value.limit_type == "daily_drawdown"
    
    def test_check_daily_drawdown_positive_pnl_passes(self):
        """Test daily DD check passes when PnL is positive."""
        check_daily_drawdown_limit("micro", current_daily_pnl_pct=1.0)
    
    def test_check_weekly_drawdown_limit_passes(self):
        """Test weekly DD check passes when within limit."""
        # Micro profile: max -2.0% weekly loss
        check_weekly_drawdown_limit("micro", current_weekly_pnl_pct=-1.5)
    
    def test_check_weekly_drawdown_limit_fails(self):
        """Test weekly DD check fails when breached."""
        # Micro profile: max -2.0% weekly loss
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_weekly_drawdown_limit("micro", current_weekly_pnl_pct=-2.5)
        
        assert exc_info.value.profile == "micro"
        assert exc_info.value.limit_type == "weekly_drawdown"
    
    def test_check_all_profile_limits_passes(self):
        """Test comprehensive check passes when all limits OK."""
        check_all_profile_limits(
            profile_name="normal",
            strategy_id="safe_strategy",
            requested_leverage=2,
            current_positions=3,
            trade_risk_pct=0.8,
            current_daily_pnl_pct=-1.0,
            current_weekly_pnl_pct=-5.0
        )
    
    def test_check_all_profile_limits_fails_on_strategy(self):
        """Test comprehensive check fails on strategy blacklist."""
        STRATEGIES_BLACKLIST["micro"] = {"blocked"}
        
        with pytest.raises(StrategyNotAllowedError):
            check_all_profile_limits(
                profile_name="micro",
                strategy_id="blocked",
                requested_leverage=1,
                current_positions=0
            )
    
    def test_check_all_profile_limits_fails_on_leverage(self):
        """Test comprehensive check fails on leverage limit."""
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_all_profile_limits(
                profile_name="micro",
                strategy_id="safe",
                requested_leverage=5,  # Micro max is 1x
                current_positions=0
            )
        
        assert "leverage" in str(exc_info.value)
    
    def test_check_all_profile_limits_fails_on_positions(self):
        """Test comprehensive check fails on position count."""
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_all_profile_limits(
                profile_name="micro",
                strategy_id="safe",
                requested_leverage=1,
                current_positions=2  # Micro max is 2, adding 1 = 3
            )
        
        assert "position_count" in str(exc_info.value)
    
    def test_check_all_profile_limits_fails_on_daily_dd(self):
        """Test comprehensive check fails on daily DD."""
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_all_profile_limits(
                profile_name="micro",
                strategy_id="safe",
                requested_leverage=1,
                current_positions=0,
                current_daily_pnl_pct=-1.0  # Micro limit is -0.5%
            )
        
        assert "daily_drawdown" in str(exc_info.value)
    
    def test_check_all_profile_limits_skips_optional_checks(self):
        """Test that optional metrics are skipped if not provided."""
        # Should pass without errors (no PnL metrics provided)
        check_all_profile_limits(
            profile_name="micro",
            strategy_id="safe",
            requested_leverage=1,
            current_positions=0
        )
