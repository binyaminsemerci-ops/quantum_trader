"""
Tests for Capital Profiles

EPIC-P10-TESTS-001: Test suite for Prompt 10 GO-LIVE Program
"""

import pytest
from backend.policies.capital_profiles import (
    CapitalProfile,
    get_profile,
    list_profiles,
    get_next_profile,
    get_previous_profile,
    PROFILES,
    PROGRESSION_LADDER,
)


class TestCapitalProfiles:
    """Test capital profile model and functions."""
    
    def test_all_profiles_exist(self):
        """Test that all 4 profiles are defined."""
        assert len(PROFILES) == 4
        assert "micro" in PROFILES
        assert "low" in PROFILES
        assert "normal" in PROFILES
        assert "aggressive" in PROFILES
    
    def test_get_profile_micro(self):
        """Test retrieving micro profile."""
        profile = get_profile("micro")
        assert profile.name == "micro"
        assert profile.max_daily_loss_pct == -0.5
        assert profile.max_weekly_loss_pct == -2.0
        assert profile.max_single_trade_risk_pct == 0.2
        assert profile.max_open_positions == 2
        assert profile.allowed_leverage == 1
    
    def test_get_profile_low(self):
        """Test retrieving low profile."""
        profile = get_profile("low")
        assert profile.name == "low"
        assert profile.max_daily_loss_pct == -1.0
        assert profile.max_weekly_loss_pct == -3.5
        assert profile.max_single_trade_risk_pct == 0.5
        assert profile.max_open_positions == 3
        assert profile.allowed_leverage == 2
    
    def test_get_profile_normal(self):
        """Test retrieving normal profile."""
        profile = get_profile("normal")
        assert profile.name == "normal"
        assert profile.max_daily_loss_pct == -2.0
        assert profile.max_weekly_loss_pct == -7.0
        assert profile.max_single_trade_risk_pct == 1.0
        assert profile.max_open_positions == 5
        assert profile.allowed_leverage == 3
    
    def test_get_profile_aggressive(self):
        """Test retrieving aggressive profile."""
        profile = get_profile("aggressive")
        assert profile.name == "aggressive"
        assert profile.max_daily_loss_pct == -3.5
        assert profile.max_weekly_loss_pct == -12.0
        assert profile.max_single_trade_risk_pct == 2.0
        assert profile.max_open_positions == 8
        assert profile.allowed_leverage == 5
    
    def test_get_profile_invalid(self):
        """Test retrieving invalid profile raises KeyError."""
        with pytest.raises(KeyError):
            get_profile("invalid")
    
    def test_list_profiles(self):
        """Test listing all profiles in order."""
        profiles = list_profiles()
        assert len(profiles) == 4
        assert profiles[0].name == "micro"
        assert profiles[1].name == "low"
        assert profiles[2].name == "normal"
        assert profiles[3].name == "aggressive"
    
    def test_progression_ladder_order(self):
        """Test progression ladder is correctly ordered."""
        assert PROGRESSION_LADDER == ["micro", "low", "normal", "aggressive"]
    
    def test_limits_increase_with_profile(self):
        """Test that limits become less conservative as profile increases."""
        micro = get_profile("micro")
        low = get_profile("low")
        normal = get_profile("normal")
        aggressive = get_profile("aggressive")
        
        # Daily DD limits (more negative = more risk allowed)
        assert micro.max_daily_loss_pct > low.max_daily_loss_pct
        assert low.max_daily_loss_pct > normal.max_daily_loss_pct
        assert normal.max_daily_loss_pct > aggressive.max_daily_loss_pct
        
        # Trade risk increases
        assert micro.max_single_trade_risk_pct < low.max_single_trade_risk_pct
        assert low.max_single_trade_risk_pct < normal.max_single_trade_risk_pct
        assert normal.max_single_trade_risk_pct < aggressive.max_single_trade_risk_pct
        
        # Position limits increase
        assert micro.max_open_positions < low.max_open_positions
        assert low.max_open_positions < normal.max_open_positions
        assert normal.max_open_positions < aggressive.max_open_positions
        
        # Leverage increases
        assert micro.allowed_leverage < low.allowed_leverage
        assert low.allowed_leverage < normal.allowed_leverage
        assert normal.allowed_leverage < aggressive.allowed_leverage


class TestProgressionLadder:
    """Test profile progression logic."""
    
    def test_get_next_profile_micro_to_low(self):
        """Test promotion from micro to low."""
        assert get_next_profile("micro") == "low"
    
    def test_get_next_profile_low_to_normal(self):
        """Test promotion from low to normal."""
        assert get_next_profile("low") == "normal"
    
    def test_get_next_profile_normal_to_aggressive(self):
        """Test promotion from normal to aggressive."""
        assert get_next_profile("normal") == "aggressive"
    
    def test_get_next_profile_aggressive_is_max(self):
        """Test that aggressive has no next profile."""
        assert get_next_profile("aggressive") is None
    
    def test_get_next_profile_invalid(self):
        """Test invalid profile returns None."""
        assert get_next_profile("invalid") is None
    
    def test_get_previous_profile_low_to_micro(self):
        """Test downgrade from low to micro."""
        assert get_previous_profile("low") == "micro"
    
    def test_get_previous_profile_normal_to_low(self):
        """Test downgrade from normal to low."""
        assert get_previous_profile("normal") == "low"
    
    def test_get_previous_profile_aggressive_to_normal(self):
        """Test downgrade from aggressive to normal."""
        assert get_previous_profile("aggressive") == "normal"
    
    def test_get_previous_profile_micro_is_min(self):
        """Test that micro has no previous profile."""
        assert get_previous_profile("micro") is None
    
    def test_get_previous_profile_invalid(self):
        """Test invalid profile returns None."""
        assert get_previous_profile("invalid") is None
