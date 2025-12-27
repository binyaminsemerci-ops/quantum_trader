"""
Tests for Strategy Profile Policy

EPIC-P10-TESTS-001: Test suite for strategy whitelist/blacklist
"""

import pytest
from backend.policies.strategy_profile_policy import (
    is_strategy_allowed,
    check_strategy_allowed,
    get_allowed_strategies,
    add_strategy_to_whitelist,
    add_strategy_to_blacklist,
    remove_strategy_from_whitelist,
    remove_strategy_from_blacklist,
    StrategyNotAllowedError,
    STRATEGIES_WHITELIST,
    STRATEGIES_BLACKLIST,
    GLOBAL_BLACKLIST,
)


class TestStrategyWhitelistBlacklist:
    """Test strategy whitelist/blacklist logic."""
    
    def setup_method(self):
        """Reset policies before each test."""
        STRATEGIES_WHITELIST.clear()
        STRATEGIES_BLACKLIST.clear()
        GLOBAL_BLACKLIST.clear()
    
    def test_allow_all_when_no_whitelist_no_blacklist(self):
        """Test that strategies are allowed when no whitelist/blacklist."""
        assert is_strategy_allowed("micro", "any_strategy")
        assert is_strategy_allowed("normal", "any_strategy")
    
    def test_blacklist_blocks_strategy(self):
        """Test that blacklisted strategy is blocked."""
        STRATEGIES_BLACKLIST["micro"] = {"risky_strategy"}
        
        assert not is_strategy_allowed("micro", "risky_strategy")
        assert is_strategy_allowed("micro", "safe_strategy")
    
    def test_global_blacklist_blocks_all_profiles(self):
        """Test that global blacklist applies to all profiles."""
        GLOBAL_BLACKLIST.add("banned_strategy")
        
        assert not is_strategy_allowed("micro", "banned_strategy")
        assert not is_strategy_allowed("low", "banned_strategy")
        assert not is_strategy_allowed("normal", "banned_strategy")
        assert not is_strategy_allowed("aggressive", "banned_strategy")
    
    def test_whitelist_only_allows_listed_strategies(self):
        """Test that whitelist restricts to only listed strategies."""
        STRATEGIES_WHITELIST["micro"] = {"safe_strategy_1", "safe_strategy_2"}
        
        assert is_strategy_allowed("micro", "safe_strategy_1")
        assert is_strategy_allowed("micro", "safe_strategy_2")
        assert not is_strategy_allowed("micro", "other_strategy")
    
    def test_blacklist_overrides_whitelist(self):
        """Test that blacklist takes precedence over whitelist."""
        STRATEGIES_WHITELIST["micro"] = {"strategy_1", "strategy_2"}
        STRATEGIES_BLACKLIST["micro"] = {"strategy_1"}
        
        assert not is_strategy_allowed("micro", "strategy_1")
        assert is_strategy_allowed("micro", "strategy_2")
    
    def test_check_strategy_allowed_raises_on_blocked(self):
        """Test that check_strategy_allowed raises exception when blocked."""
        STRATEGIES_BLACKLIST["micro"] = {"blocked_strategy"}
        
        with pytest.raises(StrategyNotAllowedError) as exc_info:
            check_strategy_allowed("micro", "blocked_strategy")
        
        assert exc_info.value.profile == "micro"
        assert exc_info.value.strategy_id == "blocked_strategy"
        assert "blocked_strategy" in str(exc_info.value)
    
    def test_check_strategy_allowed_passes_when_allowed(self):
        """Test that check_strategy_allowed passes when allowed."""
        STRATEGIES_WHITELIST["micro"] = {"allowed_strategy"}
        
        # Should not raise
        check_strategy_allowed("micro", "allowed_strategy")
    
    def test_get_allowed_strategies_with_whitelist(self):
        """Test getting allowed strategies when whitelist defined."""
        STRATEGIES_WHITELIST["micro"] = {"strategy_1", "strategy_2", "strategy_3"}
        STRATEGIES_BLACKLIST["micro"] = {"strategy_2"}
        
        allowed = get_allowed_strategies("micro")
        
        assert "strategy_1" in allowed
        assert "strategy_2" not in allowed  # Blacklisted
        assert "strategy_3" in allowed
    
    def test_get_allowed_strategies_without_whitelist(self):
        """Test getting allowed strategies when no whitelist."""
        all_strategies = {"strategy_1", "strategy_2", "strategy_3"}
        STRATEGIES_BLACKLIST["micro"] = {"strategy_2"}
        
        allowed = get_allowed_strategies("micro", all_strategies)
        
        assert "strategy_1" in allowed
        assert "strategy_2" not in allowed  # Blacklisted
        assert "strategy_3" in allowed
    
    def test_add_strategy_to_whitelist(self):
        """Test adding strategy to whitelist at runtime."""
        add_strategy_to_whitelist("micro", "new_strategy")
        
        assert "new_strategy" in STRATEGIES_WHITELIST["micro"]
    
    def test_add_strategy_to_blacklist(self):
        """Test adding strategy to blacklist at runtime."""
        add_strategy_to_blacklist("micro", "bad_strategy")
        
        assert "bad_strategy" in STRATEGIES_BLACKLIST["micro"]
    
    def test_remove_strategy_from_whitelist(self):
        """Test removing strategy from whitelist."""
        STRATEGIES_WHITELIST["micro"] = {"strategy_1", "strategy_2"}
        
        remove_strategy_from_whitelist("micro", "strategy_1")
        
        assert "strategy_1" not in STRATEGIES_WHITELIST["micro"]
        assert "strategy_2" in STRATEGIES_WHITELIST["micro"]
    
    def test_remove_strategy_from_blacklist(self):
        """Test removing strategy from blacklist."""
        STRATEGIES_BLACKLIST["micro"] = {"strategy_1", "strategy_2"}
        
        remove_strategy_from_blacklist("micro", "strategy_1")
        
        assert "strategy_1" not in STRATEGIES_BLACKLIST["micro"]
        assert "strategy_2" in STRATEGIES_BLACKLIST["micro"]
    
    def test_different_profiles_have_different_policies(self):
        """Test that different profiles can have different policies."""
        STRATEGIES_WHITELIST["micro"] = {"safe_only"}
        STRATEGIES_BLACKLIST["normal"] = {"risky_one"}
        
        # Micro: Only safe_only allowed
        assert is_strategy_allowed("micro", "safe_only")
        assert not is_strategy_allowed("micro", "other")
        
        # Normal: All except risky_one allowed
        assert is_strategy_allowed("normal", "safe_only")
        assert is_strategy_allowed("normal", "other")
        assert not is_strategy_allowed("normal", "risky_one")
