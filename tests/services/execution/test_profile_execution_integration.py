"""
Tests for Profile Guard Integration with Execution

EPIC-P10-TESTS-001: Test suite for profile enforcement in execution flow
"""

import pytest
from backend.services.execution.execution import check_profile_limits_for_signal
from backend.policies.account_config import (
    AccountConfig,
    register_account,
    ACCOUNTS,
)
from backend.policies.strategy_profile_policy import (
    StrategyNotAllowedError,
    STRATEGIES_BLACKLIST,
    STRATEGIES_WHITELIST,
)
from backend.services.risk.profile_guard import ProfileLimitViolationError


class TestProfileExecutionIntegration:
    """Test profile guard integration in execution flow."""
    
    def setup_method(self):
        """Reset state before each test."""
        ACCOUNTS.clear()
        STRATEGIES_WHITELIST.clear()
        STRATEGIES_BLACKLIST.clear()
    
    def test_check_profile_limits_passes_for_valid_signal(self):
        """Test profile check passes for valid signal."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="normal"
        )
        register_account(account)
        
        # Should not raise
        check_profile_limits_for_signal(
            account_name="test_account",
            strategy_id="safe_strategy",
            requested_leverage=2
        )
    
    def test_check_profile_limits_blocks_blacklisted_strategy(self):
        """Test profile check blocks blacklisted strategy."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="micro"
        )
        register_account(account)
        
        STRATEGIES_BLACKLIST["micro"] = {"risky_strategy"}
        
        with pytest.raises(StrategyNotAllowedError):
            check_profile_limits_for_signal(
                account_name="test_account",
                strategy_id="risky_strategy",
                requested_leverage=1
            )
    
    def test_check_profile_limits_blocks_excessive_leverage(self):
        """Test profile check blocks excessive leverage."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="micro"  # Max 1x leverage
        )
        register_account(account)
        
        with pytest.raises(ProfileLimitViolationError) as exc_info:
            check_profile_limits_for_signal(
                account_name="test_account",
                strategy_id="safe_strategy",
                requested_leverage=3
            )
        
        assert "leverage" in str(exc_info.value)
    
    def test_check_profile_limits_different_profiles(self):
        """Test that different accounts enforce different profile limits."""
        # Micro account
        micro_account = AccountConfig(
            name="micro_account",
            exchange="binance",
            api_key="key1",
            api_secret="secret1",
            capital_profile="micro"
        )
        register_account(micro_account)
        
        # Normal account
        normal_account = AccountConfig(
            name="normal_account",
            exchange="binance",
            api_key="key2",
            api_secret="secret2",
            capital_profile="normal"
        )
        register_account(normal_account)
        
        # Micro account: 2x leverage should fail (max 1x)
        with pytest.raises(ProfileLimitViolationError):
            check_profile_limits_for_signal(
                account_name="micro_account",
                strategy_id="test",
                requested_leverage=2
            )
        
        # Normal account: 2x leverage should pass (max 3x)
        check_profile_limits_for_signal(
            account_name="normal_account",
            strategy_id="test",
            requested_leverage=2
        )
    
    def test_check_profile_limits_account_not_found(self):
        """Test profile check fails when account doesn't exist."""
        with pytest.raises(KeyError):
            check_profile_limits_for_signal(
                account_name="nonexistent",
                strategy_id="test",
                requested_leverage=1
            )
    
    def test_check_profile_limits_with_whitelist(self):
        """Test profile check respects strategy whitelist."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="micro"
        )
        register_account(account)
        
        STRATEGIES_WHITELIST["micro"] = {"approved_strategy"}
        
        # Approved strategy should pass
        check_profile_limits_for_signal(
            account_name="test_account",
            strategy_id="approved_strategy",
            requested_leverage=1
        )
        
        # Non-whitelisted strategy should fail
        with pytest.raises(StrategyNotAllowedError):
            check_profile_limits_for_signal(
                account_name="test_account",
                strategy_id="unapproved_strategy",
                requested_leverage=1
            )
    
    def test_full_execution_flow_simulation(self):
        """Test full execution flow with profile checks."""
        # Setup: Main account (normal profile) and friend account (micro profile)
        main_account = AccountConfig(
            name="main_binance",
            exchange="binance",
            api_key="main_key",
            api_secret="main_secret",
            capital_profile="normal"
        )
        
        friend_account = AccountConfig(
            name="friend_firi",
            exchange="firi",
            api_key="friend_key",
            api_secret="friend_secret",
            client_id="friend_client",
            capital_profile="micro"
        )
        
        register_account(main_account)
        register_account(friend_account)
        
        # Setup strategy policies
        STRATEGIES_BLACKLIST["micro"] = {"aggressive_scalper"}
        
        # Scenario 1: Main account with aggressive strategy (should pass)
        check_profile_limits_for_signal(
            account_name="main_binance",
            strategy_id="aggressive_scalper",
            requested_leverage=3
        )
        
        # Scenario 2: Friend account with aggressive strategy (should fail)
        with pytest.raises(StrategyNotAllowedError):
            check_profile_limits_for_signal(
                account_name="friend_firi",
                strategy_id="aggressive_scalper",
                requested_leverage=1
            )
        
        # Scenario 3: Friend account with safe strategy but high leverage (should fail)
        with pytest.raises(ProfileLimitViolationError):
            check_profile_limits_for_signal(
                account_name="friend_firi",
                strategy_id="safe_strategy",
                requested_leverage=2
            )
        
        # Scenario 4: Friend account with safe strategy and low leverage (should pass)
        check_profile_limits_for_signal(
            account_name="friend_firi",
            strategy_id="safe_strategy",
            requested_leverage=1
        )
