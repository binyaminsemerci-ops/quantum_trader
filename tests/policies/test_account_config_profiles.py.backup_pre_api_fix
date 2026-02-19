"""
Tests for Account Config Profile Integration

EPIC-P10-TESTS-001: Test suite for account-level capital profiles
"""

import pytest
from backend.policies.account_config import (
    AccountConfig,
    register_account,
    get_account,
    get_capital_profile_for_account,
    set_capital_profile_for_account,
    ACCOUNTS,
)


class TestAccountConfigProfiles:
    """Test capital profile integration with AccountConfig."""
    
    def setup_method(self):
        """Reset accounts before each test."""
        ACCOUNTS.clear()
    
    def test_account_config_has_capital_profile_field(self):
        """Test that AccountConfig has capital_profile field."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="normal"
        )
        
        assert account.capital_profile == "normal"
    
    def test_account_config_default_profile_is_micro(self):
        """Test that default profile is micro."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        assert account.capital_profile == "micro"
    
    def test_register_account_with_profile(self):
        """Test registering account with explicit profile."""
        account = AccountConfig(
            name="test_binance",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="normal"
        )
        
        register_account(account)
        
        retrieved = get_account("test_binance")
        assert retrieved.capital_profile == "normal"
    
    def test_get_capital_profile_for_account(self):
        """Test getting capital profile for account."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="low"
        )
        register_account(account)
        
        profile = get_capital_profile_for_account("test_account")
        
        assert profile == "low"
    
    def test_get_capital_profile_for_account_not_found(self):
        """Test getting profile for non-existent account raises error."""
        with pytest.raises(KeyError):
            get_capital_profile_for_account("nonexistent")
    
    def test_set_capital_profile_for_account(self):
        """Test setting capital profile at runtime."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="micro"
        )
        register_account(account)
        
        # Promote to low
        set_capital_profile_for_account("test_account", "low")
        
        profile = get_capital_profile_for_account("test_account")
        assert profile == "low"
    
    def test_set_capital_profile_for_account_not_found(self):
        """Test setting profile for non-existent account raises error."""
        with pytest.raises(KeyError):
            set_capital_profile_for_account("nonexistent", "normal")
    
    def test_multiple_accounts_different_profiles(self):
        """Test that multiple accounts can have different profiles."""
        account1 = AccountConfig(
            name="main_binance",
            exchange="binance",
            api_key="key1",
            api_secret="secret1",
            capital_profile="normal"
        )
        
        account2 = AccountConfig(
            name="friend_firi",
            exchange="firi",
            api_key="key2",
            api_secret="secret2",
            client_id="client2",
            capital_profile="micro"
        )
        
        register_account(account1)
        register_account(account2)
        
        assert get_capital_profile_for_account("main_binance") == "normal"
        assert get_capital_profile_for_account("friend_firi") == "micro"
    
    def test_account_profile_upgrade_path(self):
        """Test upgrading account through profile ladder."""
        account = AccountConfig(
            name="test_account",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            capital_profile="micro"
        )
        register_account(account)
        
        # Simulate progression: micro → low → normal
        assert get_capital_profile_for_account("test_account") == "micro"
        
        set_capital_profile_for_account("test_account", "low")
        assert get_capital_profile_for_account("test_account") == "low"
        
        set_capital_profile_for_account("test_account", "normal")
        assert get_capital_profile_for_account("test_account") == "normal"
