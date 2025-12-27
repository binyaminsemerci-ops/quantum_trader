"""
Multi-Account Routing Tests

EPIC-MT-ACCOUNTS-001: Test multi-account routing logic.
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.policies.account_config import (
    AccountConfig,
    register_account,
    get_account,
    get_default_account_for_exchange,
    ACCOUNTS
)
from backend.policies.account_mapping import (
    get_account_for_strategy,
    set_strategy_account_mapping,
    STRATEGY_ACCOUNT_MAP
)
from backend.integrations.exchanges.factory import (
    get_exchange_config_for_account,
    get_exchange_client_for_account,
    ExchangeConfig,
    ExchangeType
)


# ============================================================================
# UNIT TESTS: Account Config
# ============================================================================

def test_register_and_get_account():
    """Can register and retrieve accounts."""
    account = AccountConfig(
        name="test_account_1",
        exchange="binance",
        api_key="test_key",
        api_secret="test_secret"
    )
    
    register_account(account)
    
    retrieved = get_account("test_account_1")
    assert retrieved.name == "test_account_1"
    assert retrieved.exchange == "binance"
    assert retrieved.api_key == "test_key"


def test_get_account_not_found():
    """get_account raises KeyError for unknown account."""
    with pytest.raises(KeyError, match="not found"):
        get_account("nonexistent_account")


def test_get_default_account_for_exchange():
    """get_default_account_for_exchange returns first account for exchange."""
    # Clear accounts
    ACCOUNTS.clear()
    
    # Register test accounts
    register_account(AccountConfig(
        name="binance_1",
        exchange="binance",
        api_key="key1",
        api_secret="secret1"
    ))
    register_account(AccountConfig(
        name="binance_2",
        exchange="binance",
        api_key="key2",
        api_secret="secret2"
    ))
    
    default = get_default_account_for_exchange("binance")
    assert default in ("binance_1", "binance_2")


def test_get_default_account_no_accounts():
    """get_default_account_for_exchange returns None if no accounts."""
    ACCOUNTS.clear()
    
    default = get_default_account_for_exchange("bybit")
    assert default is None


# ============================================================================
# UNIT TESTS: Account Mapping
# ============================================================================

def test_get_account_for_strategy_explicit_mapping():
    """Strategy with explicit mapping returns mapped account."""
    STRATEGY_ACCOUNT_MAP.clear()
    set_strategy_account_mapping({"test_strategy": "custom_account"})
    
    account_name = get_account_for_strategy("test_strategy", "binance")
    
    assert account_name == "custom_account"


def test_get_account_for_strategy_default():
    """Strategy without mapping returns default account."""
    STRATEGY_ACCOUNT_MAP.clear()
    
    account_name = get_account_for_strategy("unmapped_strategy", "bybit")
    
    assert account_name == "main_bybit"


def test_get_account_for_strategy_none():
    """None strategy_id returns default account."""
    account_name = get_account_for_strategy(None, "okx")
    
    assert account_name == "main_okx"


# ============================================================================
# UNIT TESTS: Factory Integration
# ============================================================================

def test_get_exchange_config_for_account():
    """AccountConfig converts to ExchangeConfig correctly."""
    account = AccountConfig(
        name="test_binance",
        exchange="binance",
        api_key="test_key",
        api_secret="test_secret",
        testnet=True
    )
    
    config = get_exchange_config_for_account(account)
    
    assert isinstance(config, ExchangeConfig)
    assert config.exchange == ExchangeType.BINANCE
    assert config.api_key == "test_key"
    assert config.api_secret == "test_secret"
    assert config.testnet is True


def test_get_exchange_config_for_account_with_passphrase():
    """AccountConfig with passphrase converts correctly."""
    account = AccountConfig(
        name="test_okx",
        exchange="okx",
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_passphrase"
    )
    
    config = get_exchange_config_for_account(account)
    
    assert config.exchange == ExchangeType.OKX
    assert config.passphrase == "test_passphrase"


def test_get_exchange_config_for_account_with_client_id():
    """AccountConfig with client_id converts correctly."""
    account = AccountConfig(
        name="test_firi",
        exchange="firi",
        api_key="test_key",
        api_secret="test_secret",
        client_id="test_client_id"
    )
    
    config = get_exchange_config_for_account(account)
    
    assert config.exchange == ExchangeType.FIRI
    assert config.client_id == "test_client_id"


def test_get_exchange_config_invalid_exchange():
    """Invalid exchange name raises ValueError."""
    account = AccountConfig(
        name="test_invalid",
        exchange="invalid_exchange",  # type: ignore
        api_key="test_key",
        api_secret="test_secret"
    )
    
    with pytest.raises(ValueError, match="Invalid exchange name"):
        get_exchange_config_for_account(account)


def test_get_exchange_client_for_account_by_object():
    """get_exchange_client_for_account works with AccountConfig object."""
    ACCOUNTS.clear()
    account = AccountConfig(
        name="test_account",
        exchange="kucoin",
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_passphrase"
    )
    register_account(account)
    
    # Mock get_exchange_client to avoid real HTTP
    with patch("backend.integrations.exchanges.factory.get_exchange_client") as mock_client:
        mock_client.return_value = MagicMock()
        
        client = get_exchange_client_for_account(account)
        
        assert mock_client.called
        assert client is not None


def test_get_exchange_client_for_account_by_name():
    """get_exchange_client_for_account works with account name string."""
    ACCOUNTS.clear()
    account = AccountConfig(
        name="test_account_2",
        exchange="kraken",
        api_key="test_key",
        api_secret="test_secret"
    )
    register_account(account)
    
    # Mock get_exchange_client to avoid real HTTP
    with patch("backend.integrations.exchanges.factory.get_exchange_client") as mock_client:
        mock_client.return_value = MagicMock()
        
        client = get_exchange_client_for_account("test_account_2")
        
        assert mock_client.called
        assert client is not None


# ============================================================================
# INTEGRATION TESTS: Execution Routing
# ============================================================================

def test_resolve_account_for_signal_explicit():
    """Explicit signal.account_name takes priority."""
    from backend.services.execution.execution import resolve_account_for_signal
    
    account = resolve_account_for_signal(
        signal_account_name="friend_1_binance",
        strategy_id="test_strategy",
        exchange_name="binance"
    )
    
    assert account == "friend_1_binance"


def test_resolve_account_for_signal_strategy_mapping():
    """Strategy mapping used when no explicit account."""
    from backend.services.execution.execution import resolve_account_for_signal
    
    STRATEGY_ACCOUNT_MAP.clear()
    set_strategy_account_mapping({"test_strategy_2": "custom_firi"})
    
    account = resolve_account_for_signal(
        signal_account_name=None,
        strategy_id="test_strategy_2",
        exchange_name="firi"
    )
    
    assert account == "custom_firi"


def test_resolve_account_for_signal_default():
    """Default account used when no explicit account or strategy."""
    from backend.services.execution.execution import resolve_account_for_signal
    
    STRATEGY_ACCOUNT_MAP.clear()
    
    account = resolve_account_for_signal(
        signal_account_name=None,
        strategy_id=None,
        exchange_name="bybit"
    )
    
    assert account == "main_bybit"


def test_resolve_account_full_flow():
    """Full flow: signal → exchange routing → account routing."""
    from backend.services.execution.execution import (
        resolve_exchange_for_signal,
        resolve_account_for_signal
    )
    from backend.policies.exchange_policy import set_strategy_exchange_mapping
    
    # Setup: Strategy maps to OKX
    set_strategy_exchange_mapping({"test_strat": "okx"})
    STRATEGY_ACCOUNT_MAP.clear()
    set_strategy_account_mapping({"test_strat": "okx_account_1"})
    
    # Step 1: Resolve exchange
    exchange = resolve_exchange_for_signal(
        signal_exchange=None,
        strategy_id="test_strat"
    )
    assert exchange == "okx"
    
    # Step 2: Resolve account
    account = resolve_account_for_signal(
        signal_account_name=None,
        strategy_id="test_strat",
        exchange_name=exchange
    )
    assert account == "okx_account_1"
