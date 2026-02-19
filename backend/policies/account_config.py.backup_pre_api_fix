"""
Multi-Account Configuration

EPIC-MT-ACCOUNTS-001: Private multi-account trading support.
Allows trading for multiple private accounts in same instance.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)


# Supported exchanges
ExchangeName = Literal["binance", "bybit", "okx", "kucoin", "kraken", "firi"]


@dataclass
class AccountConfig:
    """
    Trading account configuration.
    
    Represents a single trading account with its own API credentials.
    Used for private multi-account trading (not multi-tenant SaaS).
    
    Attributes:
        name: Unique account identifier (e.g., "main_binance", "firi_nok", "friend_1")
        exchange: Exchange name
        api_key: API key for this account
        api_secret: API secret for this account
        passphrase: API passphrase (OKX, KuCoin only)
        client_id: Client ID (Firi only)
        testnet: Use testnet/sandbox (default: False)
        mode: Trading mode - "real" or "paper" (default: "real")
        description: Optional human-readable description
        capital_profile: Capital profile for risk limits (default: "micro")
    
    Example:
        main_account = AccountConfig(
            name="main_binance",
            exchange="binance",
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            capital_profile="normal",
            description="Primary Binance account"
        )
        
        friend_account = AccountConfig(
            name="friend_1_firi",
            exchange="firi",
            api_key=os.getenv("FRIEND1_FIRI_API_KEY"),
            api_secret=os.getenv("FRIEND1_FIRI_SECRET_KEY"),
            client_id=os.getenv("FRIEND1_FIRI_CLIENT_ID"),
            capital_profile="micro",
            description="Friend 1 - NOK trading on Firi"
        )
    """
    name: str
    exchange: ExchangeName
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # OKX, KuCoin only
    client_id: Optional[str] = None   # Firi only
    testnet: bool = False
    mode: Literal["real", "paper"] = "real"
    description: Optional[str] = None
    capital_profile: Literal["micro", "low", "normal", "aggressive"] = "micro"  # EPIC-P10


# ============================================================================
# ACCOUNT REGISTRY
# ============================================================================

# Define trading accounts here or load from environment
# This is config-driven, not database-backed (private multi-account, not SaaS)
ACCOUNTS: Dict[str, AccountConfig] = {}


def register_account(account: AccountConfig) -> None:
    """
    Register a trading account.
    
    Args:
        account: Account configuration to register
    
    Example:
        register_account(AccountConfig(
            name="main_binance",
            exchange="binance",
            api_key="...",
            api_secret="..."
        ))
    """
    if account.name in ACCOUNTS:
        logger.warning(
            "Overwriting existing account",
            extra={"account_name": account.name, "exchange": account.exchange}
        )
    
    ACCOUNTS[account.name] = account
    logger.info(
        "Account registered",
        extra={
            "account_name": account.name,
            "exchange": account.exchange,
            "testnet": account.testnet,
            "mode": account.mode
        }
    )


def get_account(name: str) -> AccountConfig:
    """
    Get account configuration by name.
    
    Args:
        name: Account name
    
    Returns:
        Account configuration
    
    Raises:
        KeyError: Account not found
    
    Example:
        account = get_account("main_binance")
        print(account.api_key)
    """
    if name not in ACCOUNTS:
        raise KeyError(
            f"Account '{name}' not found. "
            f"Available accounts: {list(ACCOUNTS.keys())}"
        )
    
    return ACCOUNTS[name]


def list_accounts() -> Dict[str, AccountConfig]:
    """
    Get all registered accounts.
    
    Returns:
        Dictionary of account_name → AccountConfig
    """
    return ACCOUNTS.copy()


def get_default_account_for_exchange(exchange: ExchangeName) -> Optional[str]:
    """
    Get default account name for an exchange.
    
    Returns first account found for the exchange, or None if no accounts.
    
    Args:
        exchange: Exchange name
    
    Returns:
        Account name or None
    
    Example:
        default = get_default_account_for_exchange("binance")
        # Returns "main_binance" if that's the first binance account
    """
    for name, account in ACCOUNTS.items():
        if account.exchange == exchange:
            return name
    return None


# ============================================================================
# AUTO-REGISTER FROM ENVIRONMENT
# ============================================================================

def load_accounts_from_env() -> None:
    """
    Auto-register accounts from environment variables.
    
    Expected format:
        QT_ACCOUNT_<NAME>_EXCHANGE=binance
        QT_ACCOUNT_<NAME>_API_KEY=xxx
        QT_ACCOUNT_<NAME>_API_SECRET=yyy
        QT_ACCOUNT_<NAME>_PASSPHRASE=zzz (optional)
        QT_ACCOUNT_<NAME>_CLIENT_ID=zzz (optional)
        QT_ACCOUNT_<NAME>_TESTNET=true|false (optional)
        QT_ACCOUNT_<NAME>_MODE=real|paper (optional)
        QT_ACCOUNT_<NAME>_DESCRIPTION="..." (optional)
    
    Example:
        # Main Binance account
        export QT_ACCOUNT_MAIN_BINANCE_EXCHANGE=binance
        export QT_ACCOUNT_MAIN_BINANCE_API_KEY=xxx
        export QT_ACCOUNT_MAIN_BINANCE_API_SECRET=yyy
        
        # Friend's Firi account
        export QT_ACCOUNT_FRIEND1_FIRI_EXCHANGE=firi
        export QT_ACCOUNT_FRIEND1_FIRI_API_KEY=xxx
        export QT_ACCOUNT_FRIEND1_FIRI_API_SECRET=yyy
        export QT_ACCOUNT_FRIEND1_FIRI_CLIENT_ID=zzz
    """
    prefix = "QT_ACCOUNT_"
    account_names = set()
    
    # Discover account names from environment
    for key in os.environ.keys():
        if key.startswith(prefix):
            parts = key[len(prefix):].split("_", 1)
            if len(parts) >= 1:
                account_names.add(parts[0])
    
    # Register each account
    for account_name in account_names:
        name_lower = account_name.lower()
        
        exchange = os.getenv(f"{prefix}{account_name}_EXCHANGE")
        api_key = os.getenv(f"{prefix}{account_name}_API_KEY")
        api_secret = os.getenv(f"{prefix}{account_name}_API_SECRET")
        
        if not (exchange and api_key and api_secret):
            logger.warning(
                "Incomplete account configuration in environment",
                extra={"account_prefix": account_name}
            )
            continue
        
        account = AccountConfig(
            name=name_lower,
            exchange=exchange.lower(),  # type: ignore
            api_key=api_key,
            api_secret=api_secret,
            passphrase=os.getenv(f"{prefix}{account_name}_PASSPHRASE"),
            client_id=os.getenv(f"{prefix}{account_name}_CLIENT_ID"),
            testnet=os.getenv(f"{prefix}{account_name}_TESTNET", "false").lower() == "true",
            mode=os.getenv(f"{prefix}{account_name}_MODE", "real"),  # type: ignore
            description=os.getenv(f"{prefix}{account_name}_DESCRIPTION")
        )
        
        register_account(account)


# Auto-load on module import
load_accounts_from_env()


# ============================================================================
# LEGACY COMPATIBILITY: Register default account from existing env vars
# ============================================================================

def register_legacy_accounts() -> None:
    """
    Register accounts from legacy environment variables.
    
    Provides backward compatibility with existing single-account setup.
    Checks for BINANCE_API_KEY, FIRI_API_KEY, etc.
    """
    # Binance
    binance_key = os.getenv("BINANCE_API_KEY")
    binance_secret = os.getenv("BINANCE_API_SECRET")
    if binance_key and binance_secret and "main_binance" not in ACCOUNTS:
        register_account(AccountConfig(
            name="main_binance",
            exchange="binance",
            api_key=binance_key,
            api_secret=binance_secret,
            description="Legacy Binance account"
        ))
    
    # Firi
    firi_key = os.getenv("FIRI_API_KEY")
    firi_secret = os.getenv("FIRI_SECRET_KEY")
    firi_client_id = os.getenv("FIRI_CLIENT_ID")
    if firi_key and firi_secret and "main_firi" not in ACCOUNTS:
        register_account(AccountConfig(
            name="main_firi",
            exchange="firi",
            api_key=firi_key,
            api_secret=firi_secret,
            client_id=firi_client_id,
            description="Legacy Firi account"
        ))
    
    # Add more exchanges as needed...


# Auto-register legacy accounts on import
register_legacy_accounts()


# ============================================================================
# CAPITAL PROFILE HELPERS (EPIC-P10)
# ============================================================================

def get_capital_profile_for_account(account_name: str) -> str:
    """
    Get capital profile for account.
    
    Args:
        account_name: Account name
        
    Returns:
        Capital profile name (micro/low/normal/aggressive)
        
    Raises:
        KeyError: If account not found
        
    Example:
        >>> profile = get_capital_profile_for_account("main_binance")
        >>> print(profile)
        "normal"
    """
    account = get_account(account_name)
    return account.capital_profile


def set_capital_profile_for_account(
    account_name: str,
    profile: Literal["micro", "low", "normal", "aggressive"]
) -> None:
    """
    Set capital profile for account (runtime modification).
    
    Args:
        account_name: Account name
        profile: New capital profile
        
    Raises:
        KeyError: If account not found
        
    Example:
        >>> set_capital_profile_for_account("main_binance", "normal")
    """
    account = get_account(account_name)
    account.capital_profile = profile
    logger.info(
        "Capital profile updated",
        extra={
            "account_name": account_name,
            "new_profile": profile
        }
    )
