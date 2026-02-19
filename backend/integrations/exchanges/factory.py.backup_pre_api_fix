"""
Exchange Factory & Routing

EPIC-EXCH-001: Factory for creating exchange adapters + symbol routing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from backend.integrations.exchanges.base import IExchangeClient, ExchangeAPIError
from backend.integrations.exchanges.binance_adapter import BinanceAdapter
from backend.integrations.exchanges.bybit_adapter import BybitAdapter
from backend.integrations.exchanges.okx_adapter import OKXAdapter
from backend.integrations.exchanges.kucoin_adapter import KuCoinAdapter
from backend.integrations.exchanges.kraken_adapter import KrakenAdapter
from backend.integrations.exchanges.firi_adapter import FiriAdapter

logger = logging.getLogger(__name__)


class ExchangeType(str, Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    KUCOIN = "kucoin"
    KRAKEN = "kraken"
    FIRI = "firi"


@dataclass
class ExchangeConfig:
    """
    Exchange connection configuration.
    
    Attributes:
        exchange: Exchange type (binance/bybit/okx/kucoin/kraken/firi)
        api_key: API key
        api_secret: API secret
        passphrase: API passphrase (OKX, KuCoin only)
        client_id: Client ID (Firi only)
        testnet: Use testnet/sandbox (default: False)
        futures: Use futures trading (default: True)
        client: Optional pre-initialized client (Binance)
        wrapper: Optional rate limit wrapper (Binance)
    
    Example:
        config = ExchangeConfig(
            exchange=ExchangeType.FIRI,
            api_key=os.getenv("FIRI_API_KEY"),
            api_secret=os.getenv("FIRI_SECRET_KEY"),
            client_id=os.getenv("FIRI_CLIENT_ID"),
            testnet=False,
            futures=False
        )
    """
    exchange: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # OKX, KuCoin only
    client_id: Optional[str] = None  # Firi only
    testnet: bool = False
    futures: bool = True
    client: Optional[any] = None  # Binance Client instance
    wrapper: Optional[any] = None  # BinanceClientWrapper instance


# Global symbol→exchange mapping (default: all to Binance)
_SYMBOL_EXCHANGE_MAP: Dict[str, ExchangeType] = {}


def get_exchange_client(config: ExchangeConfig) -> IExchangeClient:
    """
    Factory function to create exchange adapter.
    
    Creates appropriate adapter based on exchange type.
    Binance: Requires client + wrapper
    Bybit/OKX: Skeleton (NotImplementedError)
    KuCoin/Kraken: HTTP-based adapters (httpx)
    
    Args:
        config: Exchange configuration
    
    Returns:
        Exchange adapter implementing IExchangeClient
    
    Raises:
        ValueError: Unknown exchange type or missing required config
        ExchangeAPIError: Adapter initialization failed
    
    Example:
        config = ExchangeConfig(
            exchange=ExchangeType.KUCOIN,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase
        )
        client = get_exchange_client(config)
        result = await client.place_order(order_request)
    """
    try:
        if config.exchange == ExchangeType.BINANCE:
            if not config.client:
                raise ValueError(
                    "Binance adapter requires 'client' (Binance Client instance) in config"
                )
            
            adapter = BinanceAdapter(
                client=config.client,
                wrapper=config.wrapper,  # Optional
                testnet=config.testnet
            )
            
            logger.info(
                "BinanceAdapter created",
                extra={"exchange": "binance", "testnet": config.testnet}
            )
            return adapter
        
        elif config.exchange == ExchangeType.BYBIT:
            adapter = BybitAdapter(
                api_key=config.api_key,
                api_secret=config.api_secret,
                testnet=config.testnet
            )
            
            logger.info(
                "BybitAdapter created (skeleton)",
                extra={"exchange": "bybit", "testnet": config.testnet}
            )
            return adapter
        
        elif config.exchange == ExchangeType.OKX:
            if not config.passphrase:
                raise ValueError("OKX adapter requires 'passphrase' in config")
            
            adapter = OKXAdapter(
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.passphrase,
                testnet=config.testnet
            )
            
            logger.info(
                "OKXAdapter created (skeleton)",
                extra={"exchange": "okx", "testnet": config.testnet}
            )
            return adapter
        
        elif config.exchange == ExchangeType.KUCOIN:
            if not config.passphrase:
                raise ValueError("KuCoin adapter requires 'passphrase' in config")
            
            adapter = KuCoinAdapter(
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.passphrase,
                testnet=config.testnet
            )
            
            logger.info(
                "KuCoinAdapter created",
                extra={"exchange": "kucoin", "testnet": config.testnet}
            )
            return adapter
        
        elif config.exchange == ExchangeType.KRAKEN:
            adapter = KrakenAdapter(
                api_key=config.api_key,
                api_secret=config.api_secret,
                testnet=config.testnet
            )
            
            logger.info(
                "KrakenAdapter created",
                extra={"exchange": "kraken", "testnet": config.testnet}
            )
            return adapter
        
        elif config.exchange == ExchangeType.FIRI:
            if not config.client_id:
                raise ValueError("Firi adapter requires 'client_id' in config")
            
            adapter = FiriAdapter(
                api_key=config.api_key,
                client_id=config.client_id,
                secret_key=config.api_secret,  # Firi uses 'secret_key' naming
                testnet=config.testnet
            )
            
            logger.info(
                "FiriAdapter created",
                extra={"exchange": "firi", "testnet": config.testnet}
            )
            return adapter
        
        else:
            raise ValueError(f"Unknown exchange type: {config.exchange}")
    
    except Exception as e:
        logger.error(
            f"Failed to create exchange client: {e}",
            extra={"exchange": config.exchange.value},
            exc_info=True
        )
        raise ExchangeAPIError(
            message=f"Failed to create {config.exchange.value} adapter: {e}",
            exchange=config.exchange.value,
            original_error=e
        )


def resolve_exchange_for_symbol(symbol: str) -> ExchangeType:
    """
    Resolve which exchange to use for a symbol.
    
    Checks symbol→exchange mapping. Defaults to Binance for all symbols
    to maintain backward compatibility.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
    
    Returns:
        Exchange type for this symbol
    
    Example:
        exchange = resolve_exchange_for_symbol("BTCUSDT")
        # ExchangeType.BINANCE (default)
        
        # After setting mapping:
        set_symbol_exchange_mapping({"ETHUSDT": ExchangeType.BYBIT})
        exchange = resolve_exchange_for_symbol("ETHUSDT")
        # ExchangeType.BYBIT
    """
    symbol_upper = symbol.upper()
    exchange = _SYMBOL_EXCHANGE_MAP.get(symbol_upper, ExchangeType.BINANCE)
    
    logger.debug(
        f"Symbol routing: {symbol} -> {exchange.value}",
        extra={"symbol": symbol, "exchange": exchange.value}
    )
    
    return exchange


def set_symbol_exchange_mapping(mapping: Dict[str, ExchangeType]) -> None:
    """
    Set symbol→exchange routing table.
    
    Allows configuration of which symbols trade on which exchanges.
    Symbols not in mapping default to Binance.
    
    Args:
        mapping: Dict of symbol -> exchange type
    
    Example:
        set_symbol_exchange_mapping({
            "BTCUSDT": ExchangeType.BINANCE,
            "ETHUSDT": ExchangeType.BYBIT,
            "SOLUSDT": ExchangeType.OKX,
        })
    """
    global _SYMBOL_EXCHANGE_MAP
    
    # Normalize symbols to uppercase
    _SYMBOL_EXCHANGE_MAP = {
        symbol.upper(): exchange
        for symbol, exchange in mapping.items()
    }
    
    logger.info(
        f"Symbol→exchange mapping updated: {len(_SYMBOL_EXCHANGE_MAP)} symbols",
        extra={"symbol_count": len(_SYMBOL_EXCHANGE_MAP)}
    )


def load_symbol_mapping_from_policy(policy_store) -> None:
    """
    Load symbol→exchange mapping from PolicyStore.
    
    Expects policy key: "symbol_exchange_mapping"
    Format: {"BTCUSDT": "binance", "ETHUSDT": "bybit", ...}
    
    Args:
        policy_store: PolicyStore instance
    
    Example:
        from backend.policy_store import PolicyStore
        
        policy_store = PolicyStore()
        load_symbol_mapping_from_policy(policy_store)
    """
    try:
        mapping_raw = policy_store.get("symbol_exchange_mapping", {})
        
        # Convert string exchange names to ExchangeType
        mapping = {
            symbol: ExchangeType(exchange.lower())
            for symbol, exchange in mapping_raw.items()
        }
        
        set_symbol_exchange_mapping(mapping)
        
        logger.info(
            f"Loaded symbol mapping from PolicyStore: {len(mapping)} symbols",
            extra={"source": "policy_store", "symbol_count": len(mapping)}
        )
    
    except Exception as e:
        logger.warning(
            f"Failed to load symbol mapping from PolicyStore: {e}. Using defaults.",
            extra={"error": str(e)},
            exc_info=True
        )


def get_current_symbol_mapping() -> Dict[str, ExchangeType]:
    """
    Get current symbol→exchange mapping.
    
    Returns:
        Copy of current routing table
    
    Example:
        mapping = get_current_symbol_mapping()
        print(mapping)
        # {"BTCUSDT": ExchangeType.BINANCE, ...}
    """
    return _SYMBOL_EXCHANGE_MAP.copy()


# ============================================================================
# MULTI-ACCOUNT SUPPORT (EPIC-MT-ACCOUNTS-001)
# ============================================================================

def get_exchange_config_for_account(account) -> ExchangeConfig:
    """
    Convert AccountConfig to ExchangeConfig for factory.
    
    EPIC-MT-ACCOUNTS-001: Enables multi-account trading by mapping
    account credentials to existing ExchangeConfig model.
    
    Args:
        account: AccountConfig with credentials and exchange info
    
    Returns:
        ExchangeConfig ready for get_exchange_client()
    
    Example:
        from backend.policies.account_config import get_account
        
        account = get_account("main_binance")
        config = get_exchange_config_for_account(account)
        client = get_exchange_client(config)
    """
    from backend.policies.account_config import AccountConfig
    
    if not isinstance(account, AccountConfig):
        raise TypeError(f"Expected AccountConfig, got {type(account)}")
    
    # Map exchange name string to ExchangeType enum
    try:
        exchange_type = ExchangeType(account.exchange)
    except ValueError:
        raise ValueError(
            f"Invalid exchange name: {account.exchange}. "
            f"Supported: {[e.value for e in ExchangeType]}"
        )
    
    return ExchangeConfig(
        exchange=exchange_type,
        api_key=account.api_key,
        api_secret=account.api_secret,
        passphrase=account.passphrase,
        client_id=account.client_id,
        testnet=account.testnet,
        futures=True,  # Default to futures (can be overridden)
        client=None,
        wrapper=None
    )


def get_exchange_client_for_account(account) -> IExchangeClient:
    """
    Create exchange client for a specific trading account.
    
    EPIC-MT-ACCOUNTS-001: Main entry point for multi-account trading.
    
    Args:
        account: AccountConfig or account name (str)
    
    Returns:
        Exchange client for this account
    
    Raises:
        KeyError: Account name not found
        ValueError: Invalid account configuration
    
    Example:
        # By account object
        account = get_account("main_binance")
        client = get_exchange_client_for_account(account)
        
        # By account name
        client = get_exchange_client_for_account("friend_1_firi")
    """
    from backend.policies.account_config import AccountConfig, get_account
    
    # Allow passing account name as string (convenience)
    if isinstance(account, str):
        account = get_account(account)
    
    if not isinstance(account, AccountConfig):
        raise TypeError(f"Expected AccountConfig or str, got {type(account)}")
    
    logger.info(
        "Creating exchange client for account",
        extra={
            "account_name": account.name,
            "exchange": account.exchange,
            "testnet": account.testnet,
            "mode": account.mode
        }
    )
    
    config = get_exchange_config_for_account(account)
    return get_exchange_client(config)


__all__ = [
    "ExchangeType",
    "ExchangeConfig",
    "get_exchange_client",
    "get_exchange_config_for_account",  # EPIC-MT-ACCOUNTS-001
    "get_exchange_client_for_account",  # EPIC-MT-ACCOUNTS-001
    "resolve_exchange_for_symbol",
    "set_exchange_for_symbol",
]
