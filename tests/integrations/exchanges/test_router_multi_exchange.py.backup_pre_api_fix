"""
Multi-Exchange Router Tests

EPIC-EXCH-002: Verify factory router correctly instantiates KuCoin + Kraken adapters.
"""

import pytest

from backend.integrations.exchanges.factory import (
    ExchangeType,
    ExchangeConfig,
    get_exchange_client,
)
from backend.integrations.exchanges.kucoin_adapter import KuCoinAdapter
from backend.integrations.exchanges.kraken_adapter import KrakenAdapter


def test_router_creates_kucoin_adapter():
    """Factory returns KuCoinAdapter for KUCOIN exchange type."""
    config = ExchangeConfig(
        exchange=ExchangeType.KUCOIN,
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_passphrase",
        testnet=True
    )
    
    client = get_exchange_client(config)
    
    assert isinstance(client, KuCoinAdapter)
    assert client.exchange_name == "kucoin"
    assert client.testnet is True


def test_router_creates_kraken_adapter():
    """Factory returns KrakenAdapter for KRAKEN exchange type."""
    config = ExchangeConfig(
        exchange=ExchangeType.KRAKEN,
        api_key="test_key",
        api_secret="dGVzdF9zZWNyZXQ=",  # base64-encoded
        testnet=True
    )
    
    client = get_exchange_client(config)
    
    assert isinstance(client, KrakenAdapter)
    assert client.exchange_name == "kraken"
    assert client.testnet is True


def test_kucoin_requires_passphrase():
    """KuCoin adapter raises ExchangeAPIError without passphrase (wraps ValueError)."""
    from backend.integrations.exchanges.base import ExchangeAPIError
    
    config = ExchangeConfig(
        exchange=ExchangeType.KUCOIN,
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None  # Missing!
    )
    
    with pytest.raises(ExchangeAPIError, match="passphrase"):
        get_exchange_client(config)


def test_exchange_type_enum_includes_new_exchanges():
    """ExchangeType enum includes KUCOIN and KRAKEN."""
    assert ExchangeType.KUCOIN.value == "kucoin"
    assert ExchangeType.KRAKEN.value == "kraken"
    
    # Verify all 5 exchanges
    all_exchanges = {e.value for e in ExchangeType}
    assert all_exchanges == {"binance", "bybit", "okx", "kucoin", "kraken"}
