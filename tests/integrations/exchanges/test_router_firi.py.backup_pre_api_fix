"""
Firi Exchange Router Tests

EPIC-EXCH-003: Verify factory router correctly instantiates Firi adapter.
"""

import pytest

from backend.integrations.exchanges.factory import (
    ExchangeType,
    ExchangeConfig,
    get_exchange_client,
)
from backend.integrations.exchanges.firi_adapter import FiriAdapter


def test_router_creates_firi_adapter():
    """Factory returns FiriAdapter for FIRI exchange type."""
    config = ExchangeConfig(
        exchange=ExchangeType.FIRI,
        api_key="test_key",
        api_secret="test_secret",
        client_id="test_client_id",
        testnet=True
    )
    
    client = get_exchange_client(config)
    
    assert isinstance(client, FiriAdapter)
    assert client.exchange_name == "firi"
    assert client.testnet is True


def test_firi_requires_client_id():
    """Firi adapter raises ExchangeAPIError without client_id (wraps ValueError)."""
    from backend.integrations.exchanges.base import ExchangeAPIError
    
    config = ExchangeConfig(
        exchange=ExchangeType.FIRI,
        api_key="test_key",
        api_secret="test_secret",
        client_id=None  # Missing!
    )
    
    with pytest.raises(ExchangeAPIError, match="client_id"):
        get_exchange_client(config)


def test_exchange_type_enum_includes_firi():
    """ExchangeType enum includes FIRI."""
    assert ExchangeType.FIRI.value == "firi"
    
    # Verify all 6 exchanges
    all_exchanges = {e.value for e in ExchangeType}
    assert "firi" in all_exchanges
