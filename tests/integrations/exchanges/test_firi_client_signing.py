"""
Firi Adapter Signature Tests

EPIC-EXCH-003: Verify Firi HMAC-SHA256 signature generation.
"""

import hmac
import hashlib
import pytest

from backend.integrations.exchanges.firi_adapter import FiriAdapter


@pytest.fixture
def firi_adapter():
    """Create Firi adapter with test credentials."""
    return FiriAdapter(
        api_key="test_api_key",
        client_id="test_client_id",
        secret_key="test_secret_key",
        testnet=True
    )


def test_firi_signature_format(firi_adapter):
    """Verify Firi signature matches HMAC-SHA256(method + path + timestamp + body)."""
    method = "POST"
    path = "/v2/orders"
    timestamp = "1609459200000"
    body = '{"market":"BTCNOK","side":"buy"}'
    
    # Generate signature using adapter method
    signature = firi_adapter._generate_signature(method, path, timestamp, body)
    
    # Manually compute expected signature
    message = method + path + timestamp + body
    expected_sig = hmac.new(
        "test_secret_key".encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    assert signature == expected_sig


def test_firi_headers_structure(firi_adapter):
    """Verify Firi request headers include all required fields."""
    headers = firi_adapter._build_headers("GET", "/v2/time")
    
    assert "X-Firi-Client-Id" in headers
    assert "X-Firi-Api-Key" in headers
    assert "X-Firi-Signature" in headers
    assert "X-Firi-Timestamp" in headers
    assert "Content-Type" in headers
    
    assert headers["X-Firi-Client-Id"] == "test_client_id"
    assert headers["X-Firi-Api-Key"] == "test_api_key"
    assert headers["Content-Type"] == "application/json"


def test_firi_testnet_base_url(firi_adapter):
    """Verify testnet uses sandbox URL."""
    assert firi_adapter.base_url == "https://api-sandbox.firi.com"


def test_firi_production_base_url():
    """Verify production uses live URL."""
    adapter = FiriAdapter(
        api_key="key",
        client_id="client",
        secret_key="secret",
        testnet=False
    )
    assert adapter.base_url == "https://api.firi.com"


def test_firi_order_type_mapping(firi_adapter):
    """Verify Firi maps OrderType enum correctly."""
    from backend.integrations.exchanges.models import OrderType
    
    assert firi_adapter._map_order_type(OrderType.MARKET) == "market"
    assert firi_adapter._map_order_type(OrderType.LIMIT) == "limit"


def test_firi_order_status_mapping(firi_adapter):
    """Verify Firi maps order status strings to OrderStatus enum."""
    from backend.integrations.exchanges.models import OrderStatus
    
    assert firi_adapter._map_order_status("pending") == OrderStatus.NEW
    assert firi_adapter._map_order_status("open") == OrderStatus.NEW
    assert firi_adapter._map_order_status("filled") == OrderStatus.FILLED
    assert firi_adapter._map_order_status("cancelled") == OrderStatus.CANCELED
    assert firi_adapter._map_order_status("rejected") == OrderStatus.REJECTED
