"""
KuCoin Adapter Signature Tests

EPIC-EXCH-002: Verify KuCoin HMAC-SHA256 signature generation.
"""

import base64
import hmac
import hashlib
import pytest

from backend.integrations.exchanges.kucoin_adapter import KuCoinAdapter


@pytest.fixture
def kucoin_adapter():
    """Create KuCoin adapter with test credentials."""
    return KuCoinAdapter(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
        testnet=True
    )


def test_kucoin_signature_format(kucoin_adapter):
    """Verify KuCoin signature matches HMAC-SHA256(timestamp + method + endpoint + body)."""
    timestamp = "1609459200000"
    method = "POST"
    endpoint = "/api/v1/orders"
    body = '{"symbol":"XBTUSDTM","side":"buy"}'
    
    # Generate signature using adapter method
    signature = kucoin_adapter._generate_signature(timestamp, method, endpoint, body)
    
    # Manually compute expected signature
    sign_str = timestamp + method + endpoint + body
    expected_sig = hmac.new(
        "test_api_secret".encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.sha256
    ).digest()
    expected_b64 = base64.b64encode(expected_sig).decode('utf-8')
    
    assert signature == expected_b64


def test_kucoin_passphrase_signature(kucoin_adapter):
    """Verify KuCoin passphrase encryption with HMAC-SHA256."""
    passphrase_sig = kucoin_adapter._generate_passphrase_signature()
    
    # Manually compute expected passphrase signature
    expected_sig = hmac.new(
        "test_api_secret".encode('utf-8'),
        "test_passphrase".encode('utf-8'),
        hashlib.sha256
    ).digest()
    expected_b64 = base64.b64encode(expected_sig).decode('utf-8')
    
    assert passphrase_sig == expected_b64


def test_kucoin_headers_structure(kucoin_adapter):
    """Verify KuCoin request headers include all required fields."""
    headers = kucoin_adapter._build_headers("GET", "/api/v1/timestamp")
    
    assert "KC-API-KEY" in headers
    assert "KC-API-SIGN" in headers
    assert "KC-API-TIMESTAMP" in headers
    assert "KC-API-PASSPHRASE" in headers
    assert "KC-API-KEY-VERSION" in headers
    
    assert headers["KC-API-KEY"] == "test_api_key"
    assert headers["KC-API-KEY-VERSION"] == "2"
    assert headers["Content-Type"] == "application/json"


def test_kucoin_testnet_base_url(kucoin_adapter):
    """Verify testnet uses sandbox URL."""
    assert kucoin_adapter.base_url == "https://api-sandbox-futures.kucoin.com"


def test_kucoin_production_base_url():
    """Verify production uses live URL."""
    adapter = KuCoinAdapter(
        api_key="key",
        api_secret="secret",
        passphrase="pass",
        testnet=False
    )
    assert adapter.base_url == "https://api-futures.kucoin.com"
