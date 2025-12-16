"""
Kraken Adapter Signature Tests

EPIC-EXCH-002: Verify Kraken HMAC-SHA512 signature generation.
"""

import base64
import hmac
import hashlib
import urllib.parse
import pytest

from backend.integrations.exchanges.kraken_adapter import KrakenAdapter


@pytest.fixture
def kraken_adapter():
    """Create Kraken adapter with test credentials."""
    # Kraken expects base64-encoded secret
    api_secret = base64.b64encode(b"test_api_secret").decode('utf-8')
    return KrakenAdapter(
        api_key="test_api_key",
        api_secret=api_secret,
        testnet=True
    )


def test_kraken_signature_format(kraken_adapter):
    """Verify Kraken signature matches SHA256(endpoint+nonce+postdata) -> HMAC-SHA512."""
    endpoint = "/derivatives/api/v3/sendorder"
    nonce = "1609459200000"
    postdata = "symbol=PI_XBTUSD&side=buy&size=1"
    
    # Generate signature using adapter method
    signature = kraken_adapter._generate_signature(endpoint, nonce, postdata)
    
    # Manually compute expected signature
    message = endpoint + nonce + postdata
    sha256_hash = hashlib.sha256(message.encode('utf-8')).digest()
    
    api_secret_decoded = base64.b64decode(kraken_adapter.api_secret)
    expected_sig = hmac.new(
        api_secret_decoded,
        sha256_hash,
        hashlib.sha512
    ).digest()
    expected_b64 = base64.b64encode(expected_sig).decode('utf-8')
    
    assert signature == expected_b64


def test_kraken_headers_structure(kraken_adapter):
    """Verify Kraken request headers include all required fields."""
    postdata = "symbol=PI_XBTUSD"
    headers = kraken_adapter._build_headers("/derivatives/api/v3/sendorder", postdata)
    
    assert "APIKey" in headers
    assert "Authent" in headers
    assert "Nonce" in headers
    assert "Content-Type" in headers
    
    assert headers["APIKey"] == "test_api_key"
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"


def test_kraken_testnet_base_url(kraken_adapter):
    """Verify testnet uses demo URL."""
    assert kraken_adapter.base_url == "https://demo-futures.kraken.com"


def test_kraken_production_base_url():
    """Verify production uses live URL."""
    api_secret = base64.b64encode(b"secret").decode('utf-8')
    adapter = KrakenAdapter(
        api_key="key",
        api_secret=api_secret,
        testnet=False
    )
    assert adapter.base_url == "https://futures.kraken.com"


def test_kraken_order_type_mapping(kraken_adapter):
    """Verify Kraken maps OrderType enum correctly."""
    from backend.integrations.exchanges.models import OrderType
    
    assert kraken_adapter._map_order_type(OrderType.MARKET) == "mkt"
    assert kraken_adapter._map_order_type(OrderType.LIMIT) == "lmt"
    assert kraken_adapter._map_order_type(OrderType.STOP_MARKET) == "stp"
    assert kraken_adapter._map_order_type(OrderType.STOP_LIMIT) == "stop_limit"
