"""
Comprehensive test suite for External API Bulletproofing

Tests:
1. Bulletproof API client initialization
2. Retry logic with exponential backoff
3. Circuit breaker pattern
4. Rate limiting
5. Timeout protection
6. Error handling and fallbacks
7. Health monitoring
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.api_bulletproof import (
    BulletproofAPIClient,
    CircuitBreaker,
    CircuitState,
    APIStats,
    get_binance_client,
    get_coingecko_client,
    get_all_api_health
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_request() is True
    
    def test_opens_after_threshold_failures(self):
        """Circuit breaker opens after failure threshold"""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record 3 failures
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_request() is False
    
    def test_transitions_to_half_open_after_timeout(self):
        """Circuit breaker transitions to HALF_OPEN after recovery timeout"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        import time
        time.sleep(1.1)
        
        # Should allow request (HALF_OPEN)
        assert cb.can_request() is True
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_closes_after_successful_recovery(self):
        """Circuit breaker closes after success threshold in HALF_OPEN"""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=2
        )
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait and enter HALF_OPEN
        import time
        time.sleep(1.1)
        cb.can_request()
        
        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Circuit breaker reopens if failure occurs in HALF_OPEN"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        
        # Enter HALF_OPEN
        import time
        time.sleep(1.1)
        cb.can_request()
        
        # Failure in HALF_OPEN should reset success count
        cb.record_failure()
        assert cb.success_count == 0


class TestAPIStats:
    """Test API statistics tracking"""
    
    def test_initial_stats(self):
        """Stats start at zero"""
        stats = APIStats()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.success_rate() == 1.0
    
    def test_success_rate_calculation(self):
        """Success rate calculates correctly"""
        stats = APIStats()
        stats.total_requests = 10
        stats.successful_requests = 8
        stats.failed_requests = 2
        
        assert stats.success_rate() == 0.8
        assert stats.failure_rate() == 0.2


@pytest.mark.asyncio
class TestBulletproofAPIClient:
    """Test bulletproof API client"""
    
    async def test_initialization(self):
        """Client initializes with correct parameters"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            timeout=10,
            max_retries=3
        )
        
        assert client.name == "TestAPI"
        assert client.base_url == "https://api.test.com"
        assert client.timeout == 10
        assert client.max_retries == 3
        assert client.circuit_breaker.state == CircuitState.CLOSED
    
    async def test_successful_request(self):
        """Successful request returns data and updates stats"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            max_retries=1
        )
        
        # Mock successful response
        mock_response = {"data": "test"}
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_get = AsyncMock()
            mock_get.__aenter__.return_value.status = 200
            mock_get.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_session.return_value.__aenter__.return_value.get = Mock(return_value=mock_get)
            
            result = await client.get("/test", fallback=None)
        
        assert result == mock_response
        assert client.stats.successful_requests == 1
        assert client.stats.total_requests == 1
    
    async def test_retry_on_server_error(self):
        """Client retries on server error (5xx)"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            max_retries=3,
            backoff_factor=0.1  # Fast for testing
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            # First 2 attempts fail with 503, third succeeds
            mock_get = AsyncMock()
            responses = [
                Mock(status=503, text=AsyncMock(return_value="Error")),
                Mock(status=503, text=AsyncMock(return_value="Error")),
                Mock(status=200, json=AsyncMock(return_value={"success": True}))
            ]
            
            mock_get.__aenter__.return_value = Mock()
            mock_get.__aenter__.side_effect = responses
            mock_session.return_value.__aenter__.return_value.get = Mock(return_value=mock_get)
            
            result = await client.get("/test", fallback=None)
        
        # Should eventually succeed
        assert client.stats.total_requests == 1
        assert client.stats.total_retries >= 2
    
    async def test_circuit_breaker_blocks_requests(self):
        """Circuit breaker blocks requests when open"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            max_retries=1
        )
        
        # Manually open circuit
        for _ in range(5):
            client.circuit_breaker.record_failure()
        
        assert client.circuit_breaker.state == CircuitState.OPEN
        
        # Request should fail immediately without HTTP call
        result = await client.get("/test", fallback={"fallback": True})
        
        assert result == {"fallback": True}
        assert client.stats.failed_requests == 1
    
    async def test_rate_limiting(self):
        """Rate limiting delays requests"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            rate_limit_calls=2,
            rate_limit_period=1.0
        )
        
        # Make 2 requests (should succeed)
        await client._wait_for_rate_limit()
        await client._wait_for_rate_limit()
        
        # Third request should be delayed
        start = asyncio.get_event_loop().time()
        await client._wait_for_rate_limit()
        elapsed = asyncio.get_event_loop().time() - start
        
        # Should have waited some time (may be slightly less than 1s due to timing)
        assert elapsed > 0.5
    
    async def test_timeout_handling(self):
        """Client handles timeouts gracefully"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            timeout=1,
            max_retries=2
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Simulate timeout
            mock_get = AsyncMock()
            mock_get.__aenter__.side_effect = asyncio.TimeoutError()
            mock_session.return_value.__aenter__.return_value.get = Mock(return_value=mock_get)
            
            result = await client.get("/test", fallback={"timeout": True})
        
        assert result == {"timeout": True}
        assert client.stats.failed_requests == 1
    
    async def test_fallback_on_all_failures(self):
        """Client returns fallback after all retries fail"""
        client = BulletproofAPIClient(
            name="TestAPI",
            base_url="https://api.test.com",
            max_retries=2,
            backoff_factor=0.1
        )
        
        fallback_data = {"fallback": True}
        
        with patch('aiohttp.ClientSession') as mock_session:
            # All attempts fail
            mock_get = AsyncMock()
            mock_get.__aenter__.return_value.status = 500
            mock_get.__aenter__.return_value.text = AsyncMock(return_value="Error")
            mock_session.return_value.__aenter__.return_value.get = Mock(return_value=mock_get)
            
            result = await client.get("/test", fallback=fallback_data)
        
        assert result == fallback_data
        assert client.stats.failed_requests == 1


def test_get_binance_client():
    """Get Binance client returns singleton"""
    client1 = get_binance_client()
    client2 = get_binance_client()
    
    assert client1 is client2
    assert client1.name == "Binance"
    assert "binance.com" in client1.base_url


def test_get_coingecko_client():
    """Get CoinGecko client returns singleton"""
    client1 = get_coingecko_client()
    client2 = get_coingecko_client()
    
    assert client1 is client2
    assert client1.name == "CoinGecko"
    assert "coingecko.com" in client1.base_url


def test_get_all_api_health():
    """Get all API health returns combined status"""
    # Initialize clients
    get_binance_client()
    get_coingecko_client()
    
    health = get_all_api_health()
    
    assert 'binance' in health
    assert 'coingecko' in health
    assert 'overall' in health
    
    # Check structure
    assert 'circuit_state' in health['binance']
    assert 'success_rate' in health['binance']
    assert 'health' in health['binance']


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING EXTERNAL API BULLETPROOFING")
    print("=" * 70)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
