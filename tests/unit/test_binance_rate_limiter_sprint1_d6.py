"""
Test Suite for Binance Rate Limiter and Client Wrapper

SPRINT 1 - D6: Binance Global Rate Limiter
Tests:
- GlobalRateLimiter token bucket behavior
- BinanceClientWrapper retry logic for -1003/-1015
- Integration with mock Binance client
"""
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Import rate limiter components
from backend.integrations.binance.rate_limiter import GlobalRateLimiter, reset_rate_limiter
from backend.integrations.binance.client_wrapper import BinanceClientWrapper, BinanceAPIError, reset_binance_wrapper


class TestGlobalRateLimiter:
    """Test GlobalRateLimiter token bucket implementation."""
    
    @pytest.fixture(autouse=True)
    def reset_limiter(self):
        """Reset global limiter before each test."""
        reset_rate_limiter()
        yield
        reset_rate_limiter()
    
    @pytest.mark.asyncio
    async def test_basic_acquire(self):
        """Test basic token acquisition."""
        limiter = GlobalRateLimiter(max_requests_per_minute=60, max_burst=10)
        
        # Should allow immediate acquisition
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        
        assert elapsed < 0.1, "Basic acquire should be immediate"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_blocks(self):
        """Test that rate limiter blocks when tokens exhausted."""
        # Very low rate: 10 req/min = 1 req every 6 seconds
        limiter = GlobalRateLimiter(max_requests_per_minute=10, max_burst=2)
        
        # First 2 should be immediate (burst capacity)
        await limiter.acquire()
        await limiter.acquire()
        
        # Third should block
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        
        # Should have waited for token refill (at least 5 seconds)
        assert elapsed >= 5.0, f"Should block for ~6s, blocked {elapsed:.2f}s"
    
    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = GlobalRateLimiter(max_requests_per_minute=60, max_burst=5)
        
        # Exhaust tokens
        for _ in range(5):
            await limiter.acquire()
        
        # Check stats - should be low
        stats = await limiter.get_stats()
        assert stats['tokens'] < 1.0, "Tokens should be near zero"
        
        # Wait for refill (60 rpm = 1 per second)
        await asyncio.sleep(2.0)
        
        # Check stats - should have refilled
        stats = await limiter.get_stats()
        assert stats['tokens'] >= 1.5, f"Should refill ~2 tokens in 2s, got {stats['tokens']:.2f}"
    
    @pytest.mark.asyncio
    async def test_weighted_requests(self):
        """Test weighted requests (for heavy endpoints)."""
        limiter = GlobalRateLimiter(max_requests_per_minute=60, max_burst=10)
        
        # Heavy request (weight=5)
        await limiter.acquire(weight=5)
        
        stats = await limiter.get_stats()
        # Should have consumed 5 tokens
        assert stats['tokens'] <= 5.0, f"Should consume 5 tokens, remaining: {stats['tokens']:.2f}"
    
    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Test that multiple concurrent acquires are serialized."""
        limiter = GlobalRateLimiter(max_requests_per_minute=60, max_burst=3)
        
        results = []
        
        async def acquire_and_record():
            await limiter.acquire()
            results.append(time.monotonic())
        
        # Launch 5 concurrent acquires
        tasks = [acquire_and_record() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should have 5 timestamps
        assert len(results) == 5
        
        # Later acquires should be significantly delayed
        first_three = results[:3]
        last_two = results[3:]
        
        avg_first = sum(first_three) / 3
        avg_last = sum(last_two) / 2
        
        # Last two should be at least 1 second later (refill rate)
        assert avg_last - avg_first >= 0.8, "Later acquires should be delayed"


class TestBinanceClientWrapper:
    """Test BinanceClientWrapper retry logic."""
    
    @pytest.fixture(autouse=True)
    def reset_wrapper(self):
        """Reset global wrapper before each test."""
        reset_binance_wrapper()
        reset_rate_limiter()
        yield
        reset_binance_wrapper()
        reset_rate_limiter()
    
    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful API call without errors."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter)
        
        # Mock sync function
        mock_func = Mock(return_value={'success': True})
        
        result = await wrapper.call_async(mock_func, arg1='test')
        
        assert result == {'success': True}
        assert mock_func.called
        assert wrapper.total_calls == 1
        assert wrapper.rate_limit_hits == 0
    
    @pytest.mark.asyncio
    async def test_async_call(self):
        """Test async function call."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter)
        
        # Mock async function
        async def mock_async_func(value):
            return {'value': value}
        
        result = await wrapper.call_async(mock_async_func, value=42)
        
        assert result == {'value': 42}
        assert wrapper.total_calls == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_1003_error(self):
        """Test retry logic for -1003 (too many requests) error."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter, max_retries=3, base_backoff_sec=0.1)
        
        # Mock function that fails twice with -1003, then succeeds
        call_count = 0
        def mock_func_with_retry():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Simulate BinanceAPIException
                error = Exception("APIError(code=-1003): Too many requests")
                error.code = -1003
                raise error
            return {'success': True}
        
        start = time.time()
        result = await wrapper.call_async(mock_func_with_retry)
        elapsed = time.time() - start
        
        assert result == {'success': True}
        assert call_count == 3, "Should call 3 times (2 fails + 1 success)"
        assert wrapper.rate_limit_hits == 2
        assert wrapper.retries == 2
        
        # Should have waited for backoff (0.1s + 0.2s = 0.3s minimum)
        assert elapsed >= 0.25, f"Should wait for backoff, elapsed: {elapsed:.2f}s"
    
    @pytest.mark.asyncio
    async def test_retry_on_1015_error(self):
        """Test retry logic for -1015 (rate limit warning) error."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter, max_retries=2, base_backoff_sec=0.1)
        
        # Mock function that fails once with -1015, then succeeds
        call_count = 0
        def mock_func_with_1015():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = Exception("APIError(code=-1015): Too many orders")
                error.code = -1015
                raise error
            return {'success': True}
        
        result = await wrapper.call_async(mock_func_with_1015)
        
        assert result == {'success': True}
        assert call_count == 2
        assert wrapper.rate_limit_hits == 1
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries raises error."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter, max_retries=2, base_backoff_sec=0.05)
        
        # Mock function that always fails with -1003
        def mock_func_always_fails():
            error = Exception("APIError(code=-1003): Too many requests")
            error.code = -1003
            raise error
        
        with pytest.raises(BinanceAPIError) as exc_info:
            await wrapper.call_async(mock_func_always_fails)
        
        assert exc_info.value.code == -1003
        assert wrapper.rate_limit_hits == 3  # max_retries + 1
    
    @pytest.mark.asyncio
    async def test_non_rate_limit_error_immediate_raise(self):
        """Test that non-rate-limit errors are raised immediately."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter, max_retries=3)
        
        # Mock function with different error
        def mock_func_other_error():
            error = Exception("APIError(code=-1000): Unknown error")
            error.code = -1000
            raise error
        
        with pytest.raises(Exception) as exc_info:
            await wrapper.call_async(mock_func_other_error)
        
        assert "-1000" in str(exc_info.value)
        assert wrapper.total_calls == 1
        assert wrapper.rate_limit_hits == 0  # Not a rate limit error
        assert wrapper.retries == 0  # No retries
    
    @pytest.mark.asyncio
    async def test_error_code_extraction(self):
        """Test error code extraction from different exception types."""
        wrapper = BinanceClientWrapper()
        
        # Test with code attribute
        error1 = Exception("Test")
        error1.code = -1003
        assert wrapper._get_error_code(error1) == -1003
        
        # Test with response dict
        error2 = Exception("Test")
        error2.response = {'code': -1015, 'msg': 'Rate limit'}
        assert wrapper._get_error_code(error2) == -1015
        
        # Test with string parsing
        error3 = Exception("APIError(code=-1003): Too many requests")
        assert wrapper._get_error_code(error3) == -1003
        
        # Test unknown error
        error4 = Exception("Generic error")
        assert wrapper._get_error_code(error4) == 0
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter, max_retries=3, base_backoff_sec=0.1)
        
        call_count = 0
        timestamps = []
        
        def mock_func_track_timing():
            nonlocal call_count
            timestamps.append(time.time())
            call_count += 1
            if call_count <= 3:
                error = Exception("APIError(code=-1003)")
                error.code = -1003
                raise error
            return {'success': True}
        
        await wrapper.call_async(mock_func_track_timing)
        
        # Calculate delays between calls
        delays = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Delays should be approximately: 0.1s, 0.2s, 0.4s (exponential)
        assert delays[0] >= 0.08, f"First delay should be ~0.1s, got {delays[0]:.3f}s"
        assert delays[1] >= 0.18, f"Second delay should be ~0.2s, got {delays[1]:.3f}s"
        assert delays[2] >= 0.35, f"Third delay should be ~0.4s, got {delays[2]:.3f}s"
    
    @pytest.mark.asyncio
    async def test_wrapper_stats(self):
        """Test wrapper statistics collection."""
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        wrapper = BinanceClientWrapper(rate_limiter=limiter)
        
        # Successful call
        mock_func = Mock(return_value={'success': True})
        await wrapper.call_async(mock_func)
        
        # Failed call with retry
        call_count = 0
        def mock_func_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = Exception("APIError(code=-1003)")
                error.code = -1003
                raise error
            return {'success': True}
        
        await wrapper.call_async(mock_func_retry)
        
        stats = await wrapper.get_stats()
        
        # total_calls counts all attempts (including retries)
        assert stats['total_calls'] == 3  # 1 successful + 2 attempts (1 fail + 1 success)
        assert stats['rate_limit_hits'] == 1
        assert stats['retries'] == 1
        assert 'success_rate' in stats


class TestIntegration:
    """Integration tests with mock Binance client."""
    
    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset all globals before each test."""
        reset_binance_wrapper()
        reset_rate_limiter()
        yield
        reset_binance_wrapper()
        reset_rate_limiter()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        limiter = GlobalRateLimiter(max_requests_per_minute=120, max_burst=10)
        wrapper = BinanceClientWrapper(rate_limiter=limiter)
        
        call_count = 0
        
        async def mock_api_call(value):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate network delay
            return {'value': value}
        
        # Launch 20 concurrent requests
        tasks = [wrapper.call_async(mock_api_call, value=i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 20
        assert call_count == 20
        assert all(r['value'] == i for i, r in enumerate(results))
    
    @pytest.mark.asyncio
    async def test_rate_limiter_prevents_burst(self):
        """Test that rate limiter prevents excessive burst."""
        # Very restrictive: 10 req/min, burst=3
        limiter = GlobalRateLimiter(max_requests_per_minute=10, max_burst=3)
        wrapper = BinanceClientWrapper(rate_limiter=limiter)
        
        request_times = []
        
        async def mock_api_call():
            request_times.append(time.monotonic())
            return {'success': True}
        
        # Try to make 5 requests rapidly
        start = time.monotonic()
        tasks = [wrapper.call_async(mock_api_call) for _ in range(5)]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start
        
        # First 3 should be immediate (burst)
        # Last 2 should be delayed significantly
        assert len(request_times) == 5
        
        # Should take at least 10 seconds total (rate limit)
        assert elapsed >= 10.0, f"Should be rate limited, took {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
