"""
Binance Client Wrapper with Rate Limiting and Retry Logic

SPRINT 1 - D6: Binance Global Rate Limiter
Wraps all Binance API calls with:
- Global rate limiting (respects Binance limits)
- Retry logic for -1003 / -1015 errors
- Exponential backoff
- Comprehensive logging
"""
import asyncio
import logging
import time
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from backend.integrations.binance.rate_limiter import GlobalRateLimiter, get_rate_limiter

logger = logging.getLogger(__name__)

# Type variables for generic wrapper
P = ParamSpec('P')
T = TypeVar('T')


class BinanceAPIError(Exception):
    """
    Wrapper for Binance API errors.
    Compatible with python-binance BinanceAPIException.
    """
    def __init__(self, response, status_code: int, text: str):
        self.response = response
        self.status_code = status_code
        self.text = text
        self.code = self._extract_error_code(response)
        self.message = self._extract_error_message(response)
        super().__init__(self.message)
    
    @staticmethod
    def _extract_error_code(response: Any) -> int:
        """Extract error code from response."""
        if isinstance(response, dict):
            return response.get('code', 0)
        if hasattr(response, 'json'):
            try:
                return response.json().get('code', 0)
            except Exception:
                pass
        return 0
    
    @staticmethod
    def _extract_error_message(response: Any) -> str:
        """Extract error message from response."""
        if isinstance(response, dict):
            return response.get('msg', 'Unknown error')
        if hasattr(response, 'json'):
            try:
                return response.json().get('msg', 'Unknown error')
            except Exception:
                pass
        return str(response)


class BinanceClientWrapper:
    """
    Wrapper for Binance API calls with rate limiting and retry logic.
    
    Features:
    - Global rate limiting (respects Binance API limits)
    - Automatic retry for -1003 (too many requests) and -1015 (rate limit)
    - Exponential backoff (1s, 2s, 4s, 8s, 16s)
    - Comprehensive logging
    - Preserves original exceptions for non-rate-limit errors
    
    Args:
        rate_limiter: GlobalRateLimiter instance (default: uses global singleton)
        max_retries: Maximum retry attempts for rate limit errors (default: 5)
        base_backoff_sec: Base backoff time in seconds (default: 1.0)
    
    Example:
        wrapper = BinanceClientWrapper()
        
        # Wrap sync call
        result = await wrapper.call_async(client.futures_position_information)
        
        # Wrap async call
        result = await wrapper.call_async(client.some_async_method)
    """
    
    # Binance error codes that should trigger retry
    RATE_LIMIT_ERROR_CODES = {
        -1003,  # TOO_MANY_REQUESTS
        -1015,  # TOO_MANY_ORDERS (rate limit warning)
    }
    
    def __init__(
        self,
        rate_limiter: Optional[GlobalRateLimiter] = None,
        max_retries: int = 5,
        base_backoff_sec: float = 1.0
    ):
        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        
        # Statistics
        self.total_calls = 0
        self.rate_limit_hits = 0
        self.retries = 0
        
        logger.info(
            f"[WRAPPER] BinanceClientWrapper initialized: "
            f"max_retries={max_retries}, base_backoff={base_backoff_sec}s"
        )
    
    async def call_async(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> T:
        """
        Wrap an async or sync Binance API call with rate limiting and retry logic.
        
        Args:
            func: Function to call (can be sync or async)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Result from func
        
        Raises:
            BinanceAPIError: If max retries exceeded for rate limit errors
            Exception: Original exception for non-rate-limit errors
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                # Acquire rate limiter token before call
                weight = kwargs.pop('_weight', 1)  # Allow custom weight per call
                await self.rate_limiter.acquire(weight=weight)
                
                self.total_calls += 1
                
                # Call function (handle both sync and async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Run sync function in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                
                # Success - log if we had retries
                if attempt > 0:
                    logger.info(
                        f"[WRAPPER] ✅ Call succeeded after {attempt} retries: {func.__name__}"
                    )
                
                return result
            
            except Exception as e:
                # Check if this is a rate limit error
                error_code = self._get_error_code(e)
                
                if error_code in self.RATE_LIMIT_ERROR_CODES:
                    self.rate_limit_hits += 1
                    
                    if attempt >= self.max_retries:
                        logger.error(
                            f"[WRAPPER] ❌ Max retries ({self.max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        raise BinanceAPIError(
                            response={'code': error_code, 'msg': str(e)},
                            status_code=429,
                            text=str(e)
                        )
                    
                    # Calculate exponential backoff
                    backoff_time = self.base_backoff_sec * (2 ** attempt)
                    
                    self.retries += 1
                    attempt += 1
                    
                    logger.warning(
                        f"[WRAPPER] ⚠️  Rate limit hit (code {error_code}): {func.__name__}, "
                        f"retry {attempt}/{self.max_retries} after {backoff_time:.1f}s"
                    )
                    
                    await asyncio.sleep(backoff_time)
                    last_error = e
                    continue
                
                # Non-rate-limit error - re-raise immediately
                logger.error(
                    f"[WRAPPER] ❌ API error (code {error_code}): {func.__name__}: {e}"
                )
                raise
        
        # Should never reach here, but just in case
        raise last_error or Exception("Unknown error in BinanceClientWrapper")
    
    @staticmethod
    def _get_error_code(error: Exception) -> int:
        """
        Extract Binance error code from exception.
        
        Handles:
        - python-binance BinanceAPIException
        - aiohttp responses
        - Custom BinanceAPIError
        """
        # Check for python-binance BinanceAPIException
        if hasattr(error, 'code'):
            return int(error.code)
        
        # Check for response attribute (aiohttp)
        if hasattr(error, 'response'):
            response = error.response
            if isinstance(response, dict):
                return response.get('code', 0)
            if hasattr(response, 'json'):
                try:
                    return response.json().get('code', 0)
                except Exception:
                    pass
        
        # Check for message parsing (some libraries embed code in message)
        error_str = str(error).lower()
        if '-1003' in error_str or 'too many requests' in error_str:
            return -1003
        if '-1015' in error_str or 'too many orders' in error_str:
            return -1015
        
        return 0
    
    async def get_stats(self) -> dict:
        """
        Get wrapper statistics.
        
        Returns:
            Dict with total_calls, rate_limit_hits, retries, and rate_limiter stats
        """
        rate_limiter_stats = await self.rate_limiter.get_stats()
        
        return {
            "total_calls": self.total_calls,
            "rate_limit_hits": self.rate_limit_hits,
            "retries": self.retries,
            "success_rate": (
                (self.total_calls - self.rate_limit_hits) / self.total_calls * 100
                if self.total_calls > 0 else 100.0
            ),
            "rate_limiter": rate_limiter_stats
        }


# Global singleton wrapper instance
_global_wrapper: Optional[BinanceClientWrapper] = None


def create_binance_wrapper(
    rate_limiter: Optional[GlobalRateLimiter] = None,
    max_retries: int = 5
) -> BinanceClientWrapper:
    """
    Create or get the global Binance client wrapper singleton.
    
    Args:
        rate_limiter: Optional GlobalRateLimiter (default: uses global singleton)
        max_retries: Maximum retry attempts (default: 5)
    
    Returns:
        BinanceClientWrapper singleton instance
    """
    global _global_wrapper
    
    if _global_wrapper is None:
        _global_wrapper = BinanceClientWrapper(
            rate_limiter=rate_limiter,
            max_retries=max_retries
        )
        logger.info("[OK] Global Binance wrapper created")
    
    return _global_wrapper


def reset_binance_wrapper() -> None:
    """Reset global wrapper (for testing)."""
    global _global_wrapper
    _global_wrapper = None
