"""
Global Rate Limiter for Binance API
Implements token bucket algorithm to respect Binance rate limits.

SPRINT 1 - D6: Binance Global Rate Limiter
"""
import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    Token bucket rate limiter for Binance API calls.
    
    Binance futures have multiple rate limits:
    - Order rate limits: 300 orders / 10s per account
    - Request rate limits: 2400 weight / minute per IP
    - Some endpoints have weight > 1
    
    This implementation uses a simple token bucket for requests/minute.
    
    Features:
    - Async-safe with asyncio.Lock
    - Configurable rate and burst capacity
    - Automatically replenishes tokens over time
    - Blocks/awaits when rate limit is reached
    
    Args:
        max_requests_per_minute: Maximum requests allowed per minute (default: 1200)
        max_burst: Maximum burst capacity (default: 50)
    
    Example:
        limiter = GlobalRateLimiter(max_requests_per_minute=1200)
        
        # Before making API call
        await limiter.acquire()
        result = await client.some_api_call()
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 1200,
        max_burst: int = 50
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_burst = max_burst
        
        # Token bucket state
        self.tokens = float(max_burst)  # Start with full burst capacity
        self.max_tokens = float(max_burst)
        self.refill_rate = max_requests_per_minute / 60.0  # tokens per second
        self.last_refill = time.monotonic()
        
        # Async lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(
            f"[RATE-LIMITER] Initialized: {max_requests_per_minute} req/min, "
            f"burst={max_burst}, refill={self.refill_rate:.2f} tok/sec"
        )
    
    async def acquire(self, weight: int = 1) -> None:
        """
        Acquire tokens before making an API call.
        Blocks/awaits if insufficient tokens available.
        
        Args:
            weight: Request weight (default: 1). Some Binance endpoints have weight > 1.
        
        Note:
            This method will block until enough tokens are available.
            For high-weight requests, it may block longer.
        """
        async with self._lock:
            while True:
                # Refill tokens based on time elapsed
                now = time.monotonic()
                elapsed = now - self.last_refill
                
                # Add tokens based on elapsed time
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + (elapsed * self.refill_rate)
                )
                self.last_refill = now
                
                # Check if we have enough tokens
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                
                # Calculate wait time for next token
                tokens_needed = weight - self.tokens
                wait_time = tokens_needed / self.refill_rate
                
                # Cap wait time to prevent infinite loops
                wait_time = min(wait_time, 60.0)
                
                logger.debug(
                    f"[RATE-LIMITER] Insufficient tokens ({self.tokens:.2f}/{weight}), "
                    f"waiting {wait_time:.2f}s"
                )
                
                # Release lock while waiting to allow other coroutines
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
    
    async def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dict with tokens, refill_rate, max_tokens
        """
        async with self._lock:
            # Refill tokens based on current time
            now = time.monotonic()
            elapsed = now - self.last_refill
            current_tokens = min(
                self.max_tokens,
                self.tokens + (elapsed * self.refill_rate)
            )
            
            return {
                "tokens": current_tokens,
                "max_tokens": self.max_tokens,
                "refill_rate_per_sec": self.refill_rate,
                "max_requests_per_minute": self.max_requests_per_minute,
                "utilization_pct": (1.0 - current_tokens / self.max_tokens) * 100
            }


# Global singleton instance
_global_rate_limiter: Optional[GlobalRateLimiter] = None


def get_rate_limiter() -> GlobalRateLimiter:
    """
    Get or create the global rate limiter singleton.
    
    Configuration via environment variables:
    - BINANCE_RATE_LIMIT_RPM: Requests per minute (default: 1200)
    - BINANCE_RATE_LIMIT_BURST: Burst capacity (default: 50)
    
    Returns:
        GlobalRateLimiter singleton instance
    """
    global _global_rate_limiter
    
    if _global_rate_limiter is None:
        # Read configuration from environment
        rpm = int(os.getenv("BINANCE_RATE_LIMIT_RPM", "1200"))
        burst = int(os.getenv("BINANCE_RATE_LIMIT_BURST", "50"))
        
        _global_rate_limiter = GlobalRateLimiter(
            max_requests_per_minute=rpm,
            max_burst=burst
        )
        
        logger.info(f"[OK] Global Binance rate limiter created: {rpm} rpm, burst={burst}")
    
    return _global_rate_limiter


def reset_rate_limiter() -> None:
    """Reset global rate limiter (for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None
