"""
Dependencies for Monitoring Health Service

Provides shared dependencies like EventBus, Redis, HTTP clients.

Author: Quantum Trader AI Team
Date: December 4, 2025
"""

import logging
from typing import Optional
import httpx
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ServiceDependencies:
    """Container for all external dependencies."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        http_timeout: float = 5.0,
    ):
        self.redis_url = redis_url
        self.http_timeout = http_timeout
        
        self._redis_client: Optional[Redis] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize all connections."""
        try:
            # Initialize Redis
            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis_client.ping()
            logger.info(f"[Monitoring Health] Redis connected: {self.redis_url}")
        except Exception as e:
            logger.error(f"[Monitoring Health] Redis connection failed: {e}")
            self._redis_client = None
        
        # Initialize HTTP client
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.http_timeout),
            follow_redirects=True,
        )
        logger.info("[Monitoring Health] HTTP client initialized")
    
    async def cleanup(self) -> None:
        """Cleanup all connections."""
        if self._http_client:
            await self._http_client.aclose()
            logger.info("[Monitoring Health] HTTP client closed")
        
        if self._redis_client:
            await self._redis_client.close()
            logger.info("[Monitoring Health] Redis connection closed")
    
    @property
    def redis_client(self) -> Optional[Redis]:
        """Get Redis client."""
        return self._redis_client
    
    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        return self._http_client
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available."""
        return self._redis_client is not None
