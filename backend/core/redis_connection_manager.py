"""
Redis Connection Manager with Robust Reconnect Logic
SPRINT 3 - Part 2: Infrastructure Hardening

Provides automatic reconnection, health checks, and exponential backoff.
"""

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.asyncio.sentinel import Sentinel

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # Sentinel config (for HA)
    use_sentinel: bool = False
    sentinel_hosts: list = None  # [(host, port), ...]
    sentinel_master_name: str = "mymaster"
    
    # Reconnect config
    max_retries: int = 10
    retry_base_delay: float = 1.0  # seconds
    max_retry_delay: float = 30.0  # seconds


class RedisConnectionManager:
    """
    Robust Redis connection manager with automatic reconnection.
    
    Features:
    - Exponential backoff on connection failures
    - Health checks with latency monitoring
    - Support for both direct Redis and Sentinel
    - Automatic reconnection on connection loss
    - Circuit breaker pattern for repeated failures
    
    Usage:
        config = RedisConfig(host="localhost", port=6379)
        manager = RedisConnectionManager(config)
        
        # Get client (auto-connects if needed)
        client = await manager.get_client()
        
        # Use client
        await client.set("key", "value")
        
        # Health check
        health = await manager.health_check()
    """
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[Redis] = None
        self._sentinel: Optional[Sentinel] = None
        self._connection_lock = asyncio.Lock()
        self._last_health_check: Optional[float] = None
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[float] = None
        
        logger.info(
            f"[REDIS] Manager initialized (host={config.host}, port={config.port}, "
            f"sentinel={config.use_sentinel})"
        )
    
    async def get_client(self) -> Redis:
        """
        Get Redis client, connecting if necessary.
        
        Returns:
            Connected Redis client
        
        Raises:
            redis.ConnectionError: If connection fails after all retries
        """
        # Check circuit breaker
        if self._circuit_open_until and time.time() < self._circuit_open_until:
            wait_time = self._circuit_open_until - time.time()
            raise redis.ConnectionError(
                f"Circuit breaker open, retry in {wait_time:.1f}s"
            )
        
        # Return existing client if connected
        if self._client and await self._is_connected(self._client):
            return self._client
        
        # Need to reconnect
        async with self._connection_lock:
            # Check again after acquiring lock
            if self._client and await self._is_connected(self._client):
                return self._client
            
            # Attempt connection with retries
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    logger.info(f"[REDIS] Connection attempt {attempt}/{self.config.max_retries}")
                    
                    if self.config.use_sentinel:
                        self._client = await self._connect_via_sentinel()
                    else:
                        self._client = await self._connect_direct()
                    
                    # Test connection
                    await self._client.ping()
                    
                    # Success
                    self._consecutive_failures = 0
                    self._circuit_open_until = None
                    logger.info("[REDIS] ✅ Connected successfully")
                    
                    return self._client
                
                except Exception as e:
                    logger.warning(f"[REDIS] Connection attempt {attempt} failed: {e}")
                    
                    # Close failed client
                    if self._client:
                        await self._client.close()
                        self._client = None
                    
                    # Last attempt?
                    if attempt == self.config.max_retries:
                        self._consecutive_failures += 1
                        
                        # Open circuit breaker after 3 consecutive failures
                        if self._consecutive_failures >= 3:
                            self._circuit_open_until = time.time() + 60.0  # 1 minute
                            logger.error("[REDIS] Circuit breaker opened (60s)")
                        
                        raise redis.ConnectionError(
                            f"Failed to connect after {self.config.max_retries} attempts"
                        )
                    
                    # Exponential backoff
                    delay = min(
                        self.config.retry_base_delay * (2 ** (attempt - 1)),
                        self.config.max_retry_delay
                    )
                    logger.info(f"[REDIS] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            
            raise redis.ConnectionError("Connection failed (unexpected)")
    
    async def _connect_direct(self) -> Redis:
        """Connect directly to Redis instance."""
        return await redis.Redis(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            db=self.config.db,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=False,  # Keep bytes for compatibility
            retry_on_timeout=True,
            health_check_interval=30
        )
    
    async def _connect_via_sentinel(self) -> Redis:
        """Connect via Redis Sentinel for HA."""
        if not self.config.sentinel_hosts:
            raise ValueError("Sentinel hosts not configured")
        
        # Create Sentinel instance
        self._sentinel = Sentinel(
            self.config.sentinel_hosts,
            socket_timeout=self.config.socket_timeout,
            password=self.config.password
        )
        
        # Get master client
        master = self._sentinel.master_for(
            self.config.sentinel_master_name,
            socket_timeout=self.config.socket_timeout,
            password=self.config.password,
            db=self.config.db
        )
        
        logger.info(
            f"[REDIS] Connected to Sentinel master: {self.config.sentinel_master_name}"
        )
        
        return master
    
    async def _is_connected(self, client: Redis) -> bool:
        """Check if client is connected."""
        try:
            await asyncio.wait_for(client.ping(), timeout=1.0)
            return True
        except:
            return False
    
    async def health_check(self) -> dict:
        """
        Perform health check with latency measurement.
        
        Returns:
            dict with status, latency_ms, and details
        """
        try:
            start = time.time()
            client = await self.get_client()
            await client.ping()
            latency_ms = (time.time() - start) * 1000
            
            self._last_health_check = time.time()
            
            # Get additional info
            info = await client.info("server")
            
            return {
                "status": "OK",
                "latency_ms": round(latency_ms, 2),
                "redis_version": info.get("redis_version"),
                "uptime_seconds": info.get("uptime_in_seconds"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "last_check": time.time()
            }
        
        except redis.ConnectionError as e:
            return {
                "status": "DOWN",
                "error": str(e),
                "consecutive_failures": self._consecutive_failures,
                "circuit_open": self._circuit_open_until is not None
            }
        
        except Exception as e:
            return {
                "status": "DEGRADED",
                "error": str(e)[:100]
            }
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            logger.info("[REDIS] Closing connection")
            await self._client.close()
            self._client = None


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_direct_redis():
    """Example: Direct Redis connection."""
    config = RedisConfig(
        host="localhost",
        port=6379,
        password="your_password",
        max_retries=5
    )
    
    manager = RedisConnectionManager(config)
    
    try:
        # Get client (auto-connects)
        client = await manager.get_client()
        
        # Use client
        await client.set("test_key", "test_value")
        value = await client.get("test_key")
        print(f"Value: {value}")
        
        # Health check
        health = await manager.health_check()
        print(f"Health: {health}")
    
    finally:
        await manager.close()


async def example_sentinel_redis():
    """Example: Redis Sentinel connection (HA)."""
    config = RedisConfig(
        use_sentinel=True,
        sentinel_hosts=[
            ("redis-sentinel-1", 26379),
            ("redis-sentinel-2", 26380),
            ("redis-sentinel-3", 26381),
        ],
        sentinel_master_name="mymaster",
        password="your_password",
        max_retries=5
    )
    
    manager = RedisConnectionManager(config)
    
    try:
        # Get client (connects via Sentinel)
        client = await manager.get_client()
        
        # Use client normally
        await client.set("test_key", "test_value")
        
        # Health check
        health = await manager.health_check()
        print(f"Health: {health}")
    
    finally:
        await manager.close()


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_direct_redis())
