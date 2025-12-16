"""
Redis Health Check Helper

Provides utility functions for checking Redis connectivity,
including Sentinel-aware health checks.

Author: Quantum Trader Infrastructure Team
Date: December 4, 2025
Sprint: 3 - Infrastructure Hardening
"""

import logging
from typing import Optional, Tuple
import redis
import redis.asyncio as async_redis
from redis.sentinel import Sentinel

logger = logging.getLogger(__name__)


class RedisHealthChecker:
    """
    Redis health checker with Sentinel support.
    
    Supports both single-node and Sentinel-based Redis deployments.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        sentinel_hosts: Optional[list[Tuple[str, int]]] = None,
        master_name: str = "mymaster",
        password: Optional[str] = None,
    ):
        """
        Initialize Redis health checker.
        
        Args:
            redis_url: Direct Redis URL (e.g., "redis://localhost:6379")
            sentinel_hosts: List of (host, port) tuples for Sentinel nodes
            master_name: Sentinel master name
            password: Redis password
        """
        self.redis_url = redis_url
        self.sentinel_hosts = sentinel_hosts
        self.master_name = master_name
        self.password = password
        
        self._client: Optional[redis.Redis] = None
        self._sentinel: Optional[Sentinel] = None
    
    def connect(self) -> bool:
        """
        Connect to Redis (direct or via Sentinel).
        
        Returns:
            True if connection successful
        """
        try:
            if self.sentinel_hosts:
                # Sentinel mode
                self._sentinel = Sentinel(
                    self.sentinel_hosts,
                    password=self.password,
                    socket_timeout=3.0,
                )
                self._client = self._sentinel.master_for(
                    self.master_name,
                    password=self.password,
                    socket_timeout=3.0,
                )
                logger.info(f"Connected to Redis via Sentinel (master={self.master_name})")
            elif self.redis_url:
                # Direct mode
                self._client = redis.from_url(
                    self.redis_url,
                    password=self.password,
                    socket_timeout=3.0,
                    decode_responses=True,
                )
                logger.info(f"Connected to Redis directly: {self.redis_url}")
            else:
                raise ValueError("Either redis_url or sentinel_hosts must be provided")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def health_check(self) -> dict:
        """
        Perform Redis health check.
        
        Returns:
            {
                "status": "OK" | "DEGRADED" | "DOWN",
                "latency_ms": float,
                "master": str (if Sentinel),
                "replicas": int (if Sentinel),
                "error": str (if DOWN)
            }
        """
        if not self._client:
            if not self.connect():
                return {
                    "status": "DOWN",
                    "error": "Failed to connect to Redis"
                }
        
        import time
        start = time.perf_counter()
        
        try:
            # PING test
            response = self._client.ping()
            latency_ms = (time.perf_counter() - start) * 1000
            
            if not response:
                return {
                    "status": "DOWN",
                    "error": "PING failed"
                }
            
            # Get info
            info = {
                "status": "OK",
                "latency_ms": round(latency_ms, 2),
            }
            
            # If Sentinel, get cluster info
            if self._sentinel:
                try:
                    master_addr = self._sentinel.discover_master(self.master_name)
                    info["master"] = f"{master_addr[0]}:{master_addr[1]}"
                    
                    replicas = self._sentinel.discover_slaves(self.master_name)
                    info["replicas"] = len(replicas)
                except Exception as e:
                    logger.warning(f"Could not get Sentinel info: {e}")
            
            # Check latency threshold
            if latency_ms > 100:
                info["status"] = "DEGRADED"
                info["warning"] = f"High latency: {latency_ms:.2f}ms"
            
            return info
        
        except redis.ConnectionError as e:
            return {
                "status": "DOWN",
                "error": f"Connection failed: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "DOWN",
                "error": f"Health check failed: {str(e)}"
            }
    
    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Example 1: Direct Redis
    checker_direct = RedisHealthChecker(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        password=os.getenv("REDIS_PASSWORD"),
    )
    
    health = checker_direct.health_check()
    print("Direct Redis Health:", health)
    checker_direct.close()
    
    # Example 2: Redis Sentinel
    sentinel_hosts = [
        ("localhost", 26379),
        ("localhost", 26380),
        ("localhost", 26381),
    ]
    
    checker_sentinel = RedisHealthChecker(
        sentinel_hosts=sentinel_hosts,
        master_name="mymaster",
        password=os.getenv("REDIS_PASSWORD"),
    )
    
    health = checker_sentinel.health_check()
    print("Sentinel Redis Health:", health)
    checker_sentinel.close()
