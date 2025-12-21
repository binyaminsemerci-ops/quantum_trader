"""
Redis Connection Manager
=========================
Centralized Redis connection management with:
- Auto-reconnection with exponential backoff
- Health monitoring
- Circuit breaker for Redis calls
- Graceful degradation
"""
import asyncio
import redis.asyncio as aioredis
from typing import Optional, Callable, Any
from datetime import datetime, timezone, timedelta
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Centralized Redis connection manager with resilience features
    
    Features:
    - Auto-reconnect with exponential backoff
    - Health monitoring loop
    - Circuit breaker pattern
    - Graceful degradation when Redis unavailable
    - Metrics tracking
    """
    
    def __init__(
        self,
        url: str = "redis://redis:6379",
        health_check_interval: int = 30,
        max_retry_attempts: int = 5,
        circuit_breaker_threshold: int = 5
    ):
        self.url = url
        self.health_check_interval = health_check_interval
        self.max_retry_attempts = max_retry_attempts
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Connection state
        self.client: Optional[aioredis.Redis] = None
        self.healthy = False
        self.connected = False
        
        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.last_health_check = None
        self.consecutive_failures = 0
        
        # Metrics
        self.connection_attempts = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.last_error = None
        self.last_error_time = None
        
        # Circuit breaker state
        self.circuit_open = False
        self.circuit_opened_at = None
        self.circuit_half_open_time = timedelta(minutes=1)
    
    async def start(self):
        """Start connection manager"""
        logger.info(f"[REDIS-MGR] Starting connection manager: {self.url}")
        
        # Initial connection
        await self.connect()
        
        # Start health monitor
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        logger.info("[REDIS-MGR] ✅ Connection manager started")
    
    async def stop(self):
        """Stop connection manager gracefully"""
        logger.info("[REDIS-MGR] Stopping connection manager")
        
        # Cancel health monitor
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close connection
        await self.disconnect()
        
        logger.info("[REDIS-MGR] ✅ Connection manager stopped")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError))
    )
    async def connect(self):
        """Connect to Redis with retry logic"""
        try:
            self.connection_attempts += 1
            logger.info(f"[REDIS-MGR] Connecting to Redis (attempt {self.connection_attempts})...")
            
            # Create Redis client
            self.client = await aioredis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            
            # Success!
            self.connected = True
            self.healthy = True
            self.consecutive_failures = 0
            self.circuit_open = False
            
            logger.info("[REDIS-MGR] ✅ Connected to Redis successfully")
            
        except Exception as e:
            self.connected = False
            self.healthy = False
            self.consecutive_failures += 1
            self.last_error = str(e)
            self.last_error_time = datetime.now(timezone.utc)
            
            logger.error(f"[REDIS-MGR] ❌ Connection failed: {e}")
            
            # Open circuit breaker if too many failures
            if self.consecutive_failures >= self.circuit_breaker_threshold:
                self._open_circuit()
            
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            try:
                await self.client.close()
                logger.info("[REDIS-MGR] Disconnected from Redis")
            except Exception as e:
                logger.warning(f"[REDIS-MGR] Error during disconnect: {e}")
            finally:
                self.client = None
                self.connected = False
                self.healthy = False
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        logger.info("[REDIS-MGR] Health monitor started")
        
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Skip if circuit is open
                if self.circuit_open:
                    # Check if we should try half-open
                    if self._should_try_half_open():
                        logger.info("[REDIS-MGR] Circuit breaker: Trying HALF-OPEN")
                        await self.connect()
                    continue
                
                # Perform health check
                if self.client:
                    try:
                        await asyncio.wait_for(self.client.ping(), timeout=5)
                        
                        # Health check passed
                        self.healthy = True
                        self.consecutive_failures = 0
                        self.last_health_check = datetime.now(timezone.utc)
                        
                        logger.debug("[REDIS-MGR] Health check: OK")
                        
                    except asyncio.TimeoutError:
                        logger.warning("[REDIS-MGR] Health check: TIMEOUT")
                        self.healthy = False
                        self.consecutive_failures += 1
                        await self._handle_health_check_failure()
                        
                    except Exception as e:
                        logger.error(f"[REDIS-MGR] Health check failed: {e}")
                        self.healthy = False
                        self.consecutive_failures += 1
                        await self._handle_health_check_failure()
                else:
                    # Not connected - try to connect
                    logger.info("[REDIS-MGR] Not connected - attempting reconnect")
                    try:
                        await self.connect()
                    except Exception as e:
                        logger.error(f"[REDIS-MGR] Reconnect failed: {e}")
                
            except asyncio.CancelledError:
                logger.info("[REDIS-MGR] Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"[REDIS-MGR] Health monitor error: {e}", exc_info=True)
    
    async def _handle_health_check_failure(self):
        """Handle failed health check"""
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self._open_circuit()
            
            # Try to reconnect
            logger.info("[REDIS-MGR] Attempting reconnect after failures...")
            try:
                await self.disconnect()
                await asyncio.sleep(2)
                await self.connect()
            except Exception as e:
                logger.error(f"[REDIS-MGR] Reconnect failed: {e}")
    
    def _open_circuit(self):
        """Open circuit breaker"""
        if not self.circuit_open:
            self.circuit_open = True
            self.circuit_opened_at = datetime.now(timezone.utc)
            logger.warning(
                f"[REDIS-MGR] 🚨 Circuit breaker OPENED after {self.consecutive_failures} failures"
            )
    
    def _should_try_half_open(self) -> bool:
        """Check if circuit breaker should try half-open state"""
        if not self.circuit_open:
            return False
        
        if not self.circuit_opened_at:
            return True
        
        time_since_open = datetime.now(timezone.utc) - self.circuit_opened_at
        return time_since_open >= self.circuit_half_open_time
    
    async def publish(
        self,
        channel: str,
        message: str,
        fallback: Optional[Callable] = None
    ) -> bool:
        """
        Publish message to Redis channel with resilience
        
        Args:
            channel: Redis channel name
            message: Message to publish
            fallback: Optional fallback function if Redis unavailable
        
        Returns:
            True if published successfully, False otherwise
        """
        # Circuit breaker check
        if self.circuit_open and not self._should_try_half_open():
            logger.debug(f"[REDIS-MGR] Circuit OPEN - skipping publish to {channel}")
            self.failed_operations += 1
            
            # Try fallback
            if fallback:
                try:
                    fallback(channel, message)
                except Exception as e:
                    logger.error(f"[REDIS-MGR] Fallback failed: {e}")
            
            return False
        
        # Check connection
        if not self.healthy:
            logger.warning(f"[REDIS-MGR] Not healthy - skipping publish to {channel}")
            self.failed_operations += 1
            return False
        
        # Attempt publish
        try:
            await self.client.publish(channel, message)
            self.successful_operations += 1
            return True
            
        except Exception as e:
            logger.error(f"[REDIS-MGR] Publish failed to {channel}: {e}")
            self.failed_operations += 1
            self.consecutive_failures += 1
            self.last_error = str(e)
            self.last_error_time = datetime.now(timezone.utc)
            
            # Check if should open circuit
            if self.consecutive_failures >= self.circuit_breaker_threshold:
                self._open_circuit()
            
            return False
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis with fallback"""
        if not self.healthy or self.circuit_open:
            return default
        
        try:
            value = await self.client.get(key)
            self.successful_operations += 1
            return value if value is not None else default
        except Exception as e:
            logger.error(f"[REDIS-MGR] Get failed for {key}: {e}")
            self.failed_operations += 1
            return default
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis"""
        if not self.healthy or self.circuit_open:
            return False
        
        try:
            await self.client.set(key, value, ex=ex)
            self.successful_operations += 1
            return True
        except Exception as e:
            logger.error(f"[REDIS-MGR] Set failed for {key}: {e}")
            self.failed_operations += 1
            return False
    
    def get_stats(self) -> dict:
        """Get connection manager statistics"""
        return {
            "connected": self.connected,
            "healthy": self.healthy,
            "circuit_open": self.circuit_open,
            "connection_attempts": self.connection_attempts,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "circuit_opened_at": self.circuit_opened_at.isoformat() if self.circuit_opened_at else None
        }
