"""
SPRINT 1 - D2: Redis Streams Wrapper for EventBus

Provides robust Redis Streams operations with error handling,
retry logic, and connection management.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisStreamBus:
    """
    Redis Streams wrapper with robust error handling.
    
    Features:
    - XADD with MAXLEN (stream trimming)
    - XREADGROUP with consumer groups
    - XACK for message acknowledgment
    - Automatic stream/group creation
    - Connection health monitoring
    - Timeout handling
    """
    
    STREAM_PREFIX = "quantum:stream:"
    GROUP_PREFIX = "quantum:group:"
    MAX_STREAM_LEN = 10000
    READ_TIMEOUT = 5000  # milliseconds
    READ_COUNT = 10  # messages per batch
    
    def __init__(
        self,
        redis_client: Redis,
        service_name: str = "quantum_trader",
        consumer_id: Optional[str] = None,
    ):
        """
        Initialize Redis Streams bus.
        
        Args:
            redis_client: Async Redis client
            service_name: Service name for consumer groups
            consumer_id: Unique consumer ID
        """
        self.redis = redis_client
        self.service_name = service_name
        
        import uuid
        self.consumer_id = consumer_id or f"{service_name}_{uuid.uuid4().hex[:8]}"
        
        self._is_healthy = True
        self._last_health_check = datetime.utcnow()
        
        logger.info(
            f"RedisStreamBus initialized: service={service_name}, "
            f"consumer={self.consumer_id}"
        )
    
    def _get_stream_name(self, event_type: str) -> str:
        """Get Redis Stream name for event type."""
        return f"{self.STREAM_PREFIX}{event_type}"
    
    def _get_group_name(self, event_type: str) -> str:
        """Get consumer group name for event type."""
        return f"{self.GROUP_PREFIX}{self.service_name}:{event_type}"
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if Redis is healthy, False otherwise
        """
        try:
            # Throttle health checks (once per 5 seconds)
            now = datetime.utcnow()
            if (now - self._last_health_check).total_seconds() < 5.0:
                return self._is_healthy
            
            self._last_health_check = now
            
            # Quick PING check with timeout
            await asyncio.wait_for(self.redis.ping(), timeout=2.0)
            
            if not self._is_healthy:
                logger.info("✅ Redis connection recovered")
                self._is_healthy = True
            
            return True
        
        except Exception as e:
            if self._is_healthy:
                logger.error(f"❌ Redis connection lost: {e}")
                self._is_healthy = False
            return False
    
    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        trace_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[str]:
        """
        Publish event to Redis Stream.
        
        Args:
            event_type: Event type identifier
            payload: Event data (must be JSON-serializable)
            trace_id: Optional trace ID
            source: Optional source identifier
        
        Returns:
            Message ID from Redis, or None if failed
        """
        stream_name = self._get_stream_name(event_type)
        
        message = {
            "event_type": event_type,
            "payload": json.dumps(payload),
            "trace_id": trace_id or "",
            "timestamp": datetime.utcnow().isoformat(),
            "source": source or self.service_name,
        }
        
        try:
            # XADD with MAXLEN (trim old messages)
            message_id = await self.redis.xadd(
                stream_name,
                message,
                maxlen=self.MAX_STREAM_LEN,
                approximate=True,  # Faster trimming
            )
            
            logger.debug(
                f"📤 Published: {event_type} → {stream_name} (id={message_id})"
            )
            
            # Mark as healthy after successful write
            if not self._is_healthy:
                self._is_healthy = True
                logger.info("✅ Redis write successful - marking as healthy")
            
            return message_id
        
        except redis.RedisError as e:
            logger.error(f"❌ Failed to publish to Redis: {e}")
            self._is_healthy = False
            return None
    
    async def ensure_consumer_group(
        self,
        event_type: str,
        start_id: str = "0",
    ) -> bool:
        """
        Ensure consumer group exists for event type.
        
        Args:
            event_type: Event type to create group for
            start_id: Starting message ID (0 = from beginning, $ = new only)
        
        Returns:
            True if group exists/created, False on error
        """
        stream_name = self._get_stream_name(event_type)
        group_name = self._get_group_name(event_type)
        
        try:
            # Create consumer group (idempotent)
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id=start_id,
                mkstream=True,  # Create stream if doesn't exist
            )
            logger.info(f"✅ Consumer group created: {group_name}")
            return True
        
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists - this is OK
                logger.debug(f"Consumer group already exists: {group_name}")
                return True
            else:
                logger.error(f"Failed to create consumer group: {e}")
                return False
        
        except Exception as e:
            logger.error(f"Unexpected error creating consumer group: {e}")
            return False
    
    async def read_messages(
        self,
        event_type: str,
        count: int = READ_COUNT,
        block: int = READ_TIMEOUT,
    ) -> list[tuple[str, dict]]:
        """
        Read messages from stream using consumer group.
        
        Args:
            event_type: Event type to read from
            count: Number of messages to read
            block: Block timeout in milliseconds
        
        Returns:
            List of (message_id, message_data) tuples
        """
        stream_name = self._get_stream_name(event_type)
        group_name = self._get_group_name(event_type)
        
        try:
            # XREADGROUP: read new messages for this consumer
            response = await self.redis.xreadgroup(
                group_name,
                self.consumer_id,
                {stream_name: ">"},  # > = only new messages
                count=count,
                block=block,
            )
            
            if not response:
                return []
            
            # Parse response: [[stream_name, [(msg_id, data), ...]]]
            messages = []
            for stream, entries in response:
                for message_id, data in entries:
                    messages.append((message_id, data))
            
            return messages
        
        except redis.RedisError as e:
            logger.error(f"Failed to read from stream {stream_name}: {e}")
            self._is_healthy = False
            return []
    
    async def acknowledge(
        self,
        event_type: str,
        message_id: str,
    ) -> bool:
        """
        Acknowledge message processing (XACK).
        
        Args:
            event_type: Event type
            message_id: Message ID to acknowledge
        
        Returns:
            True if acknowledged, False on error
        """
        stream_name = self._get_stream_name(event_type)
        group_name = self._get_group_name(event_type)
        
        try:
            await self.redis.xack(stream_name, group_name, message_id)
            logger.debug(f"✅ ACK: {message_id}")
            return True
        
        except redis.RedisError as e:
            logger.error(f"Failed to ACK message {message_id}: {e}")
            return False
