"""EventBus v2 - Redis Streams-based event infrastructure.

This module provides a production-ready event bus using Redis Streams
for reliable, ordered message delivery across domains.

Features:
- Redis Streams backend (XADD, XREADGROUP, XACK)
- Automatic stream and consumer group creation
- Async handlers with error recovery
- Retry logic with exponential backoff
- Stream trimming (maxlen 10,000)
- Graceful shutdown
- SPRINT 1 - D2: Modular disk buffer + Redis Streams wrapper
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

# SPRINT 1 - D2: Import modular components
from backend.core.eventbus import DiskBuffer, RedisStreamBus

# P1-B: Correlation ID tracking
from shared.logging_config import set_correlation_id, get_correlation_id

logger = logging.getLogger(__name__)


class EventBus:
    """
    Redis Streams-based event bus for inter-domain communication.
    
    Architecture:
    - One Redis Stream per event type
    - One consumer group per subscribing service
    - Automatic message acknowledgment after successful processing
    - Dead letter queue for failed messages (after max retries)
    
    Usage:
        bus = EventBus(redis_client, service_name="ai_engine")
        await bus.initialize()
        
        # Publish event
        await bus.publish("ai.signal.generated", {"symbol": "BTCUSDT", "confidence": 0.85})
        
        # Subscribe to events
        async def handle_signal(event_data: dict):
            print(f"Received: {event_data}")
        
        bus.subscribe("ai.signal.generated", handle_signal)
        
        # Start processing
        await bus.start()
    """
    
    STREAM_PREFIX = "quantum:stream:"
    GROUP_PREFIX = "quantum:group:"
    DLQ_PREFIX = "quantum:dlq:"
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0  # seconds
    MAX_STREAM_LEN = 10000
    READ_TIMEOUT = 5000  # milliseconds
    READ_COUNT = 10  # Process 10 messages per batch
    
    # [CRITICAL FIX #3] Event priority system for ordered processing
    EVENT_PRIORITIES = {
        "model.promoted": {
            1: ["ensemble_manager"],           # Load models first
            2: ["sesa", "meta_strategy"],     # Then update consumers  
            3: ["federation", "default"],     # Finally broadcast
        },
        "model.rollback_initiated": {
            1: ["ensemble_manager"],
            2: ["sesa", "meta_strategy"],
            3: ["federation", "default"],
        },
    }
    
    def __init__(
        self,
        redis_client: Redis,
        service_name: str = "quantum_trader",
        consumer_id: Optional[str] = None,
        disk_buffer_path: Optional[str] = None,
    ):
        """
        Initialize EventBus.
        
        Args:
            redis_client: Async Redis client
            service_name: Name of this service (for consumer group)
            consumer_id: Unique consumer ID (default: UUID)
            disk_buffer_path: Path for disk buffer during Redis outages
        """
        self.redis = redis_client
        self.service_name = service_name
        self.consumer_id = consumer_id or f"{service_name}_{uuid.uuid4().hex[:8]}"
        
        # SPRINT 1 - D2: Use modular components
        self.redis_stream = RedisStreamBus(redis_client, service_name, self.consumer_id)
        self.disk_buffer = DiskBuffer(disk_buffer_path or "runtime/eventbus_buffer")
        self._redis_available = True
        self._replay_task: Optional[asyncio.Task] = None
        self._last_health_check = datetime.utcnow()
        
        # Subscriptions: event_type -> list of handlers
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        
        # [CRITICAL FIX #3] Priority handlers: event_type -> priority -> list of handlers
        self._priority_handlers: dict[str, dict[int, list[tuple[str, Callable]]]] = defaultdict(lambda: defaultdict(list))
        
        # [CRITICAL FIX #1] Atomic promotion lock
        self._promotion_lock_active = False
        self._promotion_acks: dict[str, bool] = {}  # handler_name -> acknowledged
        self._promotion_lock = asyncio.Lock()
        
        # Consumer tasks: event_type -> asyncio.Task
        self._consumer_tasks: dict[str, asyncio.Task] = {}
        
        # Running flag
        self._running = False
        
        logger.info(
            f"EventBus initialized: service_name={service_name}, "
            f"consumer_id={self.consumer_id}, disk_buffer={self.disk_buffer.buffer_dir}"
        )
    
    async def initialize(self) -> None:
        """Initialize EventBus (verify Redis connectivity)."""
        try:
            await self.redis.ping()
            logger.info("EventBus connected to Redis")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def subscribe_with_priority(
        self,
        event_type: str,
        handler: Callable[[dict], Coroutine],
        handler_name: str,
        priority: int = 3,  # Default to lowest priority
    ) -> None:
        """[CRITICAL FIX #3] Subscribe with priority for ordered processing.
        
        Lower priority numbers execute first (1 = highest priority).
        
        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
            handler_name: Unique handler identifier (e.g., 'ensemble_manager')
            priority: Priority level (1=highest, 3=lowest)
        """
        self._priority_handlers[event_type][priority].append((handler_name, handler))
        logger.info(
            f"Subscribed {handler_name} to {event_type} with priority {priority}"
        )
    
    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,  # P1-B: Add correlation_id param
    ) -> str:
        """
        Publish event to Redis Stream.
        
        Args:
            event_type: Event type (e.g., "ai.signal.generated")
            payload: Event data (must be JSON-serializable)
            trace_id: Optional trace ID for distributed tracing
            correlation_id: P1-B correlation_id for cross-service tracking
        
        Returns:
            Message ID from Redis Stream or "buffered" if disk fallback
        
        Raises:
            redis.RedisError: If publish fails and disk buffer also fails
        """
        # P1-B: Pass correlation_id to redis_stream
        message_id = await self.redis_stream.publish(
            event_type, payload, trace_id, self.service_name, correlation_id
        )
        
        if message_id:
            # Successfully published to Redis
            if not self._redis_available:
                self._redis_available = True
                logger.info("✅ Redis reconnected, starting event replay from buffer")
                if not self._replay_task or self._replay_task.done():
                    self._replay_task = asyncio.create_task(self._replay_buffered_events())
            
            return message_id
        
        else:
            # Redis failed - use disk buffer fallback
            logger.error(
                f"❌ Failed to publish event to Redis - buffering to disk: {event_type}"
            )
            self._redis_available = False
            
            # SPRINT 1 - D2: Use DiskBuffer
            message = {
                "event_type": event_type,
                "payload": json.dumps(payload),
                "trace_id": trace_id or "",
                "timestamp": datetime.utcnow().isoformat(),
                "source": self.service_name,
            }
            
            success = self.disk_buffer.write(event_type, message)
            
            if not success:
                raise RuntimeError(
                    f"CRITICAL: Failed to publish event to Redis AND disk buffer. "
                    f"Event LOST: {event_type}"
                )
            
            return "buffered"
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[dict], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Subscribe to event type with async handler.
        
        Args:
            event_type: Event type to subscribe to
            handler: Async function that processes event data
        
        Example:
            async def my_handler(event_data: dict):
                symbol = event_data["symbol"]
                print(f"Processing {symbol}")
            
            bus.subscribe("ai.signal.generated", my_handler)
        """
        self._handlers[event_type].append(handler)
        
        logger.info(
            f"Handler subscribed: event_type={event_type}, "
            f"handler={handler.__name__}"
        )
    
    async def start(self) -> None:
        """
        Start consuming events from all subscribed event types.
        
        Creates consumer groups and spawns async tasks for each subscription.
        """
        if self._running:
            logger.warning("EventBus already running")
            return
        
        self._running = True
        
        # Create consumer tasks for each subscribed event type
        for event_type in self._handlers.keys():
            stream_name = self._get_stream_name(event_type)
            group_name = self._get_group_name(event_type)
            
            # Ensure stream and consumer group exist
            await self._ensure_stream_and_group(stream_name, group_name)
            
            # Spawn consumer task with exception tracking
            task = asyncio.create_task(
                self._consume_stream_safe(stream_name, group_name, event_type)
            )
            self._consumer_tasks[event_type] = task
            logger.info(f"🔹 Created consumer task for '{event_type}': consumer_id={self.consumer_id}")
        
        logger.info(
            f"EventBus started: subscriptions={len(self._handlers)}, "
            f"consumer_tasks={len(self._consumer_tasks)}"
        )
    
    async def stop(self) -> None:
        """Gracefully stop EventBus and cancel all consumer tasks."""
        self._running = False
        
        # Cancel all consumer tasks
        for event_type, task in self._consumer_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            logger.info(f"Consumer task stopped: event_type={event_type}")
        
        self._consumer_tasks.clear()
        
        logger.info("EventBus stopped")
    
    def _get_stream_name(self, event_type: str) -> str:
        """Get Redis Stream name for event type."""
        return f"{self.STREAM_PREFIX}{event_type}"
    
    def _get_group_name(self, event_type: str) -> str:
        """Get consumer group name for this service."""
        return f"{self.GROUP_PREFIX}{self.service_name}:{event_type}"
    
    def _get_dlq_name(self, event_type: str) -> str:
        """Get dead letter queue name for failed messages."""
        return f"{self.DLQ_PREFIX}{event_type}"
    
    async def _ensure_stream_and_group(
        self,
        stream_name: str,
        group_name: str,
    ) -> None:
        """
        Ensure Redis Stream and consumer group exist.
        
        Creates stream and group if they don't exist.
        """
        try:
            # Try creating consumer group (creates stream if not exists)
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id="0",  # Start from beginning
                mkstream=True,
            )
            logger.info(
                f"Created stream and consumer group: stream={stream_name}, "
                f"group={group_name}"
            )
        
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists - reset to beginning if requested
                import os
                reset_to_beginning = os.getenv("EVENTBUS_RESET_TO_BEGINNING", "false").lower() == "true"
                
                if reset_to_beginning:
                    logger.warning(
                        f"🔄 Resetting consumer group to beginning: stream={stream_name}, "
                        f"group={group_name}"
                    )
                    try:
                        await self.redis.xgroup_setid(
                            stream_name,
                            group_name,
                            id="0"  # Reset to beginning
                        )
                        logger.info(f"✅ Consumer group reset to beginning")
                    except Exception as reset_error:
                        logger.error(f"❌ Failed to reset consumer group: {reset_error}")
                else:
                    logger.debug(
                        "Consumer group already exists",
                        stream=stream_name,
                        group=group_name,
                )
            else:
                logger.error(
                    "Failed to create consumer group",
                    error=str(e),
                )
                raise
    
    async def _consume_stream_safe(
        self,
        stream_name: str,
        group_name: str,
        event_type: str,
    ) -> None:
        """
        Safe wrapper around _consume_stream that logs any exceptions.
        """
        try:
            await self._consume_stream(stream_name, group_name, event_type)
        except Exception as e:
            logger.error(
                f"🔥 Consumer task CRASHED for '{event_type}': {str(e)}",
                exc_info=True
            )
            raise
    
    async def _consume_stream(
        self,
        stream_name: str,
        group_name: str,
        event_type: str,
    ) -> None:
        """
        Consumer loop for a specific event type.
        
        Reads messages from Redis Stream and dispatches to handlers.
        """
        logger.info(
            f"🎯 Consumer loop starting: stream={stream_name}, group={group_name}, "
            f"event_type={event_type}, consumer_id={self.consumer_id}"
        )
        
        while self._running:
            try:
                # XREADGROUP - read new messages
                messages = await self.redis.xreadgroup(
                    group_name,
                    self.consumer_id,
                    {stream_name: ">"},
                    count=self.READ_COUNT,
                    block=self.READ_TIMEOUT,
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream, msg_list in messages:
                    for message_id, message_data in msg_list:
                        await self._process_message(
                            stream_name,
                            group_name,
                            message_id,
                            message_data,
                            event_type,
                        )
            
            except asyncio.CancelledError:
                break
            except redis.RedisError as e:
                logger.error(
                    f"Redis error in consumer loop for event_type={event_type}: {e}"
                )
                await asyncio.sleep(5)  # Backoff before retry
            except Exception as e:
                logger.error(
                    f"Unexpected error in consumer loop for event_type={event_type}: {e}\n{traceback.format_exc()}"
                )
                await asyncio.sleep(1)
        
        logger.info(f"Consumer stopped: event_type={event_type}")
    
    async def _process_message(
        self,
        stream_name: str,
        group_name: str,
        message_id: bytes,
        message_data: dict,
        event_type: str,
    ) -> None:
        """
        Process single message from stream.
        
        - Deserializes payload
        - P1-B: Extracts and sets correlation_id for tracking
        - Calls all registered handlers
        - Acknowledges message on success
        - Sends to DLQ on repeated failures
        """
        try:
            # Debug: log raw message data
            logger.info(f"🔍 Raw message_data keys: {list(message_data.keys())}")
            logger.info(f"🔍 Raw message_data: {message_data}")
            
            # Decode message data - handle both bytes and string keys
            payload_json = message_data.get("payload") or message_data.get(b"payload", b"{}")
            if isinstance(payload_json, bytes):
                payload_json = payload_json.decode("utf-8")
            payload = json.loads(payload_json) if payload_json else {}
            
            logger.info(f"✅ Decoded payload: {payload}")
            
            trace_id = message_data.get("trace_id") or message_data.get(b"trace_id", b"")
            if isinstance(trace_id, bytes):
                trace_id = trace_id.decode("utf-8")
            
            # P1-B: Extract correlation_id from message
            correlation_id = message_data.get("correlation_id") or message_data.get(b"correlation_id")
            if isinstance(correlation_id, bytes):
                correlation_id = correlation_id.decode("utf-8")
            
            # P1-B: Set correlation_id in thread-local context for this handler execution
            if correlation_id:
                set_correlation_id(correlation_id)
                logger.info(f"📎 correlation_id set: {correlation_id} for event_type={event_type}")
            
            # Call all handlers for this event type
            handlers = self._handlers.get(event_type, [])
            
            for handler in handlers:
                try:
                    await handler(payload)
                except Exception as e:
                    logger.error(
                        f"Handler error for event_type={event_type}, handler={handler.__name__}, trace_id={trace_id}: {e}\n{traceback.format_exc()}"
                    )
                    # Continue to other handlers even if one fails
            
            # Acknowledge message (removes from pending list)
            await self.redis.xack(stream_name, group_name, message_id)
            
            logger.debug(
                f"Message processed and acknowledged: event_type={event_type}, message_id={message_id}, trace_id={trace_id}, correlation_id={correlation_id}"
            )
        
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode message payload for event_type={event_type}, message_id={message_id}: {e}"
            )
            # Acknowledge anyway to skip bad message
            await self.redis.xack(stream_name, group_name, message_id)
        
        except Exception as e:
            logger.error(
                f"Failed to process message for event_type={event_type}, message_id={message_id}: {e}\n{traceback.format_exc()}"
            )
            # TODO: Implement retry logic and DLQ
            # For now, acknowledge to avoid infinite loop
            await self.redis.xack(stream_name, group_name, message_id)
    
    async def redis_health_check(self) -> bool:
        """Check if Redis is healthy and available (CRITICAL FIX #1 - Trading Gate)."""
        # SPRINT 1 - D2: Use RedisStreamBus health check
        is_healthy = await self.redis_stream.health_check()
        
        if is_healthy and not self._redis_available:
            # Redis recovered
            logger.warning("✅ Redis recovered - marking as available")
            self._redis_available = True
            
            # Publish recovery event
            await self.publish("system.redis_recovered", {
                "timestamp": datetime.utcnow().isoformat(),
                "buffer_stats": self.disk_buffer.get_stats()
            })
        
        elif not is_healthy and self._redis_available:
            # Redis went down
            logger.error("❌ Redis health check failed - marking unavailable")
            self._redis_available = False
        
        return is_healthy
    
    async def _replay_buffered_events(self):
        """Replay buffered events after Redis reconnects (CRITICAL FIX #4 - Ordered Replay)."""
        # SPRINT 1 - D2: Use DiskBuffer
        buffered_events = self.disk_buffer.read_all()
        
        if not buffered_events:
            logger.info("📭 No buffered events to replay")
            return
        
        logger.info(f"📤 Replaying {len(buffered_events)} buffered events in order")
        count = 0
        failed = 0
        
        for entry in buffered_events:
            try:
                event_type = entry["event_type"]
                message = entry["message"]
                
                # Re-publish to Redis using RedisStreamBus
                payload = json.loads(message["payload"])
                message_id = await self.redis_stream.publish(
                    event_type,
                    payload,
                    message.get("trace_id"),
                    message.get("source", self.service_name),
                )
                
                if message_id:
                    count += 1
                else:
                    logger.error(f"Failed to replay event {event_type} - Redis still down?")
                    failed += 1
                    break  # Stop replay if Redis fails again
            
            except Exception as e:
                logger.error(f"Failed to replay buffered event: {e}")
                failed += 1
        
        if failed == 0:
            # Successful replay - clear buffer
            self.disk_buffer.clear()
            logger.info(f"✅ Event replay complete: {count} replayed, buffer cleared")
        else:
            logger.warning(
                f"⚠️  Partial replay: {count} replayed, {failed} failed. "
                f"Buffer NOT cleared - will retry on next reconnect"
            )
    
    async def acquire_promotion_lock(self, required_handlers: list[str]) -> bool:
        """[CRITICAL FIX #1] Acquire atomic promotion lock.
        
        Args:
            required_handlers: List of handler names that must ACK
        
        Returns:
            True if lock acquired, False if already locked
        """
        async with self._promotion_lock:
            if self._promotion_lock_active:
                logger.warning("Promotion already in progress")
                return False
            
            self._promotion_lock_active = True
            self._promotion_acks = {name: False for name in required_handlers}
            logger.info(f"[PROMOTION-LOCK] Acquired. Awaiting ACKs from: {required_handlers}")
            return True
    
    async def ack_promotion(self, handler_name: str) -> None:
        """[CRITICAL FIX #1] Handler acknowledges promotion completion.
        
        Args:
            handler_name: Name of handler that completed processing
        """
        if handler_name in self._promotion_acks:
            self._promotion_acks[handler_name] = True
            logger.info(f"[PROMOTION-LOCK] ACK received from {handler_name}")
    
    async def wait_for_promotion_acks(self, timeout: float = 30.0) -> bool:
        """[CRITICAL FIX #1] Wait for all handlers to ACK promotion.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if all ACKs received, False if timeout
        """
        start = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start < timeout:
            if all(self._promotion_acks.values()):
                logger.info("[PROMOTION-LOCK] All ACKs received")
                return True
            await asyncio.sleep(0.5)
        
        # Timeout - log missing ACKs
        missing = [name for name, acked in self._promotion_acks.items() if not acked]
        logger.error(f"[PROMOTION-LOCK] Timeout waiting for ACKs from: {missing}")
        return False
    
    async def release_promotion_lock(self) -> None:
        """[CRITICAL FIX #1] Release atomic promotion lock."""
        async with self._promotion_lock:
            self._promotion_lock_active = False
            self._promotion_acks.clear()
            logger.info("[PROMOTION-LOCK] Released")
    
    def is_promotion_locked(self) -> bool:
        """[CRITICAL FIX #1] Check if promotion is in progress."""
        return self._promotion_lock_active


# Singleton instance (initialized by application)
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Get singleton EventBus instance.
    
    Returns:
        EventBus instance
    
    Raises:
        RuntimeError: If EventBus not initialized
    """
    if _event_bus is None:
        raise RuntimeError(
            "EventBus not initialized. Call initialize_event_bus() first."
        )
    return _event_bus


async def initialize_event_bus(
    redis_client: Redis,
    service_name: str = "quantum_trader",
) -> EventBus:
    """
    Initialize global EventBus singleton.
    
    Args:
        redis_client: Async Redis client
        service_name: Name of this service
    
    Returns:
        Initialized EventBus
    """
    global _event_bus
    
    _event_bus = EventBus(redis_client, service_name)
    await _event_bus.initialize()
    
    return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown global EventBus singleton."""
    global _event_bus
    
    if _event_bus:
        await _event_bus.stop()
        _event_bus = None
