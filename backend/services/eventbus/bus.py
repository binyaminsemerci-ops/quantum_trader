"""
EventBus implementation for Quantum Trader.

Provides async pub/sub messaging with in-memory queue-based dispatch.
"""

import asyncio
import logging
from typing import Protocol, Callable, Awaitable, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .events import Event

logger = logging.getLogger(__name__)

# Type alias for event handlers (can be sync or async)
EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]


class EventBus(Protocol):
    """
    Protocol defining the EventBus interface.
    
    The EventBus provides decoupled pub/sub messaging for system components.
    """
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: The event to publish
        """
        ...
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe a handler to a specific event type.
        
        Args:
            event_type: The event type to listen for (e.g. "policy.updated")
            handler: Async or sync callable that processes the event
        """
        ...
    
    async def run_forever(self) -> None:
        """
        Continuously consume events from the queue and dispatch to handlers.
        
        This should run in a background task and never return under normal conditions.
        """
        ...


class InMemoryEventBus(EventBus):
    """
    In-memory asyncio-based EventBus implementation.
    
    Features:
        - Async queue for event buffering
        - Multiple handlers per event type
        - Support for both async and sync handlers
        - Robust error handling (handler failures don't crash the bus)
        - At-least-once delivery (best-effort)
    
    Example:
        >>> bus = InMemoryEventBus()
        >>> bus.subscribe("test.event", lambda e: print(e.payload))
        >>> await bus.publish(Event("test.event", datetime.utcnow(), {"msg": "hi"}))
        >>> asyncio.create_task(bus.run_forever())
    """
    
    def __init__(self, max_queue_size: int = 10000, max_workers: int = 4):
        """
        Initialize the EventBus.
        
        Args:
            max_queue_size: Maximum events in the queue before publish blocks
            max_workers: Thread pool size for sync handlers
        """
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._stats = {
            "published": 0,
            "dispatched": 0,
            "errors": 0,
        }
        logger.info(f"EventBus initialized with queue_size={max_queue_size}, workers={max_workers}")
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        
        The event is placed in an internal queue and will be dispatched
        asynchronously by the run_forever loop.
        
        Args:
            event: The event to publish
            
        Raises:
            asyncio.QueueFull: If the queue is at capacity (should be rare)
        """
        try:
            await self._queue.put(event)
            self._stats["published"] += 1
            logger.debug(f"Published event: {event.type} (queue_size={self._queue.qsize()})")
        except asyncio.QueueFull:
            logger.error(f"EventBus queue full! Dropping event: {event.type}")
            raise
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe a handler to a specific event type.
        
        Multiple handlers can subscribe to the same event type.
        Handlers can be async or sync functions.
        
        Args:
            event_type: The event type to listen for (e.g. "policy.updated")
            handler: Callable that processes the event
        """
        self._handlers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type} (total={len(self._handlers[event_type])})")
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe a specific handler from an event type.
        
        Args:
            event_type: The event type
            handler: The handler to remove
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.info(f"Unsubscribed handler from {event_type}")
    
    async def run_forever(self) -> None:
        """
        Continuously consume events and dispatch to handlers.
        
        This method runs an infinite loop that:
        1. Waits for events from the queue
        2. Looks up registered handlers for that event type
        3. Dispatches to all handlers (async or sync)
        4. Logs errors but continues processing
        
        Should be run as a background task:
            asyncio.create_task(bus.run_forever())
        """
        self._running = True
        logger.info("EventBus started, waiting for events...")
        
        while self._running:
            try:
                # Wait for next event
                event = await self._queue.get()
                
                # Dispatch to all registered handlers
                await self._dispatch_event(event)
                
                self._stats["dispatched"] += 1
                self._queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("EventBus run_forever cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in EventBus loop: {e}", exc_info=True)
                self._stats["errors"] += 1
                # Continue processing other events
    
    async def _dispatch_event(self, event: Event) -> None:
        """
        Dispatch an event to all registered handlers.
        
        Args:
            event: The event to dispatch
        """
        handlers = self._handlers.get(event.type, [])
        
        if not handlers:
            logger.debug(f"No handlers for event type: {event.type}")
            return
        
        logger.debug(f"Dispatching {event.type} to {len(handlers)} handler(s)")
        
        # Fire all handlers concurrently
        tasks = [self._invoke_handler(handler, event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any handler errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Handler {i} for {event.type} raised: {result}",
                    exc_info=result
                )
                self._stats["errors"] += 1
    
    async def _invoke_handler(self, handler: EventHandler, event: Event) -> None:
        """
        Invoke a single handler (async or sync).
        
        Args:
            handler: The handler function
            event: The event to pass to the handler
            
        Raises:
            Exception: Re-raises handler exceptions for gather() to catch
        """
        if asyncio.iscoroutinefunction(handler):
            # Async handler - await directly
            await handler(event)
        else:
            # Sync handler - run in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, handler, event)
    
    def stop(self) -> None:
        """Stop the event loop gracefully."""
        logger.info("Stopping EventBus...")
        self._running = False
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get EventBus statistics.
        
        Returns:
            Dict with published, dispatched, errors counts and queue size
        """
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "handler_types": len(self._handlers),
            "total_handlers": sum(len(h) for h in self._handlers.values()),
        }
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        self._executor.shutdown(wait=False)
