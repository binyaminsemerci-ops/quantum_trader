"""
EventBus - Internal Asynchronous Messaging Backbone for Quantum Trader

Provides decoupled, pub-sub communication between all system modules:
- MSC AI publishes policy updates
- Strategy Generator publishes promotions/demotions
- Health Monitor publishes health alerts
- CLM publishes model updates
- Subscribers: Logger, Discord notifier, metrics collector, etc.

Core Design:
- In-memory, asyncio-based
- Supports async and sync handlers
- Error isolation (handler failures don't crash the bus)
- Extensible to external brokers (Kafka, RabbitMQ) later

Author: Quantum Trader AI Team
Date: 2025-11-30
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable, Protocol
from concurrent.futures import ThreadPoolExecutor
import inspect

logger = logging.getLogger(__name__)


# ============================================================================
# Event Model
# ============================================================================

@dataclass
class Event:
    """
    Base event class for all EventBus messages.
    
    Attributes:
        type: Event type identifier (e.g., "policy.updated", "strategy.promoted")
        timestamp: When the event occurred (defaults to current UTC time)
        payload: Event-specific data
        source: Optional source module identifier
        correlation_id: Optional ID for tracing related events
    """
    type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    correlation_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "source": self.source,
            "correlation_id": self.correlation_id,
        }


# ============================================================================
# Specialized Event Types
# ============================================================================

@dataclass
class PolicyUpdatedEvent(Event):
    """
    Published by MSC AI when global policy changes.
    
    Payload fields:
        - risk_mode: str (AGGRESSIVE/NORMAL/DEFENSIVE)
        - max_risk_per_trade: float
        - max_positions: int
        - global_min_confidence: float
        - changes: dict (what changed)
    """
    type: str = field(default="policy.updated")
    
    @classmethod
    def create(
        cls,
        risk_mode: str | None = None,
        max_risk_per_trade: float | None = None,
        max_positions: int | None = None,
        global_min_confidence: float | None = None,
        changes: dict | None = None,
        source: str = "msc_ai"
    ) -> "PolicyUpdatedEvent":
        """Factory method for creating policy update events."""
        payload = {}
        if risk_mode is not None:
            payload["risk_mode"] = risk_mode
        if max_risk_per_trade is not None:
            payload["max_risk_per_trade"] = max_risk_per_trade
        if max_positions is not None:
            payload["max_positions"] = max_positions
        if global_min_confidence is not None:
            payload["global_min_confidence"] = global_min_confidence
        if changes is not None:
            payload["changes"] = changes
        
        return cls(
            payload=payload,
            source=source
        )


@dataclass
class StrategyPromotedEvent(Event):
    """
    Published by Strategy Generator when a strategy changes state.
    
    Payload fields:
        - strategy_id: str
        - strategy_name: str
        - from_state: str (SHADOW/PAUSED)
        - to_state: str (LIVE/SHADOW/PAUSED)
        - performance_score: float
        - reason: str
    """
    type: str = field(default="strategy.promoted")
    
    @classmethod
    def create(
        cls,
        strategy_id: str,
        strategy_name: str,
        from_state: str,
        to_state: str,
        performance_score: float | None = None,
        reason: str | None = None,
        source: str = "strategy_generator"
    ) -> "StrategyPromotedEvent":
        """Factory method for strategy promotion events."""
        return cls(
            payload={
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "from_state": from_state,
                "to_state": to_state,
                "performance_score": performance_score,
                "reason": reason,
            },
            source=source
        )


@dataclass
class ModelPromotedEvent(Event):
    """
    Published by Continuous Learning Manager when a model version changes.
    
    Payload fields:
        - model_name: str (xgboost, lightgbm, nhits, patchtst)
        - old_version: str
        - new_version: str
        - improvement_pct: float
        - shadow_test_results: dict
    """
    type: str = field(default="model.promoted")
    
    @classmethod
    def create(
        cls,
        model_name: str,
        old_version: str,
        new_version: str,
        improvement_pct: float | None = None,
        shadow_test_results: dict | None = None,
        source: str = "continuous_learning"
    ) -> "ModelPromotedEvent":
        """Factory method for model promotion events."""
        return cls(
            payload={
                "model_name": model_name,
                "old_version": old_version,
                "new_version": new_version,
                "improvement_pct": improvement_pct,
                "shadow_test_results": shadow_test_results or {},
            },
            source=source
        )


@dataclass
class HealthStatusChangedEvent(Event):
    """
    Published by System Health Monitor when health state changes.
    
    Payload fields:
        - status: str (HEALTHY/WARNING/CRITICAL)
        - previous_status: str
        - failed_modules: list[str]
        - warning_modules: list[str]
        - details: dict
    """
    type: str = field(default="health.status_changed")
    
    @classmethod
    def create(
        cls,
        status: str,
        previous_status: str,
        failed_modules: list[str] | None = None,
        warning_modules: list[str] | None = None,
        details: dict | None = None,
        source: str = "system_health_monitor"
    ) -> "HealthStatusChangedEvent":
        """Factory method for health status change events."""
        return cls(
            payload={
                "status": status,
                "previous_status": previous_status,
                "failed_modules": failed_modules or [],
                "warning_modules": warning_modules or [],
                "details": details or {},
            },
            source=source
        )


@dataclass
class OpportunitiesUpdatedEvent(Event):
    """
    Published by Opportunity Ranker when rankings are recalculated.
    
    Payload fields:
        - top_symbols: list[str]
        - rankings: dict[str, float]
        - num_symbols_scored: int
        - top_score: float
    """
    type: str = field(default="opportunities.updated")
    
    @classmethod
    def create(
        cls,
        top_symbols: list[str],
        rankings: dict[str, float],
        num_symbols_scored: int | None = None,
        top_score: float | None = None,
        source: str = "opportunity_ranker"
    ) -> "OpportunitiesUpdatedEvent":
        """Factory method for opportunity ranking updates."""
        return cls(
            payload={
                "top_symbols": top_symbols,
                "rankings": rankings,
                "num_symbols_scored": num_symbols_scored or len(rankings),
                "top_score": top_score,
            },
            source=source
        )


@dataclass
class TradeExecutedEvent(Event):
    """
    Published by Executor when a trade is filled.
    
    Payload fields:
        - symbol: str
        - side: str (BUY/SELL)
        - quantity: float
        - price: float
        - order_id: str
        - strategy_id: str
        - pnl: float (if closing position)
    """
    type: str = field(default="trade.executed")
    
    @classmethod
    def create(
        cls,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
        strategy_id: str | None = None,
        pnl: float | None = None,
        source: str = "executor"
    ) -> "TradeExecutedEvent":
        """Factory method for trade execution events."""
        return cls(
            payload={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
                "strategy_id": strategy_id,
                "pnl": pnl,
            },
            source=source
        )


# ============================================================================
# EventBus Protocol and Types
# ============================================================================

# Handler can be async or sync function taking an Event
EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]


class EventBus(Protocol):
    """
    Protocol defining the EventBus interface.
    
    Implementations must provide:
    - publish: Non-blocking event publishing
    - subscribe: Register handlers for event types
    - run_forever: Continuous event processing loop
    """
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
        """
        ...
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: Event type to listen for (e.g., "policy.updated")
            handler: Async or sync function to call when event occurs
        """
        ...
    
    async def run_forever(self) -> None:
        """
        Run the event processing loop continuously.
        
        Consumes events from the queue and dispatches to registered handlers.
        Handles errors gracefully without stopping the loop.
        """
        ...


# ============================================================================
# InMemoryEventBus Implementation
# ============================================================================

class InMemoryEventBus:
    """
    In-memory, asyncio-based EventBus implementation.
    
    Features:
    - Async/sync handler support
    - Multiple handlers per event type
    - Error isolation (handler failures don't crash bus)
    - Non-blocking publish
    - Background event processing
    
    Usage:
        bus = InMemoryEventBus()
        bus.subscribe("policy.updated", my_handler)
        await bus.publish(PolicyUpdatedEvent.create(...))
        asyncio.create_task(bus.run_forever())
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize EventBus.
        
        Args:
            max_queue_size: Maximum number of queued events before blocking
        """
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="eventbus-sync-")
        self._running = False
        self._stats = {
            "published": 0,
            "processed": 0,
            "errors": 0,
        }
        
        logger.info("[EventBus] Initialized (max_queue_size=%d)", max_queue_size)
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus (non-blocking).
        
        Args:
            event: Event to publish
        """
        try:
            await self._queue.put(event)
            self._stats["published"] += 1
            logger.debug(
                "[EventBus] Published: type=%s source=%s queue_size=%d",
                event.type,
                event.source,
                self._queue.qsize()
            )
        except asyncio.QueueFull:
            logger.error(
                "[EventBus] ❌ Queue full! Dropping event: type=%s",
                event.type
            )
            raise
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: Event type to listen for
            handler: Async or sync callable
        """
        self._handlers[event_type].append(handler)
        handler_name = getattr(handler, "__name__", str(handler))
        logger.info(
            "[EventBus] ✅ Subscribed: type=%s handler=%s (total=%d)",
            event_type,
            handler_name,
            len(self._handlers[event_type])
        )
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: Event type
            handler: Handler to remove
            
        Returns:
            True if handler was removed, False if not found
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.info(
                    "[EventBus] 🔻 Unsubscribed: type=%s handler=%s",
                    event_type,
                    getattr(handler, "__name__", str(handler))
                )
                return True
            except ValueError:
                pass
        return False
    
    async def run_forever(self) -> None:
        """
        Run the event processing loop continuously.
        
        Consumes events from queue and dispatches to handlers.
        Handles errors gracefully without stopping the loop.
        """
        self._running = True
        logger.info("[EventBus] 🚀 Starting event processing loop...")
        
        while self._running:
            try:
                # Wait for next event
                event = await self._queue.get()
                
                # Get handlers for this event type
                handlers = self._handlers.get(event.type, [])
                
                if not handlers:
                    logger.debug(
                        "[EventBus] No handlers for event type: %s",
                        event.type
                    )
                    self._stats["processed"] += 1
                    self._queue.task_done()
                    continue
                
                logger.debug(
                    "[EventBus] Processing: type=%s handlers=%d",
                    event.type,
                    len(handlers)
                )
                
                # Dispatch to all handlers
                await self._dispatch_to_handlers(event, handlers)
                
                self._stats["processed"] += 1
                self._queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("[EventBus] Event loop cancelled")
                break
            except Exception as e:
                logger.error(
                    "[EventBus] ❌ Unexpected error in event loop: %s",
                    e,
                    exc_info=True
                )
                self._stats["errors"] += 1
                # Don't break - continue processing
        
        logger.info("[EventBus] 🛑 Event processing loop stopped")
    
    async def _dispatch_to_handlers(
        self,
        event: Event,
        handlers: list[EventHandler]
    ) -> None:
        """
        Dispatch event to all registered handlers.
        
        Args:
            event: Event to dispatch
            handlers: List of handlers to call
        """
        for handler in handlers:
            try:
                # Check if handler is async or sync
                if inspect.iscoroutinefunction(handler):
                    # Async handler - await directly
                    await handler(event)
                else:
                    # Sync handler - run in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self._executor, handler, event)
                
                logger.debug(
                    "[EventBus] ✅ Handler executed: %s",
                    getattr(handler, "__name__", str(handler))
                )
                
            except Exception as e:
                # Isolate handler errors - don't crash the bus
                handler_name = getattr(handler, "__name__", str(handler))
                logger.error(
                    "[EventBus] ❌ Handler error: handler=%s event=%s error=%s",
                    handler_name,
                    event.type,
                    e,
                    exc_info=True
                )
                self._stats["errors"] += 1
    
    def stop(self) -> None:
        """Stop the event processing loop."""
        logger.info("[EventBus] Stopping event loop...")
        self._running = False
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get EventBus statistics.
        
        Returns:
            Dictionary with published, processed, error counts and queue size
        """
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "event_types_registered": len(self._handlers),
            "total_handlers": sum(len(handlers) for handlers in self._handlers.values()),
        }
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the EventBus.
        
        Waits for queue to be empty before stopping.
        """
        logger.info("[EventBus] Shutting down (waiting for queue to empty)...")
        self.stop()
        
        # Wait for queue to be processed
        await self._queue.join()
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        logger.info("[EventBus] ✅ Shutdown complete")


# ============================================================================
# Helper Functions
# ============================================================================

def create_event_bus(max_queue_size: int = 10000) -> InMemoryEventBus:
    """
    Factory function to create an EventBus instance.
    
    Args:
        max_queue_size: Maximum queue size
        
    Returns:
        Configured InMemoryEventBus
    """
    return InMemoryEventBus(max_queue_size=max_queue_size)
