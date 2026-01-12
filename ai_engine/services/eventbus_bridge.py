#!/usr/bin/env python3
"""
EventBus Bridge - Redis Streams Communication Backbone
======================================================
Handles pub/sub for Quantum Trader v5 core execution loop.

Topics:
- trade.signal.v5 - AI Engine signals
- trade.signal.safe - Risk-approved signals
- trade.execution.res - Execution results
- trade.position.update - Position updates

Author: Quantum Trader Team
Date: 2026-01-12
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


# ============================================================================
# MESSAGE SCHEMAS
# ============================================================================

@dataclass
class TradeSignal:
    """AI Engine trade signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    timestamp: str
    source: str  # "meta_v5", "xgb_v5", etc.
    meta_override: bool = False
    ensemble_votes: Optional[Dict[str, float]] = None


@dataclass
class RiskApprovedSignal:
    """Risk-approved trade signal"""
    symbol: str
    action: str
    confidence: float
    position_size_usd: float
    position_size_pct: float
    risk_amount_usd: float
    kelly_optimal: float
    timestamp: str
    source: str
    approved_by: str = "risk_safety"


@dataclass
class ExecutionResult:
    """Trade execution result"""
    symbol: str
    action: str
    entry_price: float
    position_size_usd: float
    leverage: float
    timestamp: str
    order_id: str
    status: str  # "filled", "partial", "rejected"
    slippage_pct: float = 0.0
    fee_usd: float = 0.0


@dataclass
class PositionUpdate:
    """Position state update"""
    symbol: str
    side: str  # "LONG", "SHORT", "CLOSED"
    entry_price: float
    current_price: float
    size_usd: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: str


# ============================================================================
# EVENTBUS CLIENT
# ============================================================================

class EventBusClient:
    """
    Redis Streams client for Quantum Trader v5
    
    Usage:
        async with EventBusClient() as bus:
            await bus.publish("trade.signal.v5", signal)
            async for msg in bus.subscribe("trade.execution.res"):
                print(msg)
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_stream_length: int = 10000
    ):
        self.redis_url = redis_url
        self.max_stream_length = max_stream_length
        self.redis: Optional[aioredis.Redis] = None
        self._subscriptions: Dict[str, asyncio.Task] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            logger.info(f"✅ Connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        # Cancel all subscriptions
        for topic, task in self._subscriptions.items():
            task.cancel()
            logger.info(f"🛑 Cancelled subscription: {topic}")
        
        if self.redis:
            await self.redis.close()
            logger.info("✅ Redis connection closed")
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        maxlen: Optional[int] = None
    ) -> str:
        """
        Publish message to Redis stream
        
        Args:
            topic: Stream name (e.g., "trade.signal.v5")
            message: Message payload (will be JSON-serialized)
            maxlen: Max stream length (uses default if None)
        
        Returns:
            Message ID
        """
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        # Ensure timestamp
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Serialize to JSON string
        payload = {"data": json.dumps(message)}
        
        # Publish with MAXLEN to prevent unbounded growth
        maxlen_val = maxlen or self.max_stream_length
        message_id = await self.redis.xadd(
            topic,
            payload,
            maxlen=maxlen_val,
            approximate=True
        )
        
        logger.debug(f"📤 Published to {topic}: {message_id}")
        return message_id
    
    async def publish_signal(self, signal: TradeSignal):
        """Publish trade signal to trade.signal.v5"""
        return await self.publish("trade.signal.v5", asdict(signal))
    
    async def publish_approved(self, signal: RiskApprovedSignal):
        """Publish risk-approved signal to trade.signal.safe"""
        return await self.publish("trade.signal.safe", asdict(signal))
    
    async def publish_execution(self, result: ExecutionResult):
        """Publish execution result to trade.execution.res"""
        return await self.publish("trade.execution.res", asdict(result))
    
    async def publish_position(self, update: PositionUpdate):
        """Publish position update to trade.position.update"""
        return await self.publish("trade.position.update", asdict(update))
    
    async def subscribe(
        self,
        topic: str,
        start_id: str = "$",
        block_ms: int = 1000
    ):
        """
        Subscribe to Redis stream
        
        Args:
            topic: Stream name
            start_id: Starting message ID ("$" = only new, "0" = from beginning)
            block_ms: Block timeout in milliseconds
        
        Yields:
            Parsed message dictionaries
        """
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        logger.info(f"📥 Subscribing to {topic} from {start_id}")
        last_id = start_id
        
        while True:
            try:
                # Read from stream
                messages = await self.redis.xread(
                    {topic: last_id},
                    count=10,
                    block=block_ms
                )
                
                if not messages:
                    # No new messages, continue blocking
                    continue
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        # Parse JSON payload
                        if "data" in fields:
                            try:
                                payload = json.loads(fields["data"])
                                payload["_message_id"] = message_id
                                yield payload
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON decode error: {e}")
                        
                        # Update last_id
                        last_id = message_id
            
            except asyncio.CancelledError:
                logger.info(f"🛑 Subscription cancelled: {topic}")
                break
            except Exception as e:
                logger.error(f"❌ Subscribe error on {topic}: {e}")
                await asyncio.sleep(5)  # Retry delay
    
    async def get_stream_length(self, topic: str) -> int:
        """Get number of messages in stream"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        return await self.redis.xlen(topic)
    
    async def get_stream_info(self, topic: str) -> Dict[str, Any]:
        """Get stream metadata"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        try:
            info = await self.redis.xinfo_stream(topic)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0)
            }
        except Exception as e:
            logger.warning(f"⚠️ Could not get stream info for {topic}: {e}")
            return {}
    
    async def trim_stream(self, topic: str, maxlen: int = 1000):
        """Trim stream to max length"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        await self.redis.xtrim(topic, maxlen=maxlen, approximate=True)
        logger.info(f"✂️ Trimmed {topic} to {maxlen} messages")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def publish_trade_signal(
    symbol: str,
    action: str,
    confidence: float,
    source: str = "ai_engine_v5",
    redis_url: str = "redis://localhost:6379"
):
    """
    Quick publish of trade signal
    
    Example:
        await publish_trade_signal("BTCUSDT", "BUY", 0.85)
    """
    signal = TradeSignal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=datetime.utcnow().isoformat() + "Z",
        source=source
    )
    
    async with EventBusClient(redis_url) as bus:
        await bus.publish_signal(signal)
        logger.info(f"✅ Published signal: {symbol} {action} @ {confidence:.2f}")


async def get_recent_signals(
    topic: str = "trade.signal.v5",
    count: int = 10,
    redis_url: str = "redis://localhost:6379"
) -> List[Dict[str, Any]]:
    """
    Get recent messages from stream
    
    Returns:
        List of parsed messages
    """
    async with EventBusClient(redis_url) as bus:
        if not bus.redis:
            return []
        
        # Read last N messages
        messages = await bus.redis.xrevrange(topic, count=count)
        
        results = []
        for message_id, fields in messages:
            if "data" in fields:
                try:
                    payload = json.loads(fields["data"])
                    payload["_message_id"] = message_id
                    results.append(payload)
                except json.JSONDecodeError:
                    pass
        
        return results


# ============================================================================
# MAIN - TESTING
# ============================================================================

async def main():
    """Test EventBus functionality"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    async with EventBusClient() as bus:
        # Test publish
        test_signal = TradeSignal(
            symbol="BTCUSDT",
            action="BUY",
            confidence=0.85,
            timestamp=datetime.utcnow().isoformat() + "Z",
            source="test"
        )
        
        await bus.publish_signal(test_signal)
        logger.info("✅ Published test signal")
        
        # Test stream info
        info = await bus.get_stream_info("trade.signal.v5")
        logger.info(f"📊 Stream info: {info}")
        
        # Test subscribe (read 1 message then exit)
        logger.info("📥 Testing subscription...")
        count = 0
        async for msg in bus.subscribe("trade.signal.v5", start_id="0"):
            logger.info(f"📬 Received: {msg}")
            count += 1
            if count >= 3:
                break


if __name__ == "__main__":
    asyncio.run(main())
