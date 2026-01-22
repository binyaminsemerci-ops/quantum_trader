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
import os
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_trade_intent(payload: dict) -> list[str]:
    """
    Validate trade intent payload against schema contract v1.1 BRIDGE-PATCH.
    
    v1.1: Optional AI fields (ai_size_usd, ai_leverage, ai_harvest_policy) allowed.
    Extra fields allowed (forward compatibility).
    
    Returns list of error messages (empty if valid).
    See: TRADE_INTENT_SCHEMA_CONTRACT.md
    """
    errors = []
    
    # v1.1: Required core fields only (size/lev now optional via AI or defaults)
    required = ["symbol", "action", "confidence", "timestamp"]
    for field in required:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
    
    # Validation rules
    if "symbol" in payload:
        import re
        if not re.match(r'^[A-Z]{3,10}USDT$', payload["symbol"]):
            errors.append(f"Invalid symbol format: {payload['symbol']}")
    
    if "action" in payload:
        if payload["action"] not in ["BUY", "SELL", "CLOSE"]:
            errors.append(f"Invalid action: {payload['action']}")
    
    if "confidence" in payload:
        if not (0.0 <= payload["confidence"] <= 1.0):
            errors.append(f"Invalid confidence range: {payload['confidence']}")
    
    # v1.1: position_size_usd optional (can be injected by AI)
    if "position_size_usd" in payload and payload["position_size_usd"] is not None:
        if payload["position_size_usd"] <= 0:
            errors.append(f"Invalid position_size_usd: {payload['position_size_usd']}")
    
    # v1.1: leverage optional (can be injected by AI)
    if "leverage" in payload and payload["leverage"] is not None:
        if not (1 <= payload["leverage"] <= 125):
            errors.append(f"Invalid leverage range: {payload['leverage']}")
    
    # Deprecated field warning
    if "side" in payload and "action" not in payload:
        logger.warning("⚠️ Deprecated field 'side' used - use 'action' instead")
    
    return errors


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
class HarvestPolicy:
    """Exit/harvest policy (TP/SL control)"""
    mode: str  # 'scalper', 'swing', 'trend_runner'
    trail_pct: float = 1.0  # Trailing stop %
    max_time_sec: int = 3600  # Max hold time
    partial_close_pct: float = 0.0  # Partial close threshold
    
    def to_dict(self):
        return {
            'mode': self.mode,
            'trail_pct': self.trail_pct,
            'max_time_sec': self.max_time_sec,
            'partial_close_pct': self.partial_close_pct
        }


@dataclass
class TradeIntent:
    """Trade intent from strategy layer (for execution)
    
    v1.1 BRIDGE-PATCH: Supports AI-injected sizing/leverage/policy fields.
    """
    symbol: str
    action: str  # BUY, SELL, CLOSE
    confidence: float
    timestamp: str
    # Optional sizing (can come from AI or defaults)
    position_size_usd: Optional[float] = None
    leverage: Optional[float] = None
    # Optional AI-injected fields (before governor clamping)
    ai_size_usd: Optional[float] = None
    ai_leverage: Optional[float] = None
    ai_harvest_policy: Optional[dict] = None
    harvest_policy: Optional[HarvestPolicy] = None
    source: Optional[str] = None
    risk_budget_usd: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    
    def __post_init__(self):
        """Convert ai_harvest_policy dict to HarvestPolicy if needed"""
        if self.ai_harvest_policy and isinstance(self.ai_harvest_policy, dict):
            try:
                self.harvest_policy = HarvestPolicy(**self.ai_harvest_policy)
            except Exception as e:
                logger.warning(f"Failed to parse harvest_policy: {e}")
    
    @property
    def side(self):
        """Backwards compatibility alias for action"""
        return self.action
    
    def normalized(self):
        """Return normalized intent with leverage clamped to [5..80]"""
        result = {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }
        
        if self.ai_size_usd is not None:
            result['position_size_usd'] = self.ai_size_usd
        elif self.position_size_usd is not None:
            result['position_size_usd'] = self.position_size_usd
        
        if self.ai_leverage is not None:
            result['leverage'] = max(5, min(80, self.ai_leverage))
        elif self.leverage is not None:
            result['leverage'] = max(5, min(80, self.leverage))
        
        if self.source:
            result['source'] = self.source
        if self.risk_budget_usd is not None:
            result['risk_budget_usd'] = self.risk_budget_usd
        if self.entry_price is not None:
            result['entry_price'] = self.entry_price
        if self.stop_loss is not None:
            result['stop_loss'] = self.stop_loss
        if self.take_profit is not None:
            result['take_profit'] = self.take_profit
        
        if self.harvest_policy:
            result['harvest_policy'] = self.harvest_policy.to_dict()
        elif self.ai_harvest_policy:
            result['harvest_policy'] = self.ai_harvest_policy
        
        return result


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
        stream_name: str,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "unknown",
        maxlen: Optional[int] = None,
        validate_schema: bool = True
    ) -> str:
        """
        Publish message to Redis stream (Schema Contract v1.0)
        
        Args:
            stream_name: Stream name (e.g., "quantum:stream:trade.intent")
            event_type: Event type (e.g., "trade.intent")
            payload: Message payload (will be JSON-serialized)
            source: Publisher identifier
            maxlen: Max stream length (uses default if None)
            validate_schema: Enable schema validation (fail-closed)
        
        Returns:
            Message ID
        """
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        # Schema validation (fail-closed for trade.intent)
        if validate_schema and event_type == "trade.intent":
            errors = validate_trade_intent(payload)
            if errors:
                raise ValueError(f"Schema validation failed: {errors}")
        
        # Ensure timestamp
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Build Redis stream fields (Schema Contract v1.0)
        fields = {
            "event_type": event_type,
            "payload": json.dumps(payload),  # NOTE: "payload" not "data"
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": source
        }
        
        # Publish with MAXLEN to prevent unbounded growth
        maxlen_val = maxlen or self.max_stream_length
        message_id = await self.redis.xadd(
            stream_name,
            fields,
            maxlen=maxlen_val,
            approximate=True
        )
        
        logger.debug(f"📤 Published to {stream_name}: {message_id}")
        return message_id
    
    async def publish_signal(self, signal: TradeSignal):
        """Publish trade signal to trade.signal.v5"""
        return await self.publish(
            stream_name="quantum:stream:trade.signal.v5",
            event_type="trade.signal",
            payload=asdict(signal),
            source="ai-engine",
            validate_schema=False
        )
    
    async def publish_approved(self, signal: RiskApprovedSignal):
        """Publish risk-approved signal to trade.signal.safe"""
        return await self.publish(
            stream_name="quantum:stream:trade.signal.safe",
            event_type="trade.signal.safe",
            payload=asdict(signal),
            source="risk-manager",
            validate_schema=False
        )
    
    async def publish_execution(self, result: ExecutionResult):
        """Publish execution result (primary + optional legacy stream)."""
        primary_stream = os.getenv("EXECUTION_RESULT_STREAM", "quantum:stream:execution.result")
        legacy_stream = os.getenv("EXECUTION_RESULT_STREAM_LEGACY", "trade.execution.res")
        payload = asdict(result)
        
        msg_id = await self.publish(
            stream_name=primary_stream,
            event_type="execution.result",
            payload=payload,
            source="execution-service",
            validate_schema=False
        )
        
        if legacy_stream and legacy_stream != primary_stream:
            try:
                await self.publish(
                    stream_name=legacy_stream,
                    event_type="execution.result",
                    payload=payload,
                    source="execution-service",
                    validate_schema=False
                )
            except Exception as e:
                logger.warning(f"⚠️ Legacy execution publish failed: {e}")
        return msg_id
    
    async def publish_position(self, update: PositionUpdate):
        """Publish position update to trade.position.update"""
        return await self.publish(
            stream_name="quantum:stream:position.update",
            event_type="position.update",
            payload=asdict(update),
            source="execution-service",
            validate_schema=False
        )
    
    async def subscribe_with_group(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        start_id: str = ">",
        block_ms: int = 1000,
        create_group: bool = True
    ):
        """
        Subscribe to Redis stream using consumer group (P0 FIX: Phase 2)
        
        Consumer groups provide:
        - Message acknowledgment (no data loss on restart)
        - Replay capability via XPENDING
        - Load balancing across multiple consumers
        
        Args:
            topic: Stream name
            group_name: Consumer group name
            consumer_name: This consumer's unique name
            start_id: Starting message ID (">" = only new, "0" = from beginning)
            block_ms: Block timeout in milliseconds
            create_group: Create group if not exists
        
        Yields:
            Parsed message dictionaries with _message_id for ACK
        """
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
        
        # Create consumer group if requested
        if create_group:
            try:
                await self.redis.xgroup_create(
                    topic,
                    group_name,
                    id="0",  # Start from beginning for existing stream
                    mkstream=True  # Create stream if not exists
                )
                logger.info(f"✅ Consumer group '{group_name}' created on {topic}")
            except Exception as e:
                if "BUSYGROUP" in str(e):
                    logger.info(f"✅ Consumer group '{group_name}' already exists")
                else:
                    logger.error(f"❌ Failed to create consumer group: {e}")
                    raise
        
        logger.info(f"📥 Subscribing to {topic} (group={group_name}, consumer={consumer_name})")
        last_id = start_id
        msg_count = 0
        
        # P0.D.5: Configurable batch size from environment
        read_count = int(os.getenv('XREADGROUP_COUNT', '10'))
        
        while True:
            try:
                # Read from stream using consumer group
                messages = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {topic: last_id},
                    count=read_count,
                    block=block_ms
                )
                
                if not messages:
                    # No new messages, continue blocking
                    continue
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        msg_count += 1
                        
                        # Diagnostic logging (controlled by PIPELINE_DIAG env var)
                        if os.getenv('PIPELINE_DIAG') == 'true' and msg_count % 500 == 0:
                            logger.info(f"[DIAG] Heartbeat: delivered {msg_count} messages from {stream_name}")
                        
                        # Parse JSON payload (stream contract: fields["payload"] contains JSON)
                        if "payload" in fields:
                            try:
                                payload = json.loads(fields["payload"])
                                payload["_message_id"] = message_id
                                payload["_stream_name"] = stream_name
                                payload["_group_name"] = group_name
                                
                                yield payload
                                
                                # ACK message immediately after processing
                                await self.redis.xack(stream_name, group_name, message_id)
                                
                            except json.JSONDecodeError as e:
                                # DO NOT ACK on parse error - let it retry or go to DLQ
                                logger.error(f"❌ JSON decode error on {message_id}: {e}")
                                if os.getenv('PIPELINE_DIAG') == 'true':
                                    raw_data = str(fields).encode('utf-8')[:600].decode('utf-8', errors='ignore')
                                    logger.error(f"[DIAG] Raw data: {raw_data}")
                                # Continue without ACK - message stays in pending list
                                continue
            
            except asyncio.CancelledError:
                logger.info(f"🛑 Subscription cancelled: {topic}")
                break
            except Exception as e:
                logger.error(f"❌ Subscribe error on {topic}: {e}")
                await asyncio.sleep(5)  # Retry delay
    
    async def subscribe(
        self,
        topic: str,
        start_id: str = "$",
        block_ms: int = 1000
    ):
        """
        Subscribe to Redis stream (simple mode - no consumer groups)
        
        ⚠️  WARNING: This mode loses unprocessed messages on restart.
        Use subscribe_with_group() for production reliability.
        
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
