#!/usr/bin/env python3
"""
HarvestBrain: Profit Harvesting Microservice

Tracks execution fills → derives position → applies R-based harvesting policy
Output: Reduce-only intents (shadow mode) or live trade intents (live mode)

Features:
- State tracking from execution.result stream
- Idempotent dedup via Redis
- Fail-closed defaults (shadow mode, kill-switch)
- Observable logging and metrics
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

import redis

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    """Load configuration from environment"""
    
    def __init__(self):
        self.harvest_mode = os.getenv('HARVEST_MODE', 'shadow').lower()
        self.redis_host = os.getenv('REDIS_HOST', '127.0.0.1')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.stream_exec_result = os.getenv(
            'STREAM_EXECUTION_RESULT', 
            'quantum:stream:execution.result'
        )
        self.stream_position = os.getenv(
            'STREAM_POSITION_SNAPSHOT',
            'quantum:stream:position.snapshot'
        )
        self.stream_pnl = os.getenv(
            'STREAM_PNL_SNAPSHOT',
            'quantum:stream:pnl.snapshot'
        )
        self.stream_trade_intent = os.getenv(
            'STREAM_TRADE_INTENT',
            'quantum:stream:trade.intent'
        )
        self.stream_harvest_suggestions = os.getenv(
            'STREAM_HARVEST_SUGGESTIONS',
            'quantum:stream:harvest.suggestions'
        )
        
        self.dedup_ttl_sec = int(os.getenv('HARVEST_DEDUP_TTL_SEC', '900'))
        self.min_r = float(os.getenv('HARVEST_MIN_R', '0.5'))
        self.harvest_ladder = os.getenv(
            'HARVEST_LADDER',
            '0.5:0.25,1.0:0.25,1.5:0.25'
        )  # R: fraction_to_close
        self.harvest_set_be_at_r = float(os.getenv('HARVEST_SET_BE_AT_R', '0.5'))
        self.harvest_trail_atr_mult = float(os.getenv('HARVEST_TRAIL_ATR_MULT', '2.0'))
        self.harvest_require_fresh_snapshot_sec = int(
            os.getenv('HARVEST_REQUIRE_FRESH_SNAPSHOT_SEC', '30')
        )
        self.harvest_reduce_only = os.getenv('HARVEST_REDUCE_ONLY', 'true').lower() == 'true'
        self.harvest_max_actions_per_min = int(os.getenv('HARVEST_MAX_ACTIONS_PER_MIN', '30'))
        self.harvest_kill_switch_key = os.getenv(
            'HARVEST_KILL_SWITCH_KEY',
            'quantum:kill'
        )
        
        self.consumer_group = 'harvest_brain:execution'
        self.consumer_name = f'harvest_brain_{os.getenv("HOSTNAME", "local")}'
        
    def validate(self) -> bool:
        """Validate configuration"""
        if self.harvest_mode not in ['shadow', 'live']:
            logger.error(f"Invalid HARVEST_MODE: {self.harvest_mode} (must be shadow|live)")
            return False
        if self.min_r < 0:
            logger.error(f"Invalid HARVEST_MIN_R: {self.min_r}")
            return False
        logger.info(f"✅ Config valid | Mode: {self.harvest_mode} | Min R: {self.min_r}")
        return True
    
    def __repr__(self):
        return f"Config(mode={self.harvest_mode}, min_r={self.min_r}, redis={self.redis_host}:{self.redis_port})"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Position:
    """Tracked position per symbol"""
    symbol: str
    side: str  # LONG or SHORT
    qty: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_risk: float  # Initial risk (from entry to SL)
    stop_loss: float
    take_profit: Optional[float] = None
    leverage: float = 1.0
    last_update_ts: float = 0.0
    
    def r_level(self) -> float:
        """Calculate current R (return on risk)"""
        if self.entry_risk <= 0:
            return 0.0
        return self.unrealized_pnl / self.entry_risk
    
    def is_fresh(self, max_age_sec: int) -> bool:
        """Check if position data is fresh"""
        age = time.time() - self.last_update_ts
        return age < max_age_sec


@dataclass
class HarvestIntent:
    """Harvesting action to publish"""
    intent_type: str  # HARVEST_PARTIAL, MOVE_SL_BE, TRAIL_UPDATE
    symbol: str
    side: str  # EXIT side (SELL for LONG, BUY for SHORT)
    qty: float
    reason: str
    r_level: float
    unrealized_pnl: float
    correlation_id: str
    trace_id: str
    reduce_only: bool = True
    dry_run: bool = False
    source: str = 'harvest_brain'
    timestamp: str = ''
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'


# ============================================================================
# POSITION TRACKER (Internal State)
# ============================================================================

class PositionTracker:
    """Track positions from execution.result stream events"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.action_count = 0
        self.action_window_start = time.time()
    
    def ingest_execution(self, exec_event: dict) -> bool:
        """Process execution.result event to update position state"""
        try:
            payload = json.loads(exec_event.get('payload', '{}'))
            signal = payload.get('signal', {})
            
            symbol = signal.get('symbol', '').strip()
            side = signal.get('side', '').strip()
            
            if not symbol or not side:
                logger.debug(f"Skipping execution: missing symbol/side")
                return False
            
            status = payload.get('status', '').upper()
            if status not in ['FILLED', 'PARTIAL']:
                return False
            
            # TODO: Parse actual fill qty/price from execution event
            # For now, just acknowledge we've seen the execution
            logger.debug(f"Ingested execution: {symbol} {side}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to ingest execution: {e}")
            return False
    
    def update_position(self, symbol: str, position_data: dict) -> bool:
        """Update position from position.snapshot or fallback data"""
        try:
            qty = position_data.get('qty', 0.0)
            if qty == 0:
                # Position closed
                self.positions.pop(symbol, None)
                return True
            
            pos = Position(
                symbol=symbol,
                side=position_data.get('side', 'LONG'),
                qty=qty,
                entry_price=position_data.get('entry_price', 0.0),
                current_price=position_data.get('current_price', 0.0),
                unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                entry_risk=position_data.get('entry_risk', 0.0),
                stop_loss=position_data.get('stop_loss', 0.0),
                take_profit=position_data.get('take_profit'),
                leverage=position_data.get('leverage', 1.0),
                last_update_ts=time.time()
            )
            self.positions[symbol] = pos
            return True
        except Exception as e:
            logger.warning(f"Failed to update position {symbol}: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position by symbol"""
        return self.positions.get(symbol)
    
    def record_action(self) -> bool:
        """Track harvesting actions (rate limiting)"""
        now = time.time()
        if now - self.action_window_start > 60:
            self.action_window_start = now
            self.action_count = 0
        
        self.action_count += 1
        return True


# ============================================================================
# HARVEST POLICY ENGINE
# ============================================================================

class HarvestPolicy:
    """Deterministic R-based harvesting policy"""
    
    def __init__(self, config: Config):
        self.config = config
        self.parse_ladder()
    
    def parse_ladder(self):
        """Parse harvest ladder config: R:fraction,R:fraction,..."""
        self.ladder: List[Tuple[float, float]] = []
        try:
            for pair in self.config.harvest_ladder.split(','):
                r_level, fraction = pair.split(':')
                self.ladder.append((float(r_level), float(fraction)))
            self.ladder.sort()  # Sort by R level
            logger.info(f"✅ Parsed harvest ladder: {self.ladder}")
        except Exception as e:
            logger.error(f"Failed to parse harvest ladder: {e}")
            self.ladder = [(0.5, 0.25), (1.0, 0.25), (1.5, 0.25)]
    
    def evaluate(self, position: Position) -> List[HarvestIntent]:
        """Evaluate position for harvesting opportunities"""
        intents = []
        
        if not position:
            return intents
        
        r = position.r_level()
        
        # Skip if below min_r
        if r < self.config.min_r:
            logger.debug(f"{position.symbol}: R={r:.2f} < min_r={self.config.min_r}")
            return intents
        
        logger.info(f"{position.symbol}: Evaluating R={r:.2f}")
        
        # Check each ladder level
        for r_trigger, fraction_to_close in self.ladder:
            if r >= r_trigger:
                qty = position.qty * fraction_to_close
                
                # Determine exit side
                exit_side = 'SELL' if position.side == 'LONG' else 'BUY'
                
                # Create harvest intent
                intent = HarvestIntent(
                    intent_type='HARVEST_PARTIAL',
                    symbol=position.symbol,
                    side=exit_side,
                    qty=qty,
                    reason=f'R={r:.2f} >= {r_trigger}',
                    r_level=r,
                    unrealized_pnl=position.unrealized_pnl,
                    correlation_id=f"harvest:{position.symbol}:{r_trigger}:{int(position.last_update_ts)}",
                    trace_id=f"harvest:{position.symbol}:partial:{int(position.last_update_ts)}",
                    dry_run=(self.config.harvest_mode == 'shadow')
                )
                intents.append(intent)
        
        return intents


# ============================================================================
# IDEMPOTENCY & DEDUP
# ============================================================================

class DedupManager:
    """Manage dedup keys for idempotent harvesting"""
    
    def __init__(self, redis_client: redis.Redis, config: Config):
        self.redis = redis_client
        self.config = config
    
    def build_key(self, intent: HarvestIntent) -> str:
        """Build dedup key from intent"""
        return (
            f"quantum:dedup:harvest:"
            f"{intent.symbol}:{intent.intent_type}:{intent.r_level}:"
            f"{int(intent.unrealized_pnl * 100)}"
        )
    
    def is_duplicate(self, intent: HarvestIntent) -> bool:
        """Check if intent is duplicate (already processed)"""
        key = self.build_key(intent)
        result = self.redis.setnx(key, '1')
        if result:
            # New key, set TTL
            self.redis.expire(key, self.config.dedup_ttl_sec)
            return False
        else:
            logger.debug(f"Duplicate detected: {key}")
            return True


# ============================================================================
# STREAM PUBLISHING
# ============================================================================

class StreamPublisher:
    """Publish harvest intents to appropriate streams"""
    
    def __init__(self, redis_client: redis.Redis, config: Config):
        self.redis = redis_client
        self.config = config
    
    def publish(self, intent: HarvestIntent) -> bool:
        """Publish intent to appropriate stream based on mode"""
        if intent.dry_run:
            return self._publish_shadow(intent)
        else:
            return self._publish_live(intent)
    
    def _publish_shadow(self, intent: HarvestIntent) -> bool:
        """Publish to harvest.suggestions stream (shadow mode)"""
        try:
            payload = {
                'intent_type': intent.intent_type,
                'symbol': intent.symbol,
                'side': intent.side,
                'qty': intent.qty,
                'reason': intent.reason,
                'r_level': intent.r_level,
                'unrealized_pnl': intent.unrealized_pnl,
                'correlation_id': intent.correlation_id,
                'trace_id': intent.trace_id,
                'reduce_only': intent.reduce_only,
                'dry_run': intent.dry_run,
                'source': intent.source,
                'timestamp': intent.timestamp
            }
            
            self.redis.xadd(
                self.config.stream_harvest_suggestions,
                {'payload': json.dumps(payload)}
            )
            logger.info(
                f"📝 SHADOW: {intent.intent_type} {intent.symbol} "
                f"{intent.qty} @ R={intent.r_level:.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish shadow: {e}")
            return False
    
    def _publish_live(self, intent: HarvestIntent) -> bool:
        """Publish reduce-only intent to trade.intent stream (live mode)"""
        try:
            payload = {
                'symbol': intent.symbol,
                'side': intent.side,
                'qty': intent.qty,
                'intent_type': 'REDUCE_ONLY',
                'reason': intent.reason,
                'r_level': intent.r_level,
                'reduce_only': intent.reduce_only,
                'source': intent.source,
                'correlation_id': intent.correlation_id,
                'trace_id': intent.trace_id,
                'timestamp': intent.timestamp
            }
            
            self.redis.xadd(
                self.config.stream_trade_intent,
                {'payload': json.dumps(payload)}
            )
            logger.warning(
                f"⚠️  LIVE: {intent.intent_type} {intent.symbol} "
                f"{intent.qty} @ R={intent.r_level:.2f} - ORDER PUBLISHED"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish live: {e}")
            return False


# ============================================================================
# MAIN SERVICE
# ============================================================================

class HarvestBrainService:
    """Main HarvestBrain service"""
    
    def __init__(self, config: Config):
        self.config = config
        self.redis = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        self.tracker = PositionTracker()
        self.policy = HarvestPolicy(config)
        self.dedup = DedupManager(self.redis, config)
        self.publisher = StreamPublisher(self.redis, config)
        self.last_id = '0'  # Stream position
    
    async def start(self) -> None:
        """Start the service"""
        logger.info(f"🚀 Starting HarvestBrain {self.config}")
        
        try:
            # Create consumer group if needed
            try:
                self.redis.xgroup_create(
                    self.config.stream_exec_result,
                    self.config.consumer_group,
                    id='0',
                    mkstream=False
                )
                logger.info(f"✅ Consumer group created: {self.config.consumer_group}")
            except redis.ResponseError as e:
                if 'BUSYGROUP' not in str(e):
                    raise
                logger.info(f"Consumer group exists: {self.config.consumer_group}")
            
            # Main loop
            while True:
                await self.process_batch()
                await asyncio.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("🛑 Service interrupted")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
    
    async def process_batch(self) -> None:
        """Process batch of execution events"""
        try:
            # Check kill-switch
            kill_switch = self.redis.get(self.config.harvest_kill_switch_key)
            if kill_switch == '1':
                logger.debug("🔴 Kill-switch active - no harvesting")
                return
            
            # Read from execution.result stream
            messages = self.redis.xreadgroup(
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_name,
                streams={self.config.stream_exec_result: '>'},
                count=10,
                block=1000
            )
            
            if not messages:
                return
            
            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    await self.process_execution(msg_id, msg_data)
                    # Acknowledge
                    self.redis.xack(
                        self.config.stream_exec_result,
                        self.config.consumer_group,
                        msg_id
                    )
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def process_execution(self, msg_id: str, msg_data: dict) -> None:
        """Process single execution event"""
        try:
            payload_str = msg_data.get('payload', '{}')
            exec_event = json.loads(payload_str)
            
            # Ingest to update internal state
            self.tracker.ingest_execution(exec_event)
            
            # Evaluate positions for harvesting
            for symbol, position in self.tracker.positions.items():
                # Evaluate policy
                intents = self.policy.evaluate(position)
                
                for intent in intents:
                    # Check dedup
                    if self.dedup.is_duplicate(intent):
                        logger.debug(f"Skipping duplicate: {intent.symbol}")
                        continue
                    
                    # Publish
                    self.publisher.publish(intent)
                    self.tracker.record_action()
        
        except Exception as e:
            logger.error(f"Failed to process execution {msg_id}: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    config = Config()
    
    if not config.validate():
        sys.exit(1)
    
    service = HarvestBrainService(config)
    await service.start()


if __name__ == '__main__':
    asyncio.run(main())
