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

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
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

# ============================================================================
# SYMBOL-SPECIFIC CONFIGURATION
# ============================================================================

@dataclass
class SymbolConfig:
    """Per-symbol configuration overrides"""
    symbol: str
    min_r: Optional[float] = None
    harvest_ladder: Optional[str] = None
    set_be_at_r: Optional[float] = None
    trail_atr_mult: Optional[float] = None
    
    def get_min_r(self, default: float) -> float:
        """Get min_r with fallback to default"""
        return self.min_r if self.min_r is not None else default
    
    def get_set_be_at_r(self, default: float) -> float:
        """Get set_be_at_r with fallback to default"""
        return self.set_be_at_r if self.set_be_at_r is not None else default
    
    def get_trail_atr_mult(self, default: float) -> float:
        """Get trail_atr_mult with fallback to default"""
        return self.trail_atr_mult if self.trail_atr_mult is not None else default


@dataclass
class Position:
    """Tracked position per symbol"""
    symbol: str
    side: str  # LONG or SHORT
    qty: float
    entry_price: float
    current_price: float
    entry_risk: float  # Initial risk (from entry to SL)
    stop_loss: float
    unrealized_pnl: float = 0.0
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
            # Handle both nested payload.signal and flat dict formats
            if 'payload' in exec_event:
                payload = json.loads(exec_event.get('payload', '{}'))
                signal = payload.get('signal', {})
                symbol = signal.get('symbol', '').strip()
                side = signal.get('side', '').strip()
                status = payload.get('status', '').upper()
            else:
                # Flat dict format (direct from stream)
                symbol = exec_event.get('symbol', '').strip()
                side = exec_event.get('side', '').strip()
                status = exec_event.get('status', '').upper()
            
            # PRICE_UPDATE only needs symbol, not side
            if status == 'PRICE_UPDATE':
                if not symbol:
                    logger.debug(f"Skipping PRICE_UPDATE: missing symbol")
                    return False
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    price = float(exec_event.get('price', pos.current_price))
                    pos.current_price = price
                    # Calculate PNL based on position direction
                    if pos.side in ['BUY', 'LONG']:
                        pos.unrealized_pnl = (price - pos.entry_price) * pos.qty
                    else:  # SELL, SHORT
                        pos.unrealized_pnl = (pos.entry_price - price) * pos.qty
                    pos.last_update_ts = time.time()
                    logger.debug(f"💹 Price update: {symbol} @ {price} (pnl={pos.unrealized_pnl:.2f}, R={pos.r_level():.2f})")
                    return True
                else:
                    logger.debug(f"Skipping PRICE_UPDATE for unknown symbol: {symbol}")
                    return False
            
            # FILLED/PARTIAL require both symbol and side
            if not symbol or not side:
                logger.debug(f"Skipping execution: missing symbol/side in {list(exec_event.keys())}")
                return False
            
            if status not in ['FILLED', 'PARTIAL']:
                logger.debug(f"Skipping execution: status={status} for {symbol}")
                return False
            
            # Extract fill details
            qty = float(exec_event.get('qty', 0))
            price = float(exec_event.get('price', 0))
            entry_price = float(exec_event.get('entry_price', price))
            stop_loss = float(exec_event.get('stop_loss', 0))
            take_profit = float(exec_event.get('take_profit', 0))
            
            if qty <= 0:
                logger.debug(f"Skipping execution: qty={qty}")
                return False
            
            # Update or create position
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    entry_price=entry_price,
                    current_price=price,
                    entry_risk=abs(entry_price - stop_loss) if stop_loss > 0 else abs(entry_price * 0.02),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                logger.info(f"✅ New position: {symbol} {side} {qty} @ {entry_price}")
            else:
                pos = self.positions[symbol]
                pos.qty += qty if side == pos.side else -qty
                pos.current_price = price
                # Calculate PNL based on position direction
                if pos.side in ['BUY', 'LONG']:
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.qty
                else:  # SELL, SHORT
                    pos.unrealized_pnl = (pos.entry_price - price) * pos.qty
                pos.last_update_ts = time.time()
                logger.info(f"📊 Updated position: {symbol} qty={pos.qty} pnl={pos.unrealized_pnl:.2f}")
            
            return True
        
        except Exception as e:
            logger.warning(f"Failed to ingest execution: {e}", exc_info=True)
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
    
    def __init__(self, config: Config, redis_client: redis.Redis = None):
        self.config = config
        self.redis = redis_client
        self.symbol_configs: Dict[str, SymbolConfig] = {}
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
    
    def _load_symbol_config(self, symbol: str) -> SymbolConfig:
        """Load or cache symbol-specific configuration from Redis"""
        if symbol in self.symbol_configs:
            return self.symbol_configs[symbol]
        
        sym_config = SymbolConfig(symbol=symbol)
        
        if self.redis:
            try:
                config_key = f"quantum:config:harvest:{symbol}"
                config_data = self.redis.hgetall(config_key)
                
                if config_data:
                    sym_config.min_r = float(config_data.get('min_r', sym_config.min_r))
                    sym_config.harvest_ladder = config_data.get('ladder', sym_config.harvest_ladder)
                    sym_config.set_be_at_r = float(config_data.get('set_be_at_r', sym_config.set_be_at_r)) if 'set_be_at_r' in config_data else None
                    sym_config.trail_atr_mult = float(config_data.get('trail_atr_mult', sym_config.trail_atr_mult)) if 'trail_atr_mult' in config_data else None
                    logger.debug(f"✅ Loaded symbol config for {symbol}: {config_data}")
            except Exception as e:
                logger.debug(f"No symbol-specific config for {symbol}: {e}")
        
        self.symbol_configs[symbol] = sym_config
        return sym_config
    
    def _calculate_volatility(self, position: Position) -> float:
        """
        Calculate volatility proxy from position.
        Uses entry_risk as volatility indicator:
        - High entry_risk (> 2.5% of entry price) = high volatility
        - Low entry_risk (< 1% of entry price) = low volatility
        Returns: 0.0 to 2.0 (scale factor for harvest fractions)
        """
        if position.entry_price <= 0:
            return 1.0  # Default to normal ladder
        
        risk_pct = (position.entry_risk / position.entry_price) * 100
        
        # Volatility scaling:
        # - risk_pct < 1%: volatility_scale = 1.4 (close more, 35% instead of 25%)
        # - risk_pct 1-2.5%: volatility_scale = 1.0 (normal ladder)
        # - risk_pct > 2.5%: volatility_scale = 0.6 (close less, 15% instead of 25%)
        if risk_pct < 1.0:
            volatility_scale = 1.4  # Low vol, aggressive closes
        elif risk_pct > 2.5:
            volatility_scale = 0.6  # High vol, conservative closes
        else:
            volatility_scale = 1.0  # Normal
        
        return volatility_scale
    
    def _get_dynamic_ladder(self, position: Position) -> List[Tuple[float, float]]:
        """
        Get harvest ladder adjusted for volatility.
        Returns list of (r_trigger, adjusted_fraction)
        """
        volatility_scale = self._calculate_volatility(position)
        dynamic_ladder = []
        
        for r_trigger, base_fraction in self.ladder:
            adjusted_fraction = base_fraction * volatility_scale
            # Cap at reasonable limits
            adjusted_fraction = min(max(adjusted_fraction, 0.1), 0.5)
            dynamic_ladder.append((r_trigger, adjusted_fraction))
        
        if volatility_scale != 1.0:
            logger.debug(
                f"📈 Dynamic ladder for {position.symbol}: "
                f"vol_scale={volatility_scale:.2f}, "
                f"risk_pct={(position.entry_risk/position.entry_price*100):.1f}%"
            )
        
        return dynamic_ladder
    
    def evaluate(self, position: Position) -> List[HarvestIntent]:
        """Evaluate position for harvesting opportunities"""
        intents = []
        
        if not position:
            return intents
        
        # Load symbol-specific config or use defaults
        sym_cfg = self._load_symbol_config(position.symbol)
        min_r = sym_cfg.get_min_r(self.config.min_r)
        be_trigger = sym_cfg.get_set_be_at_r(self.config.harvest_set_be_at_r)
        trail_mult = sym_cfg.get_trail_atr_mult(self.config.harvest_trail_atr_mult)
        
        r = position.r_level()
        
        # Skip if below min_r
        if r < min_r:
            logger.debug(f"{position.symbol}: R={r:.2f} < min_r={min_r}")
            return intents
        
        logger.info(f"{position.symbol}: Evaluating R={r:.2f}")
        
        # Check for trailing stop opportunity after profit accumulates
        # Trail SL by (entry_risk * trail_atr_mult) below current price
        trail_distance = position.entry_risk * trail_mult
        trailing_sl = position.current_price - trail_distance
        
        # Only move SL up (reduce loss risk), never down
        if trailing_sl > position.stop_loss and r >= min_r:
            trail_intent = HarvestIntent(
                intent_type='MOVE_SL_TRAIL',
                symbol=position.symbol,
                side='MOVE_SL',
                qty=position.qty,
                reason=f'Trail SL by {trail_distance:.2f} @ R={r:.2f}',
                r_level=r,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"trail:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"trail:{position.symbol}:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(trail_intent)
            logger.info(
                f"🔄 Trailing SL: {position.symbol} {position.stop_loss:.2f} → {trailing_sl:.2f} @ R={r:.2f}"
            )
        
        # Check if we should move SL to break-even (symbol-specific trigger)
        if r >= be_trigger and position.stop_loss < position.entry_price:
            # Move SL to break-even (entry price)
            be_intent = HarvestIntent(
                intent_type='MOVE_SL_BREAKEVEN',
                symbol=position.symbol,
                side='MOVE_SL',  # Special side for SL moves
                qty=position.qty,
                reason=f'R={r:.2f} >= {be_trigger} (BE)',
                r_level=r,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"be:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"be:{position.symbol}:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(be_intent)
            logger.info(
                f"📍 Break-Even: {position.symbol} SL → {position.entry_price} @ R={r:.2f}"
            )
        
        # Get volatility-adjusted harvest ladder for this symbol
        dynamic_ladder = self._get_dynamic_ladder(position)
        
        # Check each ladder level
        for r_trigger, fraction_to_close in dynamic_ladder:
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
                    reason=f'R={r:.2f} >= {r_trigger} (vol-adjusted)',
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
        if intent.intent_type in ['MOVE_SL_BREAKEVEN', 'MOVE_SL_TRAIL']:
            # Only one SL move per symbol (either BE or TRAIL, whichever comes first)
            # Use symbol-only key to prevent duplicate moves
            return f"quantum:dedup:harvest:{intent.symbol}:{intent.intent_type}"
        else:
            # Regular harvest dedup by R level
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
            
            entry_id = self.redis.xadd(
                self.config.stream_trade_intent,
                {'payload': json.dumps(payload)}
            )
            logger.warning(
                f"⚠️  LIVE: {intent.intent_type} {intent.symbol} "
                f"{intent.qty} @ R={intent.r_level:.2f} - ORDER PUBLISHED (ID: {entry_id})"
            )
            
            # Record harvest history for dashboard/analytics
            if intent.intent_type == 'HARVEST_PARTIAL':
                self._record_harvest_history(intent)
            
            return True
        except Exception as e:
            logger.error(f"Failed to publish live: {e}")
            return False
    
    def _record_harvest_history(self, intent: HarvestIntent) -> None:
        """Record harvest to sorted set for historical tracking"""
        try:
            history_key = f"quantum:harvest:history:{intent.symbol}"
            ts = time.time()
            
            # Store in sorted set with timestamp as score
            history_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'qty': intent.qty,
                'r_level': intent.r_level,
                'pnl': intent.unrealized_pnl,
                'reason': intent.reason
            }
            
            self.redis.zadd(
                history_key,
                {json.dumps(history_entry): ts}
            )
            
            # Keep last 100 harvests per symbol (trim older ones)
            self.redis.zremrangebyrank(history_key, 0, -101)
            
            logger.debug(f"📊 Recorded harvest history for {intent.symbol}: {ts:.0f}")
        except Exception as e:
            logger.warning(f"Failed to record harvest history: {e}")
    
    def get_harvest_history(self, symbol: str, hours: int = 24) -> List[dict]:
        """Retrieve harvest history for symbol in last N hours"""
        try:
            history_key = f"quantum:harvest:history:{symbol}"
            now = time.time()
            start_ts = now - (hours * 3600)
            
            # Get all entries in time range (sorted by score/timestamp)
            entries = self.redis.zrangebyscore(
                history_key, 
                start_ts, 
                now,
                withscores=False
            )
            
            history = []
            for entry_json in entries:
                try:
                    entry = json.loads(entry_json)
                    history.append(entry)
                except:
                    pass
            
            return history
        except Exception as e:
            logger.warning(f"Failed to get harvest history for {symbol}: {e}")
            return []


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
        self.policy = HarvestPolicy(config, self.redis)
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
            # Handle both flat dict and JSON payload formats
            if 'payload' in msg_data:
                payload_str = msg_data.get('payload', '{}')
                exec_event = json.loads(payload_str)
            else:
                # Flat dict format (direct from stream)
                exec_event = {k.decode() if isinstance(k, bytes) else k: 
                             v.decode() if isinstance(v, bytes) else v 
                             for k, v in msg_data.items()}
            
            logger.debug(f"Processing execution: {exec_event.get('symbol')} {exec_event.get('status')}")
            
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
