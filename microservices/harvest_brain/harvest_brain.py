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
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

import redis

# Import P2 risk_kernel_harvest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal
)

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
        self.stream_apply_result = os.getenv(
            'STREAM_APPLY_RESULT', 
            'quantum:stream:apply.result'
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
        self.stream_apply_plan = os.getenv(
            'STREAM_APPLY_PLAN',
            'quantum:stream:apply.plan'
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
        self.harvest_scan_interval_sec = int(os.getenv('HARVEST_SCAN_INTERVAL_SEC', '5'))
        self.harvest_scan_batch = int(os.getenv('HARVEST_SCAN_BATCH', '200'))
        
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
        return f"Config(mode={self.harvest_mode}, min_r={self.min_r}, redis={self.redis_host}:{self.redis_port}, scan_interval={self.harvest_scan_interval_sec}s)"


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
    age_sec: float = 0.0  # Position age for P2 kill_score
    peak_price: float = 0.0  # Highest price reached (LONG) or lowest (SHORT)
    trough_price: float = 0.0  # Lowest price reached (LONG) or highest (SHORT)
    
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
    
    def _get_market_state(self, symbol: str) -> MarketState:
        """Fetch market state from Redis or return defaults"""
        try:
            key = f"quantum:market:{symbol}"
            data = self.redis.hgetall(key)
            
            if data:
                return MarketState(
                    sigma=float(data.get('sigma', 0.01)),
                    ts=float(data.get('ts', 0.35)),
                    p_trend=float(data.get('p_trend', 0.5)),
                    p_mr=float(data.get('p_mr', 0.3)),
                    p_chop=float(data.get('p_chop', 0.2))
                )
        except Exception as e:
            logger.debug(f"Failed to fetch market state for {symbol}: {e}")
        
        # Default market state (neutral)
        return MarketState(
            sigma=0.01,
            ts=0.35,
            p_trend=0.5,
            p_mr=0.3,
            p_chop=0.2
        )
    
    def _get_harvest_theta(self) -> HarvestTheta:
        """Get harvest theta from config or defaults"""
        return HarvestTheta(
            fallback_stop_pct=0.02,
            cost_bps=10.0,
            T1_R=2.0,
            T2_R=4.0,
            T3_R=6.0,
            lock_R=1.5,
            be_plus_pct=0.002,
            kill_threshold=0.6
        )
    
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
        """Evaluate position for harvesting opportunities using P2 risk_kernel"""
        intents = []
        
        if not position:
            return intents
        
        # Build P2 PositionSnapshot
        pos_snapshot = PositionSnapshot(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            current_price=position.current_price,
            peak_price=position.peak_price if position.peak_price > 0 else position.current_price,
            trough_price=position.trough_price if position.trough_price > 0 else position.current_price,
            age_sec=position.age_sec,
            unrealized_pnl=position.unrealized_pnl,
            current_sl=position.stop_loss,
            current_tp=position.take_profit
        )
        
        # Fetch market state from Redis (or use defaults)
        market_state = self._get_market_state(position.symbol)
        
        # Build P1 proposal (stop distance)
        p1_proposal = P1Proposal(
            stop_dist_pct=abs(position.entry_price - position.stop_loss) / position.entry_price if position.stop_loss else 0.02
        )
        
        # Get harvest theta from config
        theta = self._get_harvest_theta()
        
        # **RUN P2 HARVEST KERNEL**
        p2_result = compute_harvest_proposal(
            position=pos_snapshot,
            market_state=market_state,
            p1_proposal=p1_proposal,
            theta=theta
        )
        
        harvest_action = p2_result['harvest_action']
        r_net = p2_result['R_net']
        kill_score = p2_result['kill_score']
        reason_codes = p2_result['reason_codes']
        
        logger.info(
            f"[HARVEST] {position.symbol} | "
            f"R={r_net:.2f}R | KILL_SCORE={kill_score:.3f} | "
            f"Action={harvest_action} | Reasons={reason_codes}"
        )
        
        # Skip if below min_r threshold
        if r_net < self.config.min_r:
            logger.debug(f"{position.symbol}: R={r_net:.2f} < min_r={self.config.min_r}")
            return intents
        
        # Translate P2 harvest_action to trade.intent
        exit_side = 'SELL' if position.side == 'LONG' else 'BUY'
        
        if harvest_action == 'PARTIAL_25':
            qty = position.qty * 0.25
            intent = HarvestIntent(
                intent_type='HARVEST_PARTIAL_25',
                symbol=position.symbol,
                side=exit_side,
                qty=qty,
                reason=f'[P2] R={r_net:.2f}R >= T1=2R (25% harvest)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"p2:harvest25:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"p2:{position.symbol}:harvest25",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.info(f"[HARVEST] {position.symbol} PARTIAL_25 @ R={r_net:.2f} (25% of {position.qty})")
        
        elif harvest_action == 'PARTIAL_50':
            qty = position.qty * 0.50
            intent = HarvestIntent(
                intent_type='HARVEST_PARTIAL_50',
                symbol=position.symbol,
                side=exit_side,
                qty=qty,
                reason=f'[P2] R={r_net:.2f}R >= T2=4R (50% harvest)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"p2:harvest50:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"p2:{position.symbol}:harvest50",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.info(f"[HARVEST] {position.symbol} PARTIAL_50 @ R={r_net:.2f} (50% of {position.qty})")
        
        elif harvest_action == 'PARTIAL_75':
            qty = position.qty * 0.75
            intent = HarvestIntent(
                intent_type='HARVEST_PARTIAL_75',
                symbol=position.symbol,
                side=exit_side,
                qty=qty,
                reason=f'[P2] R={r_net:.2f}R >= T3=6R (75% harvest)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"p2:harvest75:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"p2:{position.symbol}:harvest75",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.info(f"[HARVEST] {position.symbol} PARTIAL_75 @ R={r_net:.2f} (75% of {position.qty})")
        
        elif harvest_action == 'FULL_CLOSE_PROPOSED':
            qty = position.qty
            intent = HarvestIntent(
                intent_type='FULL_CLOSE_PROPOSED',
                symbol=position.symbol,
                side=exit_side,
                qty=qty,
                reason=f'[P2] KILL_SCORE={kill_score:.2f} >= 0.6 (regime flip)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"p2:fullclose:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"p2:{position.symbol}:fullclose",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.warning(f"[HARVEST] {position.symbol} FULL_CLOSE_PROPOSED @ KILL={kill_score:.2f}")
        
        else:
            logger.debug(f"{position.symbol}: harvest_action={harvest_action} (HOLD)")
        
        return intents
        
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
        """Publish reduce-only plan directly to apply.plan stream (live mode)"""
        try:
            # Generate plan_id from correlation_id
            import hashlib
            plan_id = hashlib.sha256(intent.correlation_id.encode()).hexdigest()[:16]
            
            ts_unix = int(time.time())
            
            # Build apply.plan with FLAT fields (Apply Layer format)
            message_fields = {
                b"plan_id": plan_id.encode(),
                b"decision": b"EXECUTE",
                b"symbol": intent.symbol.encode(),
                b"side": intent.side.encode(),
                b"type": b"MARKET",
                b"qty": str(intent.qty).encode(),
                b"reduceOnly": b"true",
                b"source": b"harvest_brain",
                b"signature": b"harvest_brain",
                b"timestamp": str(ts_unix).encode(),
                b"reason": intent.reason.encode(),
                b"r_level": str(intent.r_level).encode()
            }
            
            # Publish directly to apply.plan stream
            entry_id = self.redis.xadd(
                self.config.stream_apply_plan,
                message_fields
            )
            
            # Auto-create permit (bypass Governor P3.3)
            permit_key = f"quantum:permit:p33:{plan_id}"
            self.redis.hset(permit_key, mapping={
                "allow": "true",
                "safe_qty": "0",
                "reason": "harvest_brain_auto_permit",
                "timestamp": str(ts_unix)
            })
            
            logger.warning(
                f"⚠️  LIVE: {intent.intent_type} {intent.symbol} "
                f"{intent.qty} @ R={intent.r_level:.2f} - PLAN PUBLISHED (ID: {plan_id}, msg: {entry_id})"
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
        self.last_scan_time = 0  # Last position scan timestamp
        
        # Price cache: symbol -> (price, timestamp)
        self.price_cache = {}
        self.price_cache_ttl = 2.0  # seconds
        
        # API rate limiting
        self.api_calls_this_tick = 0
        self.max_api_calls_per_tick = 10
        self.tick_start_time = time.time()
        
        # Per-symbol skip logging (prevent spam)
        self.last_skip_log = {}  # symbol -> timestamp
        self.skip_log_interval = 60  # seconds
        
        # Exchange position cache (for entry_price backfill)
        self.exchange_positions_cache = {}  # symbol -> {entryPrice, positionAmt, ...}
        self.exchange_positions_last_fetch = 0
        self.exchange_positions_ttl = 30  # seconds
    
    async def start(self) -> None:
        """Start the service"""
        logger.warning(
            f"HARVEST_START mode={self.config.harvest_mode} "
            f"stream={self.config.stream_apply_result} "
            f"group={self.config.consumer_group} "
            f"consumer={self.config.consumer_name} "
            f"scan_interval={self.config.harvest_scan_interval_sec}s "
            f"min_r={self.config.min_r}"
        )
        
        try:
            # STARTUP: Sync positions from Binance testnet (fresh init)
            logger.info("🔄 STARTUP: Syncing positions from Binance testnet...")
            await self._sync_positions_from_binance_at_startup()
            logger.info("✅ STARTUP: Position sync complete")
            
            # Create consumer group if needed
            try:
                self.redis.xgroup_create(
                    self.config.stream_apply_result,
                    self.config.consumer_group,
                    id='$',  # Start from NEW messages only (not old ones)
                    mkstream=True
                )
                logger.info(f"✅ Consumer group created: {self.config.consumer_group}")
            except redis.ResponseError as e:
                if 'BUSYGROUP' not in str(e):
                    raise
                logger.info(f"Consumer group exists: {self.config.consumer_group}")
            
            # Main loop: interleave apply.result processing and periodic scan
            while True:
                # Process apply.result events (non-blocking)
                await self.process_batch()
                
                # Periodic position scan
                now = time.time()
                if now - self.last_scan_time >= self.config.harvest_scan_interval_sec:
                    await self.scan_and_evaluate_positions()
                    self.last_scan_time = now
                
                await asyncio.sleep(0.5)  # Short sleep to avoid tight loop
        
        except KeyboardInterrupt:
            logger.info("🛑 Service interrupted")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
    
    async def _sync_positions_from_binance_at_startup(self) -> None:
        """Fetch positions from Binance at startup and sync to Redis"""
        try:
            import urllib.request
            import json as json_lib
            import hmac
            import hashlib
            
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
            
            if not api_key or not api_secret:
                logger.warning("⚠️  STARTUP: Missing BINANCE testnet credentials, skipping position sync")
                return
            
            # Get positions from Binance positionRisk endpoint
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            url = f"https://testnet.binancefuture.com/fapi/v2/positionRisk?{query_string}&signature={signature}"
            req = urllib.request.Request(url)
            req.add_header('X-MBX-APIKEY', api_key)
            
            with urllib.request.urlopen(req, timeout=5) as response:
                positions_data = json_lib.loads(response.read().decode())
            
            # Sync ONLY positions that exist on Binance
            synced_count = 0
            for pos in positions_data:
                amt = float(pos.get("positionAmt", 0))
                if amt == 0:  # Skip zero positions
                    continue
                
                symbol = pos["symbol"]
                side = "LONG" if amt > 0 else "SHORT"
                qty = abs(amt)
                entry_price = float(pos.get("entryPrice", 0))
                unrealized_pnl = float(pos.get("unRealizedProfit", 0))
                leverage = int(pos.get("leverage", 1))
                
                pos_key = f"quantum:position:{symbol}"
                position_data = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": str(qty),
                    "entry_price": str(entry_price),
                    "unrealized_pnl": str(unrealized_pnl),
                    "leverage": str(leverage),
                    "source": "harvest_brain_startup_sync",
                    "sync_timestamp": str(int(time.time()))
                }
                self.redis.hset(pos_key, mapping=position_data)
                synced_count += 1
                logger.debug(f"  ✓ Synced {symbol}: {side} {qty:.2f} @ {entry_price:.6f}")
            
            logger.info(f"✅ STARTUP: Synced {synced_count} positions from Binance testnet")
            
            # Remove any ghost positions that don't exist on Binance
            all_pos_keys = self.redis.keys("quantum:position:*")
            ghost_count = 0
            for key in all_pos_keys:
                symbol = key.replace("quantum:position:", "")
                # Check if symbol exists in positions_data
                symbol_exists = any(p["symbol"] == symbol for p in positions_data)
                # If it doesn't exist AND the position amount is 0, delete it
                if not symbol_exists:
                    # Double-check by looking for it in actual positions
                    actual_positions = [p for p in positions_data if float(p.get("positionAmt", 0)) != 0]
                    if not any(p["symbol"] == symbol for p in actual_positions):
                        self.redis.delete(key)
                        ghost_count += 1
                        logger.debug(f"  ✗ Removed ghost position: {symbol}")
            
            if ghost_count > 0:
                logger.info(f"🧹 STARTUP: Cleaned {ghost_count} ghost positions from Redis")
            
        except Exception as e:
            logger.error(f"⚠️  STARTUP: Failed to sync positions from Binance: {e}", exc_info=True)
    
    async def process_batch(self) -> None:
        """Process batch of execution events (non-blocking)"""
        try:
            # Check kill-switch
            kill_switch = self.redis.get(self.config.harvest_kill_switch_key)
            if kill_switch == '1':
                logger.debug("🔴 Kill-switch active - no harvesting")
                return
            
            # Read from apply.result stream (non-blocking: short block time)
            messages = self.redis.xreadgroup(
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_name,
                streams={self.config.stream_apply_result: '>'},
                count=10,
                block=100  # 100ms block (was 1000ms)
            )
            
            if not messages:
                return
            
            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    await self.process_apply_result(msg_id, msg_data)
                    # Acknowledge
                    self.redis.xack(
                        self.config.stream_apply_result,
                        self.config.consumer_group,
                        msg_id
                    )
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def process_apply_result(self, msg_id: str, msg_data: dict) -> None:
        """Process single apply.result event (P3 Harvest Restore)"""
        try:
            # Parse apply.result payload
            if 'payload' in msg_data:
                payload_str = msg_data.get('payload', '{}')
                event = json.loads(payload_str)
            else:
                event = {k.decode() if isinstance(k, bytes) else k: 
                        v.decode() if isinstance(v, bytes) else v 
                        for k, v in msg_data.items()}
            
            symbol = event.get('symbol', '').strip()
            status = event.get('status', '').upper()
            
            # Log trigger receipt (including TEST events)
            logger.debug(f"HARVEST_TRIGGER_RX symbol={symbol} status={status}")
            
            # Enrich position data from Redis (fail-open compute)
            await self._enrich_position_from_redis(symbol)
            
            # Update position tracking from fills
            if status in ['FILLED', 'PARTIAL_FILL']:
                self.tracker.ingest_execution(event)
                
                # Sync to quantum:position:{symbol} Redis key
                if symbol in self.tracker.positions:
                    pos = self.tracker.positions[symbol]
                    await self._sync_position_to_redis(pos)
            
            # Evaluate positions for harvesting
            for symbol_key, position in list(self.tracker.positions.items()):
                # Update position age
                position.age_sec = time.time() - position.last_update_ts
                
                # Evaluate policy (runs P2 risk_kernel_harvest)
                intents = self.policy.evaluate(position)
                
                for intent in intents:
                    # Check dedup
                    if self.dedup.is_duplicate(intent):
                        logger.debug(f"Skipping duplicate: {intent.symbol}")
                        continue
                    
                    # Publish to trade.intent
                    self.publisher.publish(intent)
                    self.tracker.record_action()
        
        except Exception as e:
            logger.error(f"Failed to process apply.result {msg_id}: {e}", exc_info=True)
    
    async def scan_and_evaluate_positions(self) -> None:
        """Periodic scan of all open positions for harvest evaluation"""
        try:
            # Check kill-switch
            kill_switch = self.redis.get(self.config.harvest_kill_switch_key)
            if kill_switch == '1':
                return
            
            # Reset API rate limit counter for new tick
            self.api_calls_this_tick = 0
            self.tick_start_time = time.time()
            
            # Fetch exchange positions (30s cache)
            await self._fetch_exchange_positions()
            
            # Scan all position keys (use SCAN for safety, not KEYS)
            position_keys = []
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor,
                    match='quantum:position:*',
                    count=100
                )
                # Filter out ledger/snapshot keys
                position_keys.extend([
                    k for k in keys 
                    if ':ledger:' not in k and ':snapshot:' not in k
                ])
                if cursor == 0:
                    break
                if len(position_keys) >= self.config.harvest_scan_batch:
                    break
            
            if not position_keys:
                return
            
            scanned_count = 0
            evaluated_count = 0
            emitted_count = 0
            skipped_risk_count = 0
            skipped_price_count = 0
            
            for pos_key in position_keys[:self.config.harvest_scan_batch]:
                try:
                    # Parse symbol from key
                    symbol = pos_key.replace('quantum:position:', '')
                    if not symbol or len(symbol) < 3:
                        continue
                    
                    scanned_count += 1
                    
                    # Enrich position data (compute missing fields)
                    await self._enrich_position_from_redis(symbol)
                    
                    # Get position data
                    pos_data = self.redis.hgetall(pos_key)
                    if not pos_data:
                        continue
                    
                    # Parse position fields
                    side = pos_data.get('side', 'LONG')
                    qty = float(pos_data.get('quantity', 0.0))
                    entry_price = float(pos_data.get('entry_price', 0.0))
                    entry_risk_usdt = float(pos_data.get('entry_risk_usdt', 0.0))
                    unrealized_pnl = float(pos_data.get('unrealized_pnl', 0.0))
                    risk_missing = int(pos_data.get('risk_missing', 0))
                    
                    # Skip if essential data missing
                    if qty == 0 or entry_price == 0:
                        continue
                    
                    # Skip if risk data missing
                    if risk_missing == 1 or entry_risk_usdt <= 0:
                        missing_fields = []
                        if entry_risk_usdt <= 0:
                            missing_fields.append('entry_risk_usdt')
                        if pos_data.get('atr_value', '0') == '0':
                            missing_fields.append('atr_value')
                        if pos_data.get('volatility_factor', '0') == '0':
                            missing_fields.append('volatility_factor')
                        
                        # Rate-limited logging
                        now = time.time()
                        if symbol not in self.last_skip_log or now - self.last_skip_log[symbol] > self.skip_log_interval:
                            logger.warning(
                                f"SKIP_RISK_MISSING symbol={symbol} "
                                f"missing_fields={','.join(missing_fields)}"
                            )
                            self.last_skip_log[symbol] = now
                        
                        skipped_risk_count += 1
                        continue
                    
                    # Get current mark price (with API fallback)
                    mark_price, mark_source = await self._get_mark_price(symbol)
                    if mark_price == 0:
                        # Rate-limited logging
                        now = time.time()
                        if symbol not in self.last_skip_log or now - self.last_skip_log[symbol] > self.skip_log_interval:
                            logger.info(
                                f"SKIP_NO_MARK_PRICE symbol={symbol} source={mark_source}"
                            )
                            self.last_skip_log[symbol] = now
                        
                        skipped_price_count += 1
                        continue
                    
                    # Compute R_net
                    cost_bps = 10.0  # 10 bps cost estimate
                    cost_est = unrealized_pnl * (cost_bps / 10000.0)
                    R_net = (unrealized_pnl - cost_est) / entry_risk_usdt if entry_risk_usdt > 0 else 0.0
                    
                    # Log evaluation
                    logger.info(
                        f"HARVEST_EVAL symbol={symbol} side={side} "
                        f"mark={mark_price:.6f} entry={entry_price:.6f} "
                        f"pnl={unrealized_pnl:.4f} risk={entry_risk_usdt:.4f} "
                        f"R_net={R_net:.3f} mark_source={mark_source}"
                    )
                    
                    evaluated_count += 1
                    
                    # Build Position object for policy evaluation
                    position = Position(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        entry_price=entry_price,
                        current_price=mark_price,
                        unrealized_pnl=unrealized_pnl,
                        entry_risk=entry_risk_usdt,
                        stop_loss=float(pos_data.get('stop_loss', 0.0)),
                        take_profit=float(pos_data.get('take_profit')) if pos_data.get('take_profit') else None,
                        leverage=float(pos_data.get('leverage', 1.0)),
                        last_update_ts=time.time()
                    )
                    
                    # Evaluate policy
                    intents = self.policy.evaluate(position)
                    
                    for intent in intents:
                        # Check dedup
                        if self.dedup.is_duplicate(intent):
                            logger.debug(f"Skipping duplicate: {intent.symbol} {intent.intent_type}")
                            continue
                        
                        # Log emission
                        logger.warning(
                            f"HARVEST_EMIT action={intent.intent_type} "
                            f"symbol={intent.symbol} "
                            f"close_pct={getattr(intent, 'close_pct', 'N/A')} "
                            f"reduceOnly={intent.reduce_only} "
                            f"R_net={R_net:.3f}"
                        )
                        
                        # Publish
                        self.publisher.publish(intent)
                        emitted_count += 1
                
                except Exception as e:
                    logger.debug(f"Error evaluating position {pos_key}: {e}")
            
            # Log tick summary (rate-limited: once per scan)
            if scanned_count > 0:
                logger.info(
                    f"HARVEST_TICK scanned={scanned_count} "
                    f"evaluated={evaluated_count} "
                    f"emitted={emitted_count} "
                    f"skipped_risk={skipped_risk_count} "
                    f"skipped_price={skipped_price_count} "
                    f"api_calls={self.api_calls_this_tick}"
                )
        
        except Exception as e:
            logger.error(f"Error in scan_and_evaluate_positions: {e}", exc_info=True)
    
    async def _fetch_exchange_positions(self) -> None:
        """Fetch all positions from Binance API and cache (30s TTL)"""
        try:
            now = time.time()
            if now - self.exchange_positions_last_fetch < self.exchange_positions_ttl:
                return  # Use cached data
            
            # Call Binance API /fapi/v2/positionRisk
            import urllib.request
            import json as json_lib
            import hmac
            import hashlib
            
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
            
            if not api_key or not api_secret:
                logger.warning("Missing BINANCE_TESTNET_API_KEY or SECRET, cannot fetch entry prices")
                return
            
            # Sign request
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            url = f"https://testnet.binancefuture.com/fapi/v2/positionRisk?{query_string}&signature={signature}"
            req = urllib.request.Request(url)
            req.add_header('X-MBX-APIKEY', api_key)
            
            with urllib.request.urlopen(req, timeout=5) as response:
                positions = json_lib.loads(response.read().decode())
                
                # Build cache
                self.exchange_positions_cache = {}
                for pos in positions:
                    symbol = pos.get('symbol')
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0:  # Only cache open positions
                        self.exchange_positions_cache[symbol] = {
                            'entryPrice': float(pos.get('entryPrice', 0)),
                            'positionAmt': position_amt,
                            'unrealizedProfit': float(pos.get('unRealizedProfit', 0))
                        }
                
                self.exchange_positions_last_fetch = now
                logger.info(f"Fetched {len(self.exchange_positions_cache)} exchange positions from Binance API")
        
        except Exception as e:
            logger.warning(f"Failed to fetch exchange positions: {e}")
    
    async def _enrich_position_from_redis(self, symbol: str) -> None:
        """Enrich position data with computed fields (fail-open compute)"""
        try:
            pos_key = f"quantum:position:{symbol}"
            pos_data = self.redis.hgetall(pos_key)
            
            if not pos_data:
                return
            
            # Parse position data
            side = pos_data.get('side', 'LONG')
            qty = float(pos_data.get('quantity', 0.0))
            entry_price = float(pos_data.get('entry_price', 0.0))
            
            # Backfill entry_price from exchange if missing
            if qty != 0 and entry_price == 0:
                if symbol in self.exchange_positions_cache:
                    exchange_entry = self.exchange_positions_cache[symbol]['entryPrice']
                    if exchange_entry > 0:
                        entry_price = exchange_entry
                        self.redis.hset(pos_key, 'entry_price', str(entry_price))
                        logger.info(f"[BACKFILL] {symbol}: entry_price={entry_price:.6f} from exchange API")
            
            if qty == 0 or entry_price == 0:
                return
            
            # Compute unrealized_pnl if missing
            unrealized_pnl = pos_data.get('unrealized_pnl')
            if not unrealized_pnl or unrealized_pnl == '' or float(unrealized_pnl) == 0:
                # Fetch mark price (or use cached ticker)
                mark_price, _ = await self._get_mark_price(symbol)  # Unpack tuple
                if mark_price > 0:
                    if side == 'LONG':
                        unrealized_pnl = (mark_price - entry_price) * qty
                    else:  # SHORT
                        unrealized_pnl = (entry_price - mark_price) * abs(qty)
                    
                    # Store computed unrealized_pnl
                    self.redis.hset(pos_key, 'unrealized_pnl', str(unrealized_pnl))
                    logger.debug(f"[HARVEST] {symbol}: Computed unrealized_pnl={unrealized_pnl:.4f}")
            
            # Compute entry_risk if missing
            entry_risk = pos_data.get('entry_risk_usdt')
            if not entry_risk or entry_risk == '' or float(entry_risk) <= 0:
                # Try to compute from atr_value/volatility_factor
                atr_value = float(pos_data.get('atr_value', 0.0))
                volatility_factor = float(pos_data.get('volatility_factor', 0.0))
                
                if atr_value > 0 and volatility_factor > 0:
                    risk_price = atr_value * volatility_factor
                    entry_risk_usdt = abs(qty) * risk_price
                    
                    # Store computed entry_risk
                    self.redis.hset(pos_key, 'entry_risk_usdt', str(entry_risk_usdt))
                    self.redis.hset(pos_key, 'risk_price', str(risk_price))
                    self.redis.hset(pos_key, 'risk_missing', '0')
                    logger.info(f"[HARVEST] {symbol}: Computed entry_risk_usdt={entry_risk_usdt:.4f} (atr={atr_value}, vol={volatility_factor})")
                else:
                    logger.warning(f"[HARVEST] {symbol}: SKIP_RISK_MISSING (atr={atr_value}, vol={volatility_factor})")
                    self.redis.hset(pos_key, 'risk_missing', '1')
        
        except Exception as e:
            logger.warning(f"Failed to enrich position {symbol}: {e}")
    
    async def _get_mark_price(self, symbol: str) -> tuple[float, str]:
        """Get mark price from Redis cache or Binance API. Returns (price, source)"""
        try:
            # Check in-memory cache first
            if symbol in self.price_cache:
                price, cached_ts = self.price_cache[symbol]
                if time.time() - cached_ts < self.price_cache_ttl:
                    return (price, 'cache')
            
            # Try quantum:ticker:{symbol}
            ticker_key = f"quantum:ticker:{symbol}"
            ticker_data = self.redis.hgetall(ticker_key)
            
            if ticker_data:
                mark_price = float(ticker_data.get('markPrice', 0.0))
                if mark_price > 0:
                    self.price_cache[symbol] = (mark_price, time.time())
                    return (mark_price, 'redis_ticker')
            
            # Try quantum:market:{symbol}
            market_key = f"quantum:market:{symbol}"
            market_data = self.redis.hgetall(market_key)
            if market_data:
                price = float(market_data.get('price', 0.0))
                if price > 0:
                    self.price_cache[symbol] = (price, time.time())
                    return (price, 'redis_market')
            
            # API fallback (rate-limited)
            if self.api_calls_this_tick >= self.max_api_calls_per_tick:
                return (0.0, 'rate_limited')
            
            # Call Binance testnet API
            import urllib.request
            import json as json_lib
            
            url = f"https://testnet.binancefuture.com/fapi/v1/ticker/price?symbol={symbol}"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json_lib.loads(response.read().decode())
                price = float(data.get('price', 0.0))
                
                if price > 0:
                    self.api_calls_this_tick += 1
                    self.price_cache[symbol] = (price, time.time())
                    logger.debug(f"Fetched {symbol} price from API: {price}")
                    return (price, 'api')
            
            return (0.0, 'unavailable')
        
        except Exception as e:
            logger.debug(f"Failed to get mark price for {symbol}: {e}")
            return (0.0, 'error')
    
    async def _sync_position_to_redis(self, position: Position) -> None:
        """Sync position to quantum:position:{symbol} Redis key"""
        try:
            key = f"quantum:position:{position.symbol}"
            
            position_data = {
                'symbol': position.symbol,
                'side': position.side,
                'qty': str(position.qty),
                'entry_price': str(position.entry_price),
                'current_price': str(position.current_price),
                'entry_risk': str(position.entry_risk),
                'stop_loss': str(position.stop_loss),
                'take_profit': str(position.take_profit) if position.take_profit else '',
                'unrealized_pnl': str(position.unrealized_pnl),
                'leverage': str(position.leverage),
                'age_sec': str(position.age_sec),
                'last_update_ts': str(position.last_update_ts),
                'source': 'harvest_brain'
            }
            
            self.redis.hset(key, mapping=position_data)
            self.redis.expire(key, 86400)  # 24h TTL
            
            logger.debug(f"[HARVEST] Synced quantum:position:{position.symbol}")
        except Exception as e:
            logger.warning(f"Failed to sync position to Redis: {e}")


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
