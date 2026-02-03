#!/usr/bin/env python3
"""
P3.2 Governor - Fund-Grade Limits + Auto-Disarm

Enforces rate limits and safety gates for Apply Layer execution.
Writes Redis permit keys that Apply Layer checks before executing trades.
Can automatically disarm the system (force dry_run) under unsafe conditions.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import logging
import redis
import subprocess
import urllib.request
import urllib.parse
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from prometheus_client import Counter, Gauge, start_http_server

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
METRIC_ALLOW = Counter('quantum_govern_allow_total', 'Plans allowed', ['symbol'])
METRIC_BLOCK = Counter('quantum_govern_block_total', 'Plans blocked', ['symbol', 'reason'])
METRIC_DISARM = Counter('quantum_govern_disarm_total', 'Auto-disarm events', ['reason'])
METRIC_EXEC_HOUR = Gauge('quantum_govern_exec_count_hour', 'Executions in last hour', ['symbol'])
METRIC_EXEC_5MIN = Gauge('quantum_govern_exec_count_5min', 'Executions in last 5min', ['symbol'])

# P2.9 Capital Allocation metrics
METRIC_P29_CHECKED = Counter('gov_p29_checked_total', 'P2.9 allocation checks', ['symbol'])
METRIC_P29_BLOCK = Counter('gov_p29_block_total', 'P2.9 allocation blocks', ['symbol'])
METRIC_P29_MISSING = Counter('gov_p29_missing_total', 'P2.9 allocation target missing', ['symbol'])
METRIC_P29_STALE = Counter('gov_p29_stale_total', 'P2.9 allocation target stale', ['symbol'])
METRIC_TESTNET_P29_ENABLED = Gauge('gov_testnet_p29_enabled', 'Testnet P2.9 gate enabled (0/1)')

# Testnet flatten metrics
METRIC_TESTNET_FLATTEN_ENABLED = Gauge('gov_testnet_flatten_enabled', 'Testnet flatten enabled (0/1)')
METRIC_TESTNET_FLATTEN_ATTEMPT = Counter('gov_testnet_flatten_attempt_total', 'Testnet flatten attempts')
METRIC_TESTNET_FLATTEN_NOOP = Counter('gov_testnet_flatten_noop_total', 'Testnet flatten no-ops (preconditions not met)')
METRIC_TESTNET_FLATTEN_ORDERS = Counter('gov_testnet_flatten_orders_total', 'Testnet flatten orders placed')
METRIC_TESTNET_FLATTEN_ERRORS = Counter('gov_testnet_flatten_errors_total', 'Testnet flatten errors')

# P3.1 Capital Efficiency metrics
METRIC_EFF_APPLY = Counter('p32_eff_apply_total', 'P3.1 efficiency applications', ['action', 'reason'])
METRIC_EFF_FACTOR = Gauge('p32_eff_factor', 'P3.1 efficiency downsize factor', ['symbol'])

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    
    # Rate limits
    MAX_EXEC_PER_HOUR = int(os.getenv('GOV_MAX_EXEC_PER_HOUR', '3'))
    MAX_EXEC_PER_5MIN = int(os.getenv('GOV_MAX_EXEC_PER_5MIN', '2'))
    MAX_REDUCE_NOTIONAL_PER_DAY_USD = float(os.getenv('GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD', '5000'))
    MAX_REDUCE_QTY_PER_DAY = float(os.getenv('GOV_MAX_REDUCE_QTY_PER_DAY', '0.02'))
    
    # Fund caps (testnet protection)
    MAX_OPEN_POSITIONS = int(os.getenv('GOV_MAX_OPEN_POSITIONS', '10'))
    MAX_NOTIONAL_PER_TRADE_USDT = float(os.getenv('GOV_MAX_NOTIONAL_PER_TRADE_USDT', '200'))
    MAX_TOTAL_NOTIONAL_USDT = float(os.getenv('GOV_MAX_TOTAL_NOTIONAL_USDT', '2000'))
    SYMBOL_COOLDOWN_SECONDS = int(os.getenv('GOV_SYMBOL_COOLDOWN_SECONDS', '60'))
    
    # Kill score gates (entry/exit separation)
    KILL_SCORE_CRITICAL = float(os.getenv('GOV_KILL_SCORE_CRITICAL', '0.8'))
    KILL_SCORE_OPEN_THRESHOLD = float(os.getenv('GOV_KILL_SCORE_OPEN_THRESHOLD', '0.85'))  # Higher = more permissive for entries
    KILL_SCORE_CLOSE_THRESHOLD = float(os.getenv('GOV_KILL_SCORE_CLOSE_THRESHOLD', '0.65'))  # Lower = stricter for exits
    KILL_SCORE_OPEN_QTY_SCALE_ALPHA = float(os.getenv('GOV_KILL_SCORE_OPEN_QTY_SCALE_ALPHA', '2.0'))  # Exponential scale factor
    KILL_SCORE_OPEN_MIN_SCALE = float(os.getenv('GOV_KILL_SCORE_OPEN_MIN_SCALE', '0.25'))  # Min 25% of original qty
    
    # P3.1 Capital Efficiency integration
    P31_MIN_CONF = float(os.getenv('P31_MIN_CONF', '0.65'))
    P31_DOWNSIZE_THRESHOLD = float(os.getenv('P31_DOWNSIZE_THRESHOLD', '0.45'))
    P31_MIN_FACTOR = float(os.getenv('P31_MIN_FACTOR', '0.25'))
    P31_MAX_EXTRA_COOLDOWN_SEC = int(os.getenv('P31_MAX_EXTRA_COOLDOWN_SEC', '120'))
    P31_EFF_TTL_SEC = int(os.getenv('P31_EFF_TTL_SEC', '600'))
    
    # Auto-disarm
    ENABLE_AUTO_DISARM = os.getenv('GOV_ENABLE_AUTO_DISARM', 'true').lower() == 'true'
    DISARM_ON_ERROR_COUNT = int(os.getenv('GOV_DISARM_ON_ERROR_COUNT', '5'))
    DISARM_ON_BURST_BREACH = os.getenv('GOV_DISARM_ON_BURST_BREACH', 'true').lower() == 'true'
    
    # Testnet P2.9 gate
    TESTNET_ENABLE_P29 = os.getenv('GOV_TESTNET_ENABLE_P29', 'false').lower() == 'true'
    
    # Testnet flatten (DANGEROUS - requires ESS + double confirmation)
    TESTNET_FORCE_FLATTEN = os.getenv('GOV_TESTNET_FORCE_FLATTEN', 'false').lower() == 'true'
    TESTNET_FORCE_FLATTEN_CONFIRM = os.getenv('GOV_TESTNET_FORCE_FLATTEN_CONFIRM', '')
    
    # Apply Layer config path (for disarm action)
    APPLY_CONFIG_PATH = os.getenv('APPLY_CONFIG_PATH', '/etc/quantum/apply-layer.env')
    
    # ========================================================================
    # ACTIVE SLOTS CONTROLLER (dynamic slot management + strict rotation)
    # ========================================================================
    ACTIVE_SLOTS_ENABLED = os.getenv('GOV_ACTIVE_SLOTS_ENABLED', 'true').lower() == 'true'
    ACTIVE_SLOTS_BASE = int(os.getenv('GOV_ACTIVE_SLOTS_BASE', '4'))
    ACTIVE_SLOTS_TREND_STRONG = int(os.getenv('GOV_ACTIVE_SLOTS_TREND_STRONG', '6'))
    ACTIVE_SLOTS_CHOP = int(os.getenv('GOV_ACTIVE_SLOTS_CHOP', '3'))
    ROTATION_THRESHOLD = float(os.getenv('GOV_ROTATION_THRESHOLD', '0.15'))  # New score must be 15% better
    MAX_CORRELATION = float(os.getenv('GOV_MAX_CORRELATION', '0.80'))  # 80% max correlation
    ROTATION_LOCK_TTL = int(os.getenv('GOV_ROTATION_LOCK_TTL', '120'))  # 120s timeout for rotation
    MAX_MARGIN_USAGE_PCT = float(os.getenv('GOV_MAX_MARGIN_USAGE_PCT', '0.65'))  # 65% margin cap
    
    # Metrics
    METRICS_PORT = int(os.getenv('METRICS_PORT', '8044'))
    
    # Streams
    STREAM_PLANS = os.getenv('STREAM_PLANS', 'quantum:stream:apply.plan')
    STREAM_RESULTS = os.getenv('STREAM_RESULTS', 'quantum:stream:apply.result')
    STREAM_EVENTS = os.getenv('STREAM_EVENTS', 'quantum:stream:governor.events')
    
    # Binance testnet credentials (for position/price fetching)
    BINANCE_TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', '')
    BINANCE_TESTNET_API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET', '')
    
    # Build tag
    BUILD_TAG = "governor-entry-exit-sep-v1"

# ============================================================================
# BINANCE TESTNET CLIENT (for real position/price data)
# ============================================================================
class BinanceTestnetClient:
    """Lightweight client for fetching position and price data"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com"
    
    def _sign_request(self, params: dict) -> str:
        """Sign request with HMAC SHA256"""
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        """Make HTTP request to Binance API"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign_request(params)
        
        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url}{endpoint}?{query_string}"
        
        req = urllib.request.Request(url, method=method)
        req.add_header('X-MBX-APIKEY', self.api_key)
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return None
    
    def get_position(self, symbol: str):
        """Get current position for symbol"""
        result = self._request('GET', '/fapi/v2/positionRisk', signed=True)
        if not result:
            return None
        
        for pos in result:
            if pos.get('symbol') == symbol:
                return {
                    'positionAmt': float(pos.get('positionAmt', 0)),
                    'side': pos.get('positionSide', 'BOTH'),  # Hedge mode uses positionSide
                    'unrealizedProfit': float(pos.get('unrealizedProfit', 0))
                }
        return None
    
    def get_mark_price(self, symbol: str):
        """Get current mark price for symbol"""
        result = self._request('GET', '/fapi/v1/premiumIndex', params={'symbol': symbol}, signed=False)
        if not result:
            return None
        
        return float(result['markPrice'])

# ============================================================================
# GOVERNOR CORE
# ============================================================================
class Governor:
    def __init__(self, config):
        self.config = config
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        # Binance client for real position/price data
        self.binance_client = None
        if config.BINANCE_TESTNET_API_KEY and config.BINANCE_TESTNET_API_SECRET:
            self.binance_client = BinanceTestnetClient(
                config.BINANCE_TESTNET_API_KEY,
                config.BINANCE_TESTNET_API_SECRET
            )
            logger.info("Binance testnet client initialized")
        else:
            logger.warning("No Binance credentials - will use fallback limits")
        
        # Execution tracking (in-memory + Redis backup)
        self.exec_history = defaultdict(list)  # {symbol: [timestamp, ...]}
        self.error_count = 0
        self.last_disarm_check = time.time()
        
        logger.info("="*80)
        logger.info(f"P3.2 Governor [{config.BUILD_TAG}]")
        logger.info("="*80)
        logger.info(f"Entry/Exit Separation: ENABLED")
        logger.info(f"  - OPEN threshold (base): {config.KILL_SCORE_OPEN_THRESHOLD} (dynamic)")
        logger.info(f"  - CLOSE threshold (base): {config.KILL_SCORE_CLOSE_THRESHOLD} (dynamic)")
        logger.info(f"  - CRITICAL threshold: {config.KILL_SCORE_CRITICAL} (hard block)")
        logger.info(f"  - Qty scale alpha: {config.KILL_SCORE_OPEN_QTY_SCALE_ALPHA}")
        logger.info(f"  - Min qty scale: {config.KILL_SCORE_OPEN_MIN_SCALE}")
        logger.info(f"Max exec/hour: {config.MAX_EXEC_PER_HOUR}, Max exec/5min: {config.MAX_EXEC_PER_5MIN}")
        logger.info(f"Auto-disarm: {config.ENABLE_AUTO_DISARM}")
        logger.info(f"Fund caps: {config.MAX_OPEN_POSITIONS} positions, ${config.MAX_NOTIONAL_PER_TRADE_USDT}/trade, ${config.MAX_TOTAL_NOTIONAL_USDT} total")
        logger.info(f"Symbol cooldown: {config.SYMBOL_COOLDOWN_SECONDS}s")
        logger.info("="*80)
        
        # Testnet P2.9 gate status
        if config.TESTNET_ENABLE_P29:
            logger.info("Testnet P2.9 gate ENABLED (GOV_TESTNET_ENABLE_P29=true)")
            METRIC_TESTNET_P29_ENABLED.set(1)
        else:
            logger.info("Testnet P2.9 gate disabled")
            METRIC_TESTNET_P29_ENABLED.set(0)
        
        # Testnet flatten status (DANGEROUS - requires ESS + confirmation)
        if config.TESTNET_FORCE_FLATTEN and config.TESTNET_FORCE_FLATTEN_CONFIRM == "FLATTEN_NOW":
            logger.warning("Testnet flatten ARMED (GOV_TESTNET_FORCE_FLATTEN=true + CONFIRM=FLATTEN_NOW)")
            logger.warning(f"Testnet flatten guards: MAX_SYMBOLS={config.MAX_FLATTEN_SYMBOLS} MIN_NOTIONAL=${config.MIN_FLATTEN_NOTIONAL_USD}")
            logger.warning("Testnet flatten requires ESS active + Redis arm key")
            METRIC_TESTNET_FLATTEN_ENABLED.set(1)
        else:
            logger.info("Testnet flatten disabled")
            METRIC_TESTNET_FLATTEN_ENABLED.set(0)
    
    def run(self):
        """Main loop: event-driven consumer group on apply.plan stream"""
        logger.info("Governor starting main loop (event-driven mode)")
        logger.info("Consumer group: governor on quantum:stream:apply.plan")
        
        consumer_group = "governor"
        consumer_id = f"gov-{os.getpid()}"
        
        # Create consumer group (idempotent)
        try:
            self.redis.xgroup_create(self.config.STREAM_PLANS, consumer_group, id='$', mkstream=True)
            logger.info(f"Consumer group '{consumer_group}' created")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Consumer group create error: {e}")
        
        while True:
            try:
                # XREADGROUP: read new messages from consumer group
                messages = self.redis.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer_id,
                    streams={self.config.STREAM_PLANS: '>'},
                    count=10,
                    block=1000  # 1s timeout
                )
                
                if not messages:
                    # Periodic tasks during idle
                    self._update_metrics()
                    self._check_auto_disarm()
                    self._check_testnet_flatten_arm()
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        # Evaluate plan
                        self._evaluate_plan(message_id, data)
                        
                        # ACK message after processing
                        self.redis.xack(self.config.STREAM_PLANS, consumer_group, message_id)
                
                # Periodic tasks
                self._update_metrics()
                self._check_auto_disarm()
                self._check_testnet_flatten_arm()
                
            except KeyboardInterrupt:
                logger.info("Governor shutting down")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.error_count += 1
                time.sleep(1)
    
    def _evaluate_plan(self, plan_id_msg, data):
        """Evaluate a plan and issue permit or block"""
        try:
            # Extract plan_id from data (Apply Layer deterministic hash)
            plan_id = data.get('plan_id')
            if not plan_id:
                logger.warning(f"Message {plan_id_msg}: Missing plan_id field")
                return
            
            symbol = data.get('symbol', 'UNKNOWN')
            action = data.get('action', 'UNKNOWN')
            decision = data.get('decision', 'SKIP')
            kill_score = float(data.get('kill_score', '0'))
            side = data.get('side', '')
            qty = data.get('qty', '0')
            
            # Get current mode
            current_mode = os.getenv('APPLY_MODE', 'testnet')
            
            logger.info(f"{symbol}: Evaluating plan {plan_id[:8]} (action={action}, decision={decision}, kill_score={kill_score:.3f}, mode={current_mode})")
            
            # Skip if decision is not EXECUTE
            if decision != 'EXECUTE':
                logger.debug(f"{symbol}: Plan {plan_id[:8]} decision={decision}, skipping permit")
                return
            
            # Check if permit already exists (idempotency)
            permit_key = f"quantum:permit:{plan_id}"
            if self.redis.exists(permit_key):
                logger.debug(f"{symbol}: Permit already exists for plan {plan_id[:8]}")
                return
            
            # ===================================================================
            # ACTIVE SLOTS CONTROLLER (strict rotation with close confirmation)
            # ===================================================================
            if self.config.ACTIVE_SLOTS_ENABLED and action not in ['FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'PARTIAL_25']:
                # Only apply for OPEN actions (skip closes)
                logger.info(f"{symbol}: Active Slots enabled - checking position limits")
                
                # Step 1: Get policy universe
                policy_data = self.redis.hget('quantum:policy:current', 'universe_symbols')
                if not policy_data:
                    logger.warning(f"{symbol}: No policy universe - BLOCKING (fail-closed)")
                    self._block_plan(plan_id, symbol, 'active_slots_no_policy')
                    return
                
                policy_universe = json.loads(policy_data)
                if symbol not in policy_universe:
                    logger.warning(f"{symbol}: NOT in policy universe ({len(policy_universe)} symbols) - BLOCKING")
                    self._block_plan(plan_id, symbol, 'active_slots_not_in_universe')
                    return
                
                logger.info(f"{symbol}: In policy universe ({len(policy_universe)} symbols)")
                
                # Step 2: Get open positions snapshot
                open_positions = self._get_open_positions_snapshot()
                open_symbols = [p['symbol'] for p in open_positions]
                
                # Step 3: Detect regime and compute desired slots
                regime, desired_slots = self._detect_market_regime(symbol)
                logger.info(f"{symbol}: Regime={regime}, desired_slots={desired_slots}, current_open={len(open_positions)}")
                
                # Step 4: Check if slots available
                if len(open_positions) < desired_slots:
                    logger.info(f"{symbol}: Slots available ({len(open_positions)}/{desired_slots}) - ALLOW")
                    # Continue to next gates
                else:
                    # Slots FULL - check for rotation
                    logger.info(f"{symbol}: Slots FULL ({len(open_positions)}/{desired_slots}) - checking rotation")
                    
                    # Step 5: Check if rotation lock exists
                    rotation_lock_key = f"quantum:rotation:lock:{plan_id}"
                    lock_data = self.redis.get(rotation_lock_key)
                    
                    if not lock_data:
                        # NO LOCK: Must initiate rotation (close weakest)
                        logger.info(f"{symbol}: No rotation lock - initiating rotation")
                        
                        # Find weakest position
                        sorted_positions = sorted(open_positions, key=lambda p: p['weakness_score'], reverse=True)
                        weakest = sorted_positions[0]
                        
                        logger.info(f"{symbol}: Weakest position: {weakest['symbol']} (weakness={weakest['weakness_score']:.4f}, pnl={weakest['pnl_pct']*100:.2f}%)")
                        
                        # Check rotation threshold: new must be significantly better
                        # For now, always rotate (can add score comparison later)
                        # TODO: Add score comparison from intent stream
                        
                        # Create CLOSE plan for weakest (emit to apply.plan stream)
                        close_plan_id = f"rot_{plan_id[:8]}_{weakest['symbol']}"
                        close_plan = {
                            'plan_id': close_plan_id,
                            'symbol': weakest['symbol'],
                            'action': 'FULL_CLOSE_PROPOSED',
                            'side': 'CLOSE',
                            'qty': '0',  # Apply Layer will compute
                            'reduceOnly': 'true',
                            'decision': 'EXECUTE',
                            'kill_score': '0',
                            'rotation_trigger': 'true',
                            'new_symbol': symbol,
                            'new_plan_id': plan_id,
                            'timestamp': str(time.time())
                        }
                        
                        self.redis.xadd(self.config.STREAM_PLANS, close_plan)
                        logger.info(f"{symbol}: ROTATION CLOSE emitted for {weakest['symbol']} (plan_id={close_plan_id[:8]})")
                        
                        # Create rotation lock with TTL
                        lock_payload = json.dumps({
                            'new_symbol': symbol,
                            'new_plan_id': plan_id,
                            'close_symbol': weakest['symbol'],
                            'close_plan_id': close_plan_id,
                            'created_at': time.time()
                        })
                        self.redis.setex(rotation_lock_key, self.config.ROTATION_LOCK_TTL, lock_payload)
                        logger.info(f"{symbol}: ROTATION LOCK created (TTL={self.config.ROTATION_LOCK_TTL}s)")
                        
                        # BLOCK entry until close confirmed
                        self._block_plan(plan_id, symbol, 'active_slots_waiting_rotation_close')
                        
                        # Publish event
                        self.redis.xadd(self.config.STREAM_EVENTS, {
                            'event': 'ROTATION_LOCK_CREATED',
                            'new_symbol': symbol,
                            'new_plan_id': plan_id,
                            'close_symbol': weakest['symbol'],
                            'close_plan_id': close_plan_id,
                            'timestamp': str(time.time())
                        })
                        return
                    
                    else:
                        # LOCK EXISTS: Check if close confirmed
                        lock_info = json.loads(lock_data)
                        close_plan_id = lock_info['close_plan_id']
                        close_symbol = lock_info['close_symbol']
                        lock_age = time.time() - lock_info['created_at']
                        
                        logger.info(f"{symbol}: Rotation lock found - checking close confirmation (age={lock_age:.1f}s)")
                        
                        if self._is_close_confirmed(close_plan_id):
                            # CLOSE CONFIRMED: Allow entry
                            logger.info(f"{symbol}: Close CONFIRMED for {close_symbol} - ALLOW entry after rotation")
                            
                            # Delete lock
                            self.redis.delete(rotation_lock_key)
                            logger.info(f"{symbol}: Rotation lock deleted")
                            
                            # Publish event
                            self.redis.xadd(self.config.STREAM_EVENTS, {
                                'event': 'ROTATION_COMPLETE',
                                'new_symbol': symbol,
                                'new_plan_id': plan_id,
                                'close_symbol': close_symbol,
                                'close_plan_id': close_plan_id,
                                'timestamp': str(time.time())
                            })
                            
                            # Continue to next gates (allow entry)
                        else:
                            # CLOSE NOT YET CONFIRMED
                            if lock_age > self.config.ROTATION_LOCK_TTL:
                                # TIMEOUT: Lock expired without confirmation
                                logger.error(f"{symbol}: Rotation TIMEOUT - lock expired without close confirmation ({lock_age:.1f}s)")
                                
                                # Delete expired lock
                                self.redis.delete(rotation_lock_key)
                                
                                # BLOCK entry (fail-closed)
                                self._block_plan(plan_id, symbol, 'active_slots_rotation_timeout')
                                
                                # Publish event
                                self.redis.xadd(self.config.STREAM_EVENTS, {
                                    'event': 'ROTATION_TIMEOUT',
                                    'new_symbol': symbol,
                                    'new_plan_id': plan_id,
                                    'close_symbol': close_symbol,
                                    'close_plan_id': close_plan_id,
                                    'lock_age': str(lock_age),
                                    'timestamp': str(time.time())
                                })
                                return
                            else:
                                # STILL WAITING: Block entry
                                logger.info(f"{symbol}: Rotation in progress - waiting for close confirmation ({lock_age:.1f}s / {self.config.ROTATION_LOCK_TTL}s)")
                                self._block_plan(plan_id, symbol, 'active_slots_waiting_rotation_close')
                                return
            
            # ===================================================================
            # END ACTIVE SLOTS CONTROLLER
            # ===================================================================
            
            # TESTNET MODE: Apply fund caps for protection
            if current_mode == 'testnet':
                logger.info(f"{symbol}: Testnet mode - applying fund caps for plan {plan_id[:8]}")
                
                # Gate -1: P2.8 Budget Check (TESTNET: log only, don't block)
                budget_violation = self._check_portfolio_budget(symbol, plan_id)
                if budget_violation:
                    logger.warning(f"{symbol}: P2.8 budget violation detected (testnet mode - NOT blocking)")
                
                # Gate 0.5: P2.9 Allocation Target Check (TESTNET: only if flag enabled)
                if self.config.TESTNET_ENABLE_P29:
                    logger.info(f"{symbol}: Testnet P2.9 gate enabled - checking allocation target")
                    p29_block, p29_reason = self._check_p29_allocation_target(symbol, plan_id)
                    if p29_block:
                        logger.warning(f"{symbol}: P2.9 allocation violation: {p29_reason} (testnet mode - NOT blocking)")
                
                # Gate 0: Kill-switch check
                kill_switch = self.redis.get('quantum:kill')
                if kill_switch == '1':
                    logger.warning(f"{symbol}: KILL-SWITCH ACTIVE - blocking all execution")
                    self._block_plan(plan_id, symbol, 'kill_switch_active')
                    return
                
                # Identify if this is a CLOSE action (exempt from fund caps)
                is_close_action = action in ['FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'PARTIAL_25']
                
                if is_close_action:
                    logger.info(f"{symbol}: CLOSE action ({action}) - bypassing fund caps")
                    self._issue_permit(plan_id, symbol, computed_qty=0.0, computed_notional=0.0)
                    self.redis.setex(f"quantum:governor:last_exec:{symbol}", 3600, str(time.time()))
                    return
                
                # OPEN actions: Apply full fund cap gates
                logger.info(f"{symbol}: OPEN action ({action}) - applying fund caps")
                
                # Gate 1: Symbol cooldown (60s between executions)
                last_exec_key = f"quantum:governor:last_exec:{symbol}"
                last_exec_ts = self.redis.get(last_exec_key)
                if last_exec_ts:
                    time_since_last = time.time() - float(last_exec_ts)
                    if time_since_last < self.config.SYMBOL_COOLDOWN_SECONDS:
                        remaining = self.config.SYMBOL_COOLDOWN_SECONDS - time_since_last
                        logger.warning(f"{symbol}: Cooldown active - {remaining:.1f}s remaining")
                        self._block_plan(plan_id, symbol, 'symbol_cooldown')
                        return
                
                # Gate 2: Max open positions (fetch from Binance)
                if self.binance_client and side in ['BUY', 'LONG']:
                    try:
                        account_data = self.binance_client._request('GET', '/fapi/v2/account', signed=True)
                        if account_data:
                            positions = account_data.get('positions', [])
                            open_positions = [p for p in positions if abs(float(p.get('positionAmt', 0))) > 0.0001]
                            position_count = len(open_positions)
                            
                            if position_count >= self.config.MAX_OPEN_POSITIONS:
                                logger.warning(f"{symbol}: Max positions reached ({position_count}/{self.config.MAX_OPEN_POSITIONS})")
                                self._block_plan(plan_id, symbol, 'max_positions_reached')
                                return
                            
                            logger.info(f"Portfolio: {position_count}/{self.config.MAX_OPEN_POSITIONS} positions open")
                    except Exception as e:
                        logger.error(f"Error checking position count: {e}")
                
                # Gate 3: Max notional per trade
                if self.binance_client:
                    try:
                        mark_price = self.binance_client.get_mark_price(symbol)
                        if mark_price and qty:
                            notional = abs(float(qty)) * mark_price
                            if notional > self.config.MAX_NOTIONAL_PER_TRADE_USDT:
                                logger.warning(f"{symbol}: Notional ${notional:.2f} exceeds ${self.config.MAX_NOTIONAL_PER_TRADE_USDT} limit")
                                self._block_plan(plan_id, symbol, 'notional_per_trade_exceeded')
                                return
                            logger.info(f"{symbol}: Notional ${notional:.2f} OK (limit ${self.config.MAX_NOTIONAL_PER_TRADE_USDT})")
                    except Exception as e:
                        logger.error(f"Error checking notional: {e}")
                
                # Gate 4: Max total portfolio notional
                if self.binance_client:
                    try:
                        account_data = self.binance_client._request('GET', '/fapi/v2/account', signed=True)
                        if account_data:
                            positions = account_data.get('positions', [])
                            total_notional = sum(abs(float(p.get('notional', 0))) for p in positions)
                            
                            if total_notional >= self.config.MAX_TOTAL_NOTIONAL_USDT:
                                logger.warning(f"Portfolio notional ${total_notional:.2f} >= ${self.config.MAX_TOTAL_NOTIONAL_USDT}")
                                self._block_plan(plan_id, symbol, 'total_notional_exceeded')
                                return
                            
                            logger.info(f"Portfolio notional: ${total_notional:.2f}/{self.config.MAX_TOTAL_NOTIONAL_USDT} USDT")
                    except Exception as e:
                        logger.error(f"Error checking total notional: {e}")
                
                # All gates passed - issue permit and record execution timestamp
                self._issue_permit(plan_id, symbol, computed_qty=0.0, computed_notional=0.0)
                self.redis.setex(last_exec_key, 3600, str(time.time()))  # Record for cooldown
                return
            
            # DRY_RUN MODE: Auto-approve (no real execution anyway)
            if current_mode == 'dry_run':
                logger.info(f"{symbol}: Dry-run mode - auto-approving plan {plan_id[:8]}")
                # Issue permit with minimal values (not executing anyway)
                self._issue_permit(plan_id, symbol, computed_qty=0.0, computed_notional=0.0)
                return
            
            # PRODUCTION MODE: Apply full risk gates
            logger.info(f"{symbol}: Production mode - applying risk gates for plan {plan_id[:8]}")
            
            # Gate 0: P2.8 Portfolio Budget (via budget governor integration)
            budget_violation = self._check_portfolio_budget(symbol, plan_id)
            if budget_violation:
                self._block_plan(plan_id, symbol, 'p28_budget_violation')
                return
            
            # Gate 0.5: P2.9 Capital Allocation Target (fail-open if missing/stale)
            # Fetch position notional for comparison (will compute after fetching position)
            allocation_violation, allocation_reason = self._check_p29_allocation_target(
                symbol, plan_id, position_notional_usd=None  # Will check after position fetch
            )
            if allocation_violation:
                self._block_plan(plan_id, symbol, allocation_reason)
                return
            
            # Gate 1: Kill score with entry/exit separation
            is_close_action = action in ['FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'PARTIAL_25']
            reduce_only = str(data.get('reduceOnly', '')).lower() == 'true'
            
            # Determine gate type and threshold
            if is_close_action or reduce_only:
                gate_type = "close"
                threshold = self._compute_close_threshold(kill_score, unrealized_pnl_pct=0.0)
                
                if kill_score >= threshold:
                    logger.warning(
                        f"{symbol}: CLOSE blocked - kill_score={kill_score:.3f} >= close_threshold={threshold:.3f} "
                        f"(action={action}, plan_id={plan_id[:8]})"
                    )
                    self._block_plan(plan_id, symbol, f'kill_score_close_blocked:k={kill_score:.3f}:t={threshold:.3f}')
                    return
                else:
                    logger.info(
                        f"{symbol}: CLOSE allowed - kill_score={kill_score:.3f} < close_threshold={threshold:.3f}"
                    )
            else:
                # OPEN action: use more permissive threshold + qty scaling
                gate_type = "open"
                confidence = float(data.get('confidence', '0.5'))
                k_regime_flip = float(data.get('k_regime_flip', '0.0'))
                
                threshold = self._compute_open_threshold(kill_score, confidence, k_regime_flip)
                qty_scale = self._qty_scale_from_kill(kill_score, threshold)
                
                if kill_score >= self.config.KILL_SCORE_CRITICAL:
                    # Extreme kill_score: hard block even for OPEN
                    logger.warning(
                        f"{symbol}: OPEN blocked - kill_score={kill_score:.3f} >= CRITICAL={self.config.KILL_SCORE_CRITICAL} "
                        f"(action={action}, plan_id={plan_id[:8]})"
                    )
                    self._block_plan(plan_id, symbol, f'kill_score_critical_open:k={kill_score:.3f}')
                    return
                elif kill_score >= threshold:
                    # Above threshold but below critical: allow with scaled qty
                    logger.info(
                        f"{symbol}: OPEN allowed with qty_scale={qty_scale:.2f} - "
                        f"kill_score={kill_score:.3f} >= open_threshold={threshold:.3f} "
                        f"(conf={confidence:.2f}, regime_flip={k_regime_flip:.2f})"
                    )
                    # TODO: Apply qty_scale to computed_qty before issuing permit
                    # For now, just log (qty scaling happens later in flow)
                else:
                    logger.info(
                        f"{symbol}: OPEN allowed - kill_score={kill_score:.3f} < open_threshold={threshold:.3f}"
                    )
            
            # Gate 2: Hourly rate limit
            recent_hour = self._get_exec_count_window(symbol, hours=1)
            if recent_hour >= self.config.MAX_EXEC_PER_HOUR:
                self._block_plan(plan_id, symbol, 'hourly_limit_exceeded')
                return
            
            # Gate 3: Burst protection (5min)
            recent_5min = self._get_exec_count_window(symbol, minutes=5)
            if recent_5min >= self.config.MAX_EXEC_PER_5MIN:
                self._block_plan(plan_id, symbol, 'burst_limit_exceeded')
                
                # Trigger auto-disarm if configured
                if self.config.ENABLE_AUTO_DISARM and self.config.DISARM_ON_BURST_BREACH:
                    self._trigger_disarm('burst_limit_breach', {
                        'symbol': symbol,
                        'exec_5min': recent_5min,
                        'limit': self.config.MAX_EXEC_PER_5MIN
                    })
                return
            
            # COMPUTE REAL CLOSE_QTY AND NOTIONAL
            computed_qty = 0.0
            computed_notional = 0.0
            
            if self.binance_client:
                try:
                    # Fetch current position
                    position = self.binance_client.get_position(symbol)
                    if not position:
                        logger.warning(f"{symbol}: Could not fetch position, blocking")
                        self._block_plan(plan_id, symbol, 'position_fetch_failed')
                        return
                    
                    pos_amt = abs(position['positionAmt'])
                    
                    # Calculate close qty based on action
                    if action == 'FULL_CLOSE_PROPOSED':
                        computed_qty = pos_amt
                    elif action == 'PARTIAL_75':
                        computed_qty = pos_amt * 0.75
                    elif action == 'PARTIAL_50':
                        computed_qty = pos_amt * 0.50
                    else:
                        computed_qty = pos_amt * 0.75  # Default fallback
                    
                    # Fetch mark price
                    mark_price = self.binance_client.get_mark_price(symbol)
                    if mark_price and mark_price > 0:
                        computed_notional = computed_qty * mark_price
                    else:
                        logger.warning(f"{symbol}: Could not fetch price, using qty limit only")
                    
                    logger.info(f"{symbol}: Computed qty={computed_qty:.4f}, notional=${computed_notional:.2f}")
                    
                except Exception as e:
                    logger.error(f"{symbol}: Error computing limits: {e}")
                    self._block_plan(plan_id, symbol, 'limit_computation_error')
                    return
            else:
                # No Binance client - use conservative fallback
                logger.warning(f"{symbol}: No Binance client, using fallback limits")
                computed_qty = 0.01  # Conservative fallback
                computed_notional = 0.0
            
            # Gate 4: Daily notional/qty limit
            if not self._check_daily_limit(symbol, computed_qty, computed_notional if computed_notional > 0 else None):
                self._block_plan(plan_id, symbol, 'daily_limit_exceeded')
                return
            
            # All gates passed - issue permit with computed values
            self._issue_permit(plan_id, symbol, computed_qty, computed_notional)
            
        except Exception as e:
            logger.error(f"Error evaluating plan {plan_id}: {e}", exc_info=True)
            self.error_count += 1
            # Fail closed: do not issue permit
            self._block_plan(plan_id, symbol, 'evaluation_error')
            
        except Exception as e:
            logger.error(f"Error evaluating plan {plan_id}: {e}", exc_info=True)
            self.error_count += 1
            # Fail closed: do not issue permit
            self._block_plan(plan_id, symbol, 'evaluation_error')
    
    def _compute_open_threshold(self, kill_score: float, confidence: float = 0.5, regime_flip: float = 0.0) -> float:
        """
        Compute dynamic threshold for OPEN actions.
        
        Args:
            kill_score: Current kill_score
            confidence: AI confidence (0-1)
            regime_flip: Regime flip penalty (0-1)
        
        Returns:
            Threshold for OPEN (higher = more permissive)
        """
        # Base threshold from config
        base_threshold = self.config.KILL_SCORE_OPEN_THRESHOLD
        
        # Adjust for regime flip: tighten threshold during regime change
        regime_adjustment = regime_flip * 0.15  # Max -0.15 during full regime flip
        
        # Adjust for confidence: higher confidence = more permissive
        confidence_adjustment = (confidence - 0.5) * 0.1  # ±0.05 adjustment
        
        threshold = base_threshold - regime_adjustment + confidence_adjustment
        
        # Clamp to reasonable range [0.6, 0.95]
        return max(0.6, min(0.95, threshold))
    
    def _compute_close_threshold(self, kill_score: float, unrealized_pnl_pct: float = 0.0) -> float:
        """
        Compute dynamic threshold for CLOSE actions.
        
        Args:
            kill_score: Current kill_score
            unrealized_pnl_pct: Unrealized PnL as percentage (e.g., -0.05 for -5%)
        
        Returns:
            Threshold for CLOSE (lower = stricter, blocks exits sooner)
        """
        # Base threshold from config
        base_threshold = self.config.KILL_SCORE_CLOSE_THRESHOLD
        
        # If losing position: allow exit even with higher kill_score
        if unrealized_pnl_pct < -0.03:  # Losing > 3%
            base_threshold += 0.2  # More permissive for loss-cutting
        
        # Clamp to reasonable range [0.5, 0.85]
        return max(0.5, min(0.85, base_threshold))
    
    def _qty_scale_from_kill(self, kill_score: float, threshold: float) -> float:
        """
        Scale down quantity based on kill_score exceeding threshold.
        
        Uses exponential decay: qty_final = qty_raw * exp(-alpha * excess)
        
        Args:
            kill_score: Current kill_score
            threshold: Threshold for this action type
        
        Returns:
            Scale factor [MIN_SCALE, 1.0]
        """
        if kill_score <= threshold:
            return 1.0  # No scaling needed
        
        excess = kill_score - threshold
        alpha = self.config.KILL_SCORE_OPEN_QTY_SCALE_ALPHA
        min_scale = self.config.KILL_SCORE_OPEN_MIN_SCALE
        
        import math
        scale = math.exp(-alpha * excess)
        
        # Clamp to [min_scale, 1.0]
        return max(min_scale, min(1.0, scale))
    
    def _block_plan(self, plan_id, symbol, reason):
        """Block a plan (no permit issued)"""
        logger.warning(f"{symbol}: BLOCKED plan {plan_id[:8]} - {reason}")
        METRIC_BLOCK.labels(symbol=symbol, reason=reason).inc()
        
        # Store block record
        block_key = f"quantum:governor:block:{plan_id}"
        self.redis.setex(block_key, 3600, json.dumps({
            'reason': reason,
            'timestamp': time.time(),
            'symbol': symbol
        }))
    
    def _issue_permit(self, plan_id, symbol, computed_qty, computed_notional):
        """
        Issue single-use execution permit with computed values (60s TTL).
        
        Also reads P3.1 Capital Efficiency and applies downsize factor hint to permit.
        Never blocks (fail-open), only modifies caps/cooldown as hint.
        """
        # Read P3.1 efficiency data
        eff_score, eff_confidence, eff_stale, eff_reason = self._read_p31_efficiency(symbol)
        
        # Compute downsize factor (never hard-block, only hint)
        eff_action = "NONE"
        eff_factor = 1.0
        extra_cooldown_sec = 0
        
        if eff_score is not None and eff_score < self.config.P31_DOWNSIZE_THRESHOLD:
            # Low efficiency = apply downsize factor
            eff_action = "DOWNSIZE"
            eff_factor = max(self.config.P31_MIN_FACTOR, eff_score / self.config.P31_DOWNSIZE_THRESHOLD)
            extra_cooldown_sec = round((1 - eff_score) * self.config.P31_MAX_EXTRA_COOLDOWN_SEC)
            
            # Update metrics
            METRIC_EFF_APPLY.labels(action="DOWNSIZE", reason="low_score").inc()
            METRIC_EFF_FACTOR.labels(symbol=symbol).set(eff_factor)
            
            logger.info(
                f"{symbol}: P3.1 DOWNSIZE - score={eff_score:.3f} < threshold={self.config.P31_DOWNSIZE_THRESHOLD} "
                f"→ factor={eff_factor:.3f} extra_cooldown={extra_cooldown_sec}s"
            )
        else:
            # Efficiency OK or missing = no downsize
            if eff_score is not None:
                logger.info(
                    f"{symbol}: P3.1 OK - score={eff_score:.3f} >= threshold={self.config.P31_DOWNSIZE_THRESHOLD}, "
                    f"factor={eff_factor:.3f}"
                )
            else:
                logger.info(f"{symbol}: P3.1 data unavailable - {eff_reason}, factor={eff_factor:.3f} (no-op)")
            
            METRIC_EFF_APPLY.labels(action="NONE", reason=eff_reason).inc()
            METRIC_EFF_FACTOR.labels(symbol=symbol).set(eff_factor)
        
        logger.info(
            f"{symbol}: ALLOW plan {plan_id[:8]} "
            f"(qty={computed_qty:.4f}, notional=${computed_notional:.2f}) "
            f"P3.1: action={eff_action} factor={eff_factor:.3f}"
        )
        
        # Write permit key with 60s TTL (single-use, race-safe)
        permit_key = f"quantum:permit:{plan_id}"
        permit_data = {
            'granted': True,
            'symbol': symbol,
            'decision': 'EXECUTE',
            'computed_qty': computed_qty,
            'computed_notional': computed_notional,
            'created_at': time.time(),
            'consumed': False,
            # P3.1 efficiency fields (always present, safe for downstream)
            'eff_score': f"{eff_score:.4f}" if eff_score is not None else "",
            'eff_confidence': f"{eff_confidence:.3f}" if eff_confidence is not None else "",
            'eff_stale': "1" if eff_stale else "0",
            'eff_factor': f"{eff_factor:.4f}",
            'eff_action': eff_action,
            'eff_reason': eff_reason,
            # Downsize hints (for Apply Layer to use)
            'downsize_factor': f"{eff_factor:.4f}",
            'extra_cooldown_sec': str(extra_cooldown_sec)
        }
        self.redis.setex(permit_key, 60, json.dumps(permit_data))  # 60s TTL
        
        # Track execution (assume will execute - apply layer confirms later)
        self.exec_history[symbol].append(time.time())
        self._trim_exec_history(symbol)
        
        METRIC_ALLOW.labels(symbol=symbol).inc()
        
        # Store in Redis for persistence across restarts
        exec_key = f"quantum:governor:exec:{symbol}"
        self.redis.lpush(exec_key, str(time.time()))
        self.redis.ltrim(exec_key, 0, 99)  # Keep last 100
        self.redis.expire(exec_key, 86400)  # 24h
    
    def _get_exec_count_window(self, symbol, hours=0, minutes=0):
        """Count executions in time window"""
        now = time.time()
        window = hours * 3600 + minutes * 60
        cutoff = now - window
        
        # Load from Redis if in-memory cache is empty
        if not self.exec_history[symbol]:
            exec_key = f"quantum:governor:exec:{symbol}"
            timestamps = self.redis.lrange(exec_key, 0, -1)
            self.exec_history[symbol] = [float(ts) for ts in timestamps if float(ts) > cutoff]
        
        # Count recent
        recent = [ts for ts in self.exec_history[symbol] if ts > cutoff]
        return len(recent)
    
    def _check_portfolio_budget(self, symbol: str, plan_id: str) -> bool:
        """
        Check P2.8 Portfolio Budget violation.
        
        Reads quantum:portfolio:budget:{symbol} hash from Portfolio Risk Governor.
        If enforce mode AND budget violation exists, returns True (block permit).
        
        Returns:
            True if violation (should block)
            False if OK (allow)
        """
        try:
            logger.info(f"{symbol}: Checking P2.8 portfolio budget (plan_id={plan_id[:8]})")
            
            # Read budget hash
            budget_key = f"quantum:portfolio:budget:{symbol}"
            budget_data = self.redis.hgetall(budget_key)
            
            if not budget_data:
                # No budget data = fail-open (P2.8 might not be running)
                logger.info(f"{symbol}: No P2.8 budget hash found - fail-open (allow)")
                return False
            
            # Decode (handle both bytes and strings)
            decoded = {}
            for k, v in budget_data.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                decoded[key] = val
            
            # Check mode
            p28_mode = decoded.get('mode', 'shadow')
            budget_usd = float(decoded.get('budget_usd', 0))
            stress_factor = float(decoded.get('stress_factor', 0))
            
            if p28_mode != 'enforce':
                # Shadow mode = don't block
                logger.info(
                    f"{symbol}: P2.8 budget=${budget_usd:.0f} stress={stress_factor:.3f} "
                    f"mode={p28_mode} - allowing (shadow mode)"
                )
                return False
            
            # Check stale data (fail-open if stale)
            timestamp = int(decoded.get('timestamp', 0))
            age = time.time() - timestamp
            if age > 60:
                logger.warning(f"{symbol}: P2.8 budget data stale ({age:.0f}s), fail-open")
                return False
            
            # Get position notional from plan data or current position
            # For now, check if we're attempting to increase position size
            # In future, could fetch exact notional from plan metadata
            
            # Read budget value
            budget_usd = float(decoded.get('budget_usd', 0))
            stress_factor = float(decoded.get('stress_factor', 0))
            
            logger.info(
                f"{symbol}: P2.8 budget check - "
                f"budget=${budget_usd:.0f} stress={stress_factor:.3f} mode={p28_mode}"
            )
            
            # For now, don't block (until we have position notional in plan data)
            # This gate serves as integration point - actual violation detection
            # happens in P2.8 service which publishes to budget.violation stream
            
            # Check if violation event exists in stream
            stream_key = "quantum:stream:budget.violation"
            recent_events = self.redis.xrevrange(stream_key, count=10)
            
            for event_id, event_data in recent_events:
                event_json = json.loads(event_data.get(b'json', b'{}'))
                event_symbol = event_json.get('symbol')
                event_ts = event_json.get('timestamp', 0)
                
                # Check if violation is for this symbol and recent (< 30s)
                if event_symbol == symbol and (time.time() - event_ts) < 30:
                    logger.warning(
                        f"{symbol}: P2.8 budget violation detected - "
                        f"over_budget=${event_json.get('over_budget', 0):.0f}"
                    )
                    return True  # Block
            
            return False  # Allow
            
        except Exception as e:
            logger.error(f"{symbol}: Error checking P2.8 budget: {e}")
            # Fail-open on errors
            return False
    
    def _check_p29_allocation_target(self, symbol: str, plan_id: str, position_notional_usd: float = None) -> tuple:
        """
        Gate 0.5: Check P2.9 Capital Allocation Target.
        
        Reads quantum:allocation:target:{symbol} hash from Capital Allocation Brain.
        If enforce mode AND requested notional > target, returns (True, reason) to block.
        
        Fail-open: missing/stale target → allow execution.
        
        Args:
            symbol: Trading symbol
            plan_id: Plan ID for logging
            position_notional_usd: Current position notional (optional, for logging)
        
        Returns:
            (should_block: bool, reason: str)
        """
        try:
            METRIC_P29_CHECKED.labels(symbol=symbol).inc()
            
            # Read allocation target hash
            target_key = f"quantum:allocation:target:{symbol}"
            target_data = self.redis.hgetall(target_key)
            
            if not target_data:
                # No target data = fail-open (P2.9 might not be running or in shadow)
                logger.info(f"{symbol}: No P2.9 allocation target found - fail-open (allow)")
                METRIC_P29_MISSING.labels(symbol=symbol).inc()
                return False, ""
            
            # Decode data
            decoded = {}
            for k, v in target_data.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                decoded[key] = val
            
            # Extract fields
            target_usd = float(decoded.get('target_usd', 0))
            mode = decoded.get('mode', 'shadow')
            timestamp = int(decoded.get('timestamp', 0))
            confidence = float(decoded.get('confidence', 0))
            
            # Check stale data (fail-open if >300s old)
            age = time.time() - timestamp
            if age > 300:
                logger.warning(f"{symbol}: P2.9 allocation target stale ({age:.0f}s) - fail-open (allow)")
                METRIC_P29_STALE.labels(symbol=symbol).inc()
                return False, ""
            
            # Shadow mode = log only, don't block
            if mode == 'shadow':
                logger.info(
                    f"{symbol}: P2.9 allocation target=${target_usd:.2f} "
                    f"mode={mode} conf={confidence:.3f} - allowing (shadow mode)"
                )
                return False, ""
            
            # Enforce mode: need position notional to compare
            if position_notional_usd is None:
                # If position not computed yet, fetch it
                try:
                    if self.binance_client:
                        position = self.binance_client.get_position(symbol)
                        if position and 'positionAmt' in position:
                            # Compute notional from position amount × mark price
                            mark_price = self.binance_client.get_mark_price(symbol)
                            if mark_price:
                                position_notional_usd = abs(float(position['positionAmt']) * mark_price)
                            else:
                                logger.warning(f"{symbol}: Could not fetch mark price for P2.9 check - fail-open")
                                return False, ""
                        else:
                            logger.warning(f"{symbol}: Could not fetch position for P2.9 check - fail-open")
                            return False, ""
                    else:
                        logger.warning(f"{symbol}: No Binance client for P2.9 position check - fail-open")
                        return False, ""
                except Exception as e:
                    logger.error(f"{symbol}: Error fetching position for P2.9: {e}")
                    return False, ""
            
            # Check if position notional exceeds allocation target
            if position_notional_usd > target_usd:
                logger.warning(
                    f"{symbol}: P2.9 ALLOCATION CAP - "
                    f"position=${position_notional_usd:.2f} > target=${target_usd:.2f} "
                    f"(mode={mode}, conf={confidence:.3f})"
                )
                METRIC_P29_BLOCK.labels(symbol=symbol).inc()
                
                # Publish event to stream
                self.redis.xadd(self.config.STREAM_EVENTS, {
                    'event': 'P29_ALLOCATION_CAP_BLOCK',
                    'symbol': symbol,
                    'plan_id': plan_id,
                    'position_notional_usd': str(position_notional_usd),
                    'target_usd': str(target_usd),
                    'mode': mode,
                    'confidence': str(confidence),
                    'timestamp': str(time.time())
                })
                
                return True, 'p29_allocation_cap'
            
            # Within allocation target
            logger.info(
                f"{symbol}: P2.9 allocation check passed - "
                f"position=${position_notional_usd:.2f} <= target=${target_usd:.2f} "
                f"(mode={mode}, conf={confidence:.3f})"
            )
            return False, ""
            
        except Exception as e:
            logger.error(f"{symbol}: Error checking P2.9 allocation: {e}")
            # Fail-open on errors
            return False, ""
    
    def _read_p31_efficiency(self, symbol: str) -> Tuple[Optional[float], Optional[float], bool, str]:
        """
        Read P3.1 Capital Efficiency data for downsize hint.
        
        Returns:
            (score: float, confidence: float, stale: bool, reason: str)
        
        Fail-open: missing/stale → returns (None, None, True, reason)
        Success: returns (score, confidence, False, "ok")
        """
        try:
            key = f"quantum:capital:efficiency:{symbol}"
            data = self.redis.hgetall(key)
            
            if not data:
                # Missing efficiency data = fail-open
                METRIC_EFF_APPLY.labels(action="NONE", reason="missing_eff").inc()
                return None, None, True, "missing_eff"
            
            # Decode data
            decoded = {}
            for k, v in data.items():
                key_str = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                decoded[key_str] = val
            
            # Extract fields
            score = float(decoded.get('efficiency_score', 0))
            confidence = float(decoded.get('confidence', 0))
            timestamp = int(decoded.get('ts', 0))
            
            # Check staleness (fail-open if too old)
            age = time.time() - timestamp
            if age > self.config.P31_EFF_TTL_SEC:
                METRIC_EFF_APPLY.labels(action="NONE", reason="stale_eff").inc()
                return None, None, True, "stale_eff"
            
            # Check confidence (fail-open if too low)
            if confidence < self.config.P31_MIN_CONF:
                METRIC_EFF_APPLY.labels(action="NONE", reason="low_conf").inc()
                return None, None, True, "low_conf"
            
            # Valid efficiency data
            return score, confidence, False, "ok"
        
        except Exception as e:
            logger.error(f"{symbol}: Error reading P3.1 efficiency: {e}")
            METRIC_EFF_APPLY.labels(action="NONE", reason="redis_error").inc()
            # Fail-open on errors
            return None, None, True, "redis_error"
    
    def _trim_exec_history(self, symbol):
        """Remove old timestamps from in-memory cache"""
        cutoff = time.time() - 3600  # Keep last hour in memory
        self.exec_history[symbol] = [ts for ts in self.exec_history[symbol] if ts > cutoff]
    
    def _check_daily_limit(self, symbol, qty, notional=None):
        """Check daily notional or quantity limit"""
        # Get today's executions
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"quantum:governor:daily:{symbol}:{today}"
        
        current_total = float(self.redis.get(daily_key) or 0)
        
        # Calculate new total
        if notional and notional > 0:
            # Notional-based (preferred)
            new_total = current_total + notional
            limit = self.config.MAX_REDUCE_NOTIONAL_PER_DAY_USD
            if new_total > limit:
                logger.warning(f"{symbol}: Daily notional limit exceeded ({new_total:.2f} > {limit})")
                return False
            # Update
            self.redis.setex(daily_key, 86400, str(new_total))
        else:
            # Qty-based fallback
            new_total = current_total + qty
            limit = self.config.MAX_REDUCE_QTY_PER_DAY
            if new_total > limit:
                logger.warning(f"{symbol}: Daily qty limit exceeded ({new_total:.4f} > {limit})")
                return False
            # Update
            self.redis.setex(daily_key, 86400, str(new_total))
        
        return True
    
    def _trigger_disarm(self, reason, context):
        """Force Apply Layer to dry_run mode (auto-disarm)"""
        # Idempotency: check if already disarmed today
        today = datetime.utcnow().strftime('%Y-%m-%d')
        disarm_key = f"quantum:governor:disarm:{today}"
        
        if self.redis.exists(disarm_key):
            logger.info(f"Disarm already triggered today ({reason}), skipping")
            return
        
        logger.critical(f"AUTO-DISARM TRIGGERED: {reason}")
        logger.info(f"Context: {json.dumps(context)}")
        
        try:
            # Method: Set APPLY_MODE=dry_run in config
            config_path = self.config.APPLY_CONFIG_PATH
            
            # Backup config
            backup_path = f"{config_path}.bak.{int(time.time())}"
            subprocess.run(['cp', config_path, backup_path], check=True)
            logger.info(f"Backed up config to {backup_path}")
            
            # Update config
            subprocess.run([
                'sed', '-i', 's/^APPLY_MODE=.*/APPLY_MODE=dry_run/',
                config_path
            ], check=True)
            logger.info(f"Set APPLY_MODE=dry_run in {config_path}")
            
            # Restart Apply Layer
            subprocess.run(['systemctl', 'restart', 'quantum-apply-layer'], check=True)
            logger.info("Restarted quantum-apply-layer.service")
            
            # Mark disarm done (24h TTL)
            self.redis.setex(disarm_key, 86400, json.dumps({
                'reason': reason,
                'context': context,
                'timestamp': time.time()
            }))
            
            # Emit event to stream
            self.redis.xadd(self.config.STREAM_EVENTS, {
                'event': 'AUTO_DISARM',
                'reason': reason,
                'context': json.dumps(context),
                'timestamp': time.time(),
                'action_taken': 'APPLY_MODE=dry_run + restart'
            })
            
            METRIC_DISARM.labels(reason=reason).inc()
            logger.critical("AUTO-DISARM COMPLETE - System is now in dry_run mode")
            
        except Exception as e:
            logger.error(f"Failed to execute disarm: {e}", exc_info=True)
            # Still mark as attempted to avoid spam
            self.redis.setex(disarm_key, 3600, json.dumps({
                'reason': reason,
                'error': str(e),
                'timestamp': time.time()
            }))
    
    def _check_testnet_flatten_arm(self):
        """Check if testnet flatten is armed via Redis key"""
        try:
            arm_key = "quantum:gov:testnet:flatten:arm"
            arm_value = self.redis.get(arm_key)
            
            if arm_value == "1":
                logger.info("Testnet flatten arm key detected - attempting flatten")
                self.redis.delete(arm_key)  # Delete immediately to prevent re-trigger
                self._testnet_flatten_if_armed()
        except Exception as e:
            logger.error(f"Error checking flatten arm key: {e}")
    
    def _testnet_flatten_if_armed(self):
        """
        TESTNET ONLY: Flatten all open positions if all safety conditions met.
        
        Safety requirements:
        1. Config flags: TESTNET_FORCE_FLATTEN=true + TESTNET_FORCE_FLATTEN_CONFIRM=FLATTEN_NOW
        2. ESS active: /var/run/quantum/ESS_ON exists
        3. Rate limit: max 1 flatten per 60s
        
        Fail-open: errors do NOT crash Governor, just log and count.
        """
        METRIC_TESTNET_FLATTEN_ATTEMPT.inc()
        
        try:
            # Check 1: Config flags
            if not self.config.TESTNET_FORCE_FLATTEN:
                logger.warning("Testnet flatten: GOV_TESTNET_FORCE_FLATTEN not true - NO-OP")
                METRIC_TESTNET_FLATTEN_NOOP.inc()
                return
            
            if self.config.TESTNET_FORCE_FLATTEN_CONFIRM != "FLATTEN_NOW":
                logger.warning(f"Testnet flatten: CONFIRM not 'FLATTEN_NOW' (got '{self.config.TESTNET_FORCE_FLATTEN_CONFIRM}') - NO-OP")
                METRIC_TESTNET_FLATTEN_NOOP.inc()
                return
            
            # Check 2: ESS active latch file
            ess_latch = "/var/run/quantum/ESS_ON"
            if not os.path.exists(ess_latch):
                logger.warning(f"Testnet flatten: ESS latch file {ess_latch} not found - NO-OP")
                METRIC_TESTNET_FLATTEN_NOOP.inc()
                return
            
            # Check 3: Rate limit (cooldown)
            cooldown_key = "quantum:gov:testnet:flatten:last_ts"
            last_ts_str = self.redis.get(cooldown_key)
            
            if last_ts_str:
                last_ts = float(last_ts_str)
                elapsed = time.time() - last_ts
                if elapsed < 60:
                    logger.warning(f"Testnet flatten: Cooldown active ({elapsed:.1f}s < 60s) - NO-OP")
                    METRIC_TESTNET_FLATTEN_NOOP.inc()
                    return
            
            # All checks passed - proceed with flatten
            logger.warning("Testnet flatten: ALL SAFETY CHECKS PASSED - executing flatten")
            
            if not self.binance_client:
                logger.error("Testnet flatten: No Binance client available")
                METRIC_TESTNET_FLATTEN_ERRORS.inc()
                return
            
            # Fetch all open positions from exchange
            try:
                result = self.binance_client._request('GET', '/fapi/v2/positionRisk', signed=True)
            except Exception as e:
                logger.error(f"Testnet flatten: Error fetching positions: {e}")
                METRIC_TESTNET_FLATTEN_ERRORS.inc()
                return
            
            if not result:
                logger.info("Testnet flatten: No positions found")
                return
            
            # Fetch mark prices for notional calculation
            try:
                mark_prices_raw = self.binance_client.futures_mark_price()
                mark_prices = {item['symbol']: float(item['markPrice']) for item in mark_prices_raw}
            except Exception as e:
                logger.warning(f"Testnet flatten: Could not fetch mark prices: {e} (will include all positions)")
                mark_prices = {}
            
            # Filter for positions with non-zero qty (float-safe threshold)
            symbols_seen = len(result)
            symbols_nonzero = 0
            open_positions = []
            for pos in result:
                symbol = pos.get('symbol')
                qty = float(pos.get('positionAmt', 0))
                
                if abs(qty) > 1e-8:  # Float-safe threshold (Binance min notional ~$5)
                    symbols_nonzero += 1
                    
                    # Check notional if mark price available
                    mark_price = mark_prices.get(symbol, 0)
                    if mark_price > 0:
                        notional = abs(qty) * mark_price
                        if notional < self.config.MIN_FLATTEN_NOTIONAL_USD:
                            logger.info(f"Testnet flatten: Skipping {symbol} dust (${notional:.2f} < ${self.config.MIN_FLATTEN_NOTIONAL_USD})")
                            continue
                    
                    open_positions.append({
                        'symbol': symbol,
                        'positionAmt': qty,
                        'positionSide': pos.get('positionSide', 'BOTH')
                    })
            
            # Hard cap check (API glitch guard)
            if len(open_positions) > self.config.MAX_FLATTEN_SYMBOLS:
                logger.error(f"Testnet flatten: SAFETY CAP TRIGGERED - found {len(open_positions)} positions (max {self.config.MAX_FLATTEN_SYMBOLS})")
                logger.error("Testnet flatten: This may indicate API glitch or universe expansion - ABORTING")
                METRIC_TESTNET_FLATTEN_ERRORS.inc()
                return
            
            if not open_positions:
                logger.info(f"Testnet flatten: No positions to close (seen={symbols_seen} nonzero={symbols_nonzero} above_notional=0)")
                return
            
            logger.warning(f"Testnet flatten: Found {len(open_positions)} positions to close (seen={symbols_seen} nonzero={symbols_nonzero})")
            logger.warning("Testnet flatten: Placing reduceOnly MARKET close orders...")
            
            orders_placed = 0
            errors = 0
            
            for pos in open_positions:
                symbol = pos['symbol']
                qty = abs(pos['positionAmt'])
                side = 'SELL' if pos['positionAmt'] > 0 else 'BUY'  # Opposite side to close
                
                try:
                    # Place reduceOnly MARKET order
                    order_params = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'MARKET',
                        'quantity': qty,
                        'reduceOnly': 'true'
                    }
                    
                    logger.info(f"Testnet flatten: Closing {symbol} {side} qty={qty}")
                    close_result = self.binance_client._request('POST', '/fapi/v1/order', params=order_params, signed=True)
                    
                    if close_result:
                        logger.info(f"Testnet flatten: {symbol} close order placed: {close_result.get('orderId')}")
                        orders_placed += 1
                        METRIC_TESTNET_FLATTEN_ORDERS.inc()
                    else:
                        logger.error(f"Testnet flatten: {symbol} close order failed (no result)")
                        errors += 1
                        METRIC_TESTNET_FLATTEN_ERRORS.inc()
                
                except Exception as e:
                    logger.error(f"Testnet flatten: {symbol} close order error: {e}")
                    errors += 1
                    METRIC_TESTNET_FLATTEN_ERRORS.inc()
                
                # Small delay between orders
                time.sleep(0.1)
            
            # Write cooldown timestamp
            self.redis.set(cooldown_key, str(time.time()), ex=3600)  # 1h expiry
            
            logger.warning(f"TESTNET_FLATTEN done: symbols_seen={symbols_seen} nonzero={symbols_nonzero} above_notional={len(open_positions)} orders_sent={orders_placed} orders_ok={orders_placed-errors} errors={errors}")
            
        except Exception as e:
            logger.error(f"Testnet flatten: Fatal error: {e}", exc_info=True)
            METRIC_TESTNET_FLATTEN_ERRORS.inc()
    
    def _check_auto_disarm(self):
        """Periodic check for auto-disarm conditions"""
        if not self.config.ENABLE_AUTO_DISARM:
            return
        
        now = time.time()
        if now - self.last_disarm_check < 60:  # Check every 60s
            return
        
        self.last_disarm_check = now
        
        # Check error count
        if self.error_count >= self.config.DISARM_ON_ERROR_COUNT:
            self._trigger_disarm('repeated_errors', {
                'error_count': self.error_count,
                'threshold': self.config.DISARM_ON_ERROR_COUNT
            })
            self.error_count = 0  # Reset after disarm
    
    def _update_metrics(self):
        """Update Prometheus metrics"""
        for symbol in self.exec_history.keys():
            count_hour = self._get_exec_count_window(symbol, hours=1)
            count_5min = self._get_exec_count_window(symbol, minutes=5)
            METRIC_EXEC_HOUR.labels(symbol=symbol).set(count_hour)
            METRIC_EXEC_5MIN.labels(symbol=symbol).set(count_5min)
    
    # ========================================================================
    # ACTIVE SLOTS CONTROLLER (strict rotation with close confirmation)
    # ========================================================================
    
    def _get_open_positions_snapshot(self):
        """
        Fetch open positions from Binance, compute weakness scores.
        
        Weakness = unrealized_pnl_pct + age_penalty (higher = weaker)
        
        Returns:
            List[dict]: [{symbol, qty, pnl_pct, entry_ts, weakness_score}]
        """
        try:
            if not self.binance_client:
                logger.warning("No Binance client - cannot fetch positions")
                return []
            
            # Fetch positions from Binance
            account_data = self.binance_client._request('GET', '/fapi/v2/positionRisk', signed=True)
            if not account_data:
                logger.warning("No position data from Binance")
                return []
            
            positions = []
            for pos in account_data:
                amt = abs(float(pos.get('positionAmt', 0)))
                if amt < 1e-8:
                    continue  # Skip empty positions
                
                symbol = pos.get('symbol')
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                notional = amt * mark_price
                
                # Compute PNL %
                pnl_pct = unrealized_pnl / notional if notional > 0 else 0
                
                # Age penalty: positions > 24h get +0.05 weakness
                # (encourage rotation of stale positions)
                update_time = int(pos.get('updateTime', 0))
                age_hours = (time.time() * 1000 - update_time) / 3600000 if update_time > 0 else 0
                age_penalty = 0.05 if age_hours > 24 else 0
                
                # Weakness score = -(pnl_pct) + age_penalty
                # Negative PNL → positive weakness (bad)
                # Positive PNL → negative weakness (good)
                weakness_score = -pnl_pct + age_penalty
                
                positions.append({
                    'symbol': symbol,
                    'qty': amt,
                    'pnl_pct': pnl_pct,
                    'entry_ts': update_time / 1000 if update_time > 0 else 0,
                    'weakness_score': weakness_score,
                    'notional': notional
                })
            
            logger.info(f"Fetched {len(positions)} open positions from Binance")
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching position snapshot: {e}")
            return []
    
    def _detect_market_regime(self, symbol):
        """
        Detect market regime from klines (TREND_STRONG, CHOP, BASE).
        
        Returns:
            (regime: str, desired_slots: int)
        """
        try:
            if not self.binance_client:
                logger.warning(f"{symbol}: No Binance client for regime detection - defaulting to BASE(4)")
                return 'BASE', self.config.ACTIVE_SLOTS_BASE
            
            # Fetch 100x 1h klines
            klines = self.binance_client._request('GET', '/fapi/v1/klines', signed=False, params={
                'symbol': symbol,
                'interval': '1h',
                'limit': 100
            })
            
            if not klines or len(klines) < 50:
                logger.warning(f"{symbol}: Insufficient klines for regime - defaulting to BASE(4)")
                return 'BASE', self.config.ACTIVE_SLOTS_BASE
            
            # Extract closes
            closes = [float(k[4]) for k in klines]  # Index 4 = close
            
            # Compute EMA20 trend strength
            import numpy as np
            ema20 = np.mean(closes[-20:])
            ema50 = np.mean(closes[-50:])
            trend_strength = abs(ema20 - ema50) / ema50
            
            # Compute ATR percentage (volatility)
            highs = [float(k[2]) for k in klines[-20:]]  # Index 2 = high
            lows = [float(k[3]) for k in klines[-20:]]   # Index 3 = low
            atr = np.mean([h - l for h, l in zip(highs, lows)])
            atr_pct = atr / ema20 if ema20 > 0 else 0
            
            # Classify regime
            if trend_strength > 0.05 and atr_pct < 0.02:
                # Strong trend, low volatility → TREND_STRONG
                regime = 'TREND_STRONG'
                slots = self.config.ACTIVE_SLOTS_TREND_STRONG
            elif atr_pct > 0.03 or trend_strength < 0.01:
                # High volatility or no trend → CHOP
                regime = 'CHOP'
                slots = self.config.ACTIVE_SLOTS_CHOP
            else:
                # Default → BASE
                regime = 'BASE'
                slots = self.config.ACTIVE_SLOTS_BASE
            
            logger.info(f"{symbol}: Regime={regime} (trend={trend_strength:.4f}, atr_pct={atr_pct:.4f}) → {slots} slots")
            return regime, slots
            
        except Exception as e:
            logger.error(f"{symbol}: Error detecting regime: {e}")
            return 'BASE', self.config.ACTIVE_SLOTS_BASE
    
    def _is_close_confirmed(self, close_plan_id):
        """
        Check if close plan was executed (from apply.result stream).
        
        Returns:
            bool: True if close confirmed with executed=true + reduceOnly=true
        """
        try:
            # Search apply.result stream for close plan confirmation
            results = self.redis.xrevrange(self.config.STREAM_RESULTS, count=50)
            
            for msg_id, data in results:
                result_plan_id = data.get(b'plan_id', b'').decode()
                if result_plan_id == close_plan_id:
                    executed = data.get(b'executed', b'false').decode().lower() == 'true'
                    reduce_only = data.get(b'reduceOnly', b'false').decode().lower() == 'true'
                    
                    if executed and reduce_only:
                        logger.info(f"Close confirmed: plan_id={close_plan_id[:8]} (executed=true, reduceOnly=true)")
                        return True
                    else:
                        logger.warning(f"Close plan {close_plan_id[:8]} found but NOT confirmed: executed={executed}, reduceOnly={reduce_only}")
                        return False
            
            # Plan not found in results (yet)
            logger.debug(f"Close plan {close_plan_id[:8]} not found in apply.result stream (yet)")
            return False
            
        except Exception as e:
            logger.error(f"Error checking close confirmation: {e}")
            return False

# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=== P3.2 GOVERNOR STARTING ===")
    logger.info(f"Metrics port: {Config.METRICS_PORT}")
    
    # Start Prometheus metrics server
    start_http_server(Config.METRICS_PORT)
    logger.info(f"Metrics server started on port {Config.METRICS_PORT}")
    
    # Initialize Governor
    governor = Governor(Config)
    
    # Run main loop
    try:
        governor.run()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Governor stopped")

if __name__ == '__main__':
    main()
