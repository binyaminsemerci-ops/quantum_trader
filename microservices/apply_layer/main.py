#!/usr/bin/env python3
"""
P3 Apply Layer - Harvest Proposal Consumer

Reads harvest proposals from Redis, creates apply plans, and optionally executes them.

Modes:
- dry_run (P3.0): Plans published, no execution
- testnet (P3.1): Plans executed against Binance testnet

Safety gates:
- Allowlist (default: BTCUSDT only)
- Kill score thresholds (>=0.8 block all, >=0.6 block risk increase)
- Idempotency (Redis SETNX dedupe)
- Kill switch (emergency stop)
"""

import os
import sys
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Install with: pip install redis")
    sys.exit(1)

# Binance client
try:
    import hmac
    import hashlib
    import urllib.request
    import urllib.parse
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("WARN: urllib not available, testnet execution disabled")

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("WARN: prometheus_client not installed, metrics disabled")


logging.basicConfig(
    level=os.getenv("APPLY_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ApplyMode(Enum):
    DRY_RUN = "dry_run"
    TESTNET = "testnet"


class Decision(Enum):
    EXECUTE = "EXECUTE"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class BinanceTestnetClient:
    """Minimal Binance Futures Testnet client for reduceOnly orders"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com"
        self.exchange_info_cache = {}
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request with HMAC SHA256"""
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Binance API"""
        if params is None:
            params = {}
        
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign_request(params)
        
        url = f"{self.base_url}{endpoint}"
        if params:
            url += '?' + urllib.parse.urlencode(params)
        
        req = urllib.request.Request(url, method=method.upper())
        req.add_header('X-MBX-APIKEY', self.api_key)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logger.error(f"Binance API error: {e.code} {error_body}")
            raise Exception(f"Binance API error: {error_body}")
    
    def ping(self) -> bool:
        """Test connectivity"""
        try:
            self._request('GET', '/fapi/v1/ping')
            return True
        except Exception as e:
            logger.error(f"Binance ping failed: {e}")
            return False
    
    def get_exchange_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get exchange info for symbol (cached)"""
        if symbol in self.exchange_info_cache:
            return self.exchange_info_cache[symbol]
        
        try:
            data = self._request('GET', '/fapi/v1/exchangeInfo')
            for sym_info in data.get('symbols', []):
                if sym_info['symbol'] == symbol:
                    # Extract filters
                    filters = {}
                    for f in sym_info.get('filters', []):
                        filters[f['filterType']] = f
                    
                    info = {
                        'symbol': symbol,
                        'status': sym_info.get('status'),
                        'pricePrecision': sym_info.get('pricePrecision'),
                        'quantityPrecision': sym_info.get('quantityPrecision'),
                        'filters': filters
                    }
                    self.exchange_info_cache[symbol] = info
                    return info
            return None
        except Exception as e:
            logger.error(f"Failed to get exchange info for {symbol}: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for symbol"""
        try:
            positions = self._request('GET', '/fapi/v2/positionRisk', signed=True)
            for pos in positions:
                if pos['symbol'] == symbol:
                    return {
                        'symbol': symbol,
                        'positionAmt': float(pos.get('positionAmt', 0)),
                        'entryPrice': float(pos.get('entryPrice', 0)),
                        'unrealizedProfit': float(pos.get('unRealizedProfit', 0)),
                        'side': 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT'
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to exchange stepSize"""
        info = self.get_exchange_info(symbol)
        if not info:
            return round(quantity, 3)  # Default
        
        lot_size = info['filters'].get('LOT_SIZE', {})
        step_size = float(lot_size.get('stepSize', '0.001'))
        
        # Round down to step size
        return float(int(quantity / step_size) * step_size)
    
    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = True) -> Dict[str, Any]:
        """Place market order (reduceOnly for closes)"""
        # Round quantity
        quantity = self.round_quantity(symbol, abs(quantity))
        
        params = {
            'symbol': symbol,
            'side': side,  # BUY or SELL
            'type': 'MARKET',
            'quantity': quantity
        }
        
        if reduce_only:
            params['reduceOnly'] = 'true'
        
        try:
            result = self._request('POST', '/fapi/v1/order', params=params, signed=True)
            return {
                'orderId': result.get('orderId'),
                'symbol': result.get('symbol'),
                'side': result.get('side'),
                'quantity': result.get('origQty'),
                'executedQty': result.get('executedQty'),
                'status': result.get('status'),
                'reduceOnly': reduce_only
            }
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise


@dataclass
class ApplyPlan:
    """Deterministic plan for applying harvest proposal"""
    plan_id: str
    symbol: str
    action: str  # harvest_action from proposal
    kill_score: float
    k_components: Dict[str, float]
    new_sl_proposed: Optional[float]
    R_net: float
    last_update_epoch: int
    computed_at_utc: str
    decision: str  # EXECUTE, SKIP, BLOCKED, ERROR
    reason_codes: List[str]
    steps: List[Dict[str, Any]]  # execution steps
    close_qty: float  # P3.2: for Governor daily limits
    price: Optional[float]  # P3.2: for Governor notional limits
    timestamp: int


@dataclass
class ApplyResult:
    """Result of executing apply plan"""
    plan_id: str
    symbol: str
    decision: str
    executed: bool
    would_execute: bool  # true in dry_run mode
    steps_results: List[Dict[str, Any]]
    error: Optional[str]
    timestamp: int


class ApplyLayer:
    """Apply layer consumer - reads harvest proposals and executes plans"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        
        self.mode = ApplyMode(os.getenv("APPLY_MODE", "dry_run"))
        self.symbols = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
        self.allowlist = set(os.getenv("APPLY_ALLOWLIST", "BTCUSDT").split(","))
        self.poll_interval = int(os.getenv("APPLY_POLL_SEC", 5))
        self.dedupe_ttl = int(os.getenv("APPLY_DEDUPE_TTL_SEC", 21600))  # 6h
        
        # Safety thresholds
        self.k_block_critical = float(os.getenv("K_BLOCK_CRITICAL", 0.80))
        self.k_block_warning = float(os.getenv("K_BLOCK_WARNING", 0.60))
        self.kill_switch = os.getenv("APPLY_KILL_SWITCH", "false").lower() == "true"
        
        # Prometheus metrics
        self.metrics_port = int(os.getenv("APPLY_METRICS_PORT", 8043))
        self.setup_metrics()
        
        logger.info(f"ApplyLayer initialized:")
        logger.info(f"  Mode: {self.mode.value}")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Allowlist: {self.allowlist}")
        logger.info(f"  Poll interval: {self.poll_interval}s")
        logger.info(f"  K thresholds: critical={self.k_block_critical}, warning={self.k_block_warning}")
        logger.info(f"  Kill switch: {self.kill_switch}")
        logger.info(f"  Metrics port: {self.metrics_port}")
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metric_plan_total = Counter(
            'quantum_apply_plan_total',
            'Total apply plans created',
            ['symbol', 'decision']
        )
        self.metric_execute_total = Counter(
            'quantum_apply_execute_total',
            'Total execution attempts',
            ['symbol', 'step', 'status']
        )
        self.metric_dedupe_hits = Counter(
            'quantum_apply_dedupe_hits_total',
            'Total duplicate plan detections'
        )
        self.metric_last_success = Gauge(
            'quantum_apply_last_success_epoch',
            'Timestamp of last successful execution',
            ['symbol']
        )
        
        # Start metrics server
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics available on :{self.metrics_port}/metrics")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    def get_harvest_proposal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Read harvest proposal from Redis"""
        try:
            key = f"quantum:harvest:proposal:{symbol}"
            data = self.redis.hgetall(key)
            if not data:
                return None
            
            # Parse required fields
            def safe_float(k: str, default=None):
                v = data.get(k)
                if v is None:
                    return default
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return default
            
            proposal = {
                "harvest_action": data.get("harvest_action"),
                "kill_score": safe_float("kill_score"),
                "k_regime_flip": safe_float("k_regime_flip", 0.0),
                "k_sigma_spike": safe_float("k_sigma_spike", 0.0),
                "k_ts_drop": safe_float("k_ts_drop", 0.0),
                "k_age_penalty": safe_float("k_age_penalty", 0.0),
                "new_sl_proposed": safe_float("new_sl_proposed"),
                "R_net": safe_float("R_net"),
                "last_update_epoch": safe_float("last_update_epoch"),
                "computed_at_utc": data.get("computed_at_utc", ""),
                "reason_codes": data.get("reason_codes", "").split(","),
            }
            
            # Validate required fields
            if proposal["harvest_action"] is None:
                logger.warning(f"{symbol}: Missing harvest_action")
                return None
            if proposal["kill_score"] is None:
                logger.warning(f"{symbol}: Missing kill_score")
                return None
            
            return proposal
            
        except Exception as e:
            logger.error(f"{symbol}: Error reading harvest proposal: {e}")
            return None
    
    def create_plan_id(self, symbol: str, proposal: Dict[str, Any]) -> str:
        """Create stable fingerprint for plan idempotency"""
        # Hash includes symbol + action + kill_score + sl_proposed + computed_at
        # This ensures same proposal generates same plan_id
        fingerprint = f"{symbol}:{proposal['harvest_action']}:{proposal['kill_score']:.6f}:{proposal.get('new_sl_proposed', 'none')}:{proposal['computed_at_utc']}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    def check_idempotency(self, plan_id: str) -> bool:
        """Check if plan already executed (returns True if duplicate)"""
        key = f"quantum:apply:dedupe:{plan_id}"
        result = self.redis.setnx(key, int(time.time()))
        if result == 1:
            # New plan, set TTL
            self.redis.expire(key, self.dedupe_ttl)
            return False
        else:
            # Duplicate
            if PROMETHEUS_AVAILABLE:
                self.metric_dedupe_hits.inc()
            return True
    
    def create_apply_plan(self, symbol: str, proposal: Dict[str, Any]) -> ApplyPlan:
        """Create apply plan from harvest proposal"""
        plan_id = self.create_plan_id(symbol, proposal)
        
        # Extract k_components
        k_components = {
            "regime_flip": proposal["k_regime_flip"],
            "sigma_spike": proposal["k_sigma_spike"],
            "ts_drop": proposal["k_ts_drop"],
            "age_penalty": proposal["k_age_penalty"],
        }
        
        # Default decision
        decision = Decision.EXECUTE
        reason_codes = []
        steps = []
        
        # Safety gate 1: Kill switch
        if self.kill_switch:
            decision = Decision.BLOCKED
            reason_codes.append("kill_switch_active")
            logger.warning(f"{symbol}: Blocked by kill switch")
        
        # Safety gate 2: Allowlist
        elif symbol not in self.allowlist:
            decision = Decision.SKIP
            reason_codes.append("not_in_allowlist")
            logger.debug(f"{symbol}: Not in allowlist")
        
        # Safety gate 3: Kill score thresholds
        elif proposal["kill_score"] >= self.k_block_critical:
            decision = Decision.BLOCKED
            reason_codes.append("kill_score_critical")
            logger.warning(f"{symbol}: Kill score {proposal['kill_score']:.3f} >= {self.k_block_critical} (critical)")
        
        elif proposal["kill_score"] >= self.k_block_warning:
            # Allow only CLOSE/tighten, block risk increase
            action = proposal["harvest_action"]
            if action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50"]:
                # Closing actions OK
                decision = Decision.EXECUTE
                reason_codes.append("kill_score_warning_close_ok")
                logger.info(f"{symbol}: Kill score {proposal['kill_score']:.3f} >= {self.k_block_warning} but action {action} is close (OK)")
            else:
                decision = Decision.BLOCKED
                reason_codes.append("kill_score_warning_risk_increase")
                logger.warning(f"{symbol}: Kill score {proposal['kill_score']:.3f} >= {self.k_block_warning}, blocking non-close action {action}")
        
        # Safety gate 4: Idempotency
        if decision == Decision.EXECUTE:
            if self.check_idempotency(plan_id):
                decision = Decision.SKIP
                reason_codes.append("duplicate_plan")
                logger.info(f"{symbol}: Plan {plan_id} already executed (duplicate)")
        
        # Build execution steps
        if decision == Decision.EXECUTE:
            action = proposal["harvest_action"]
            
            if action == "FULL_CLOSE_PROPOSED":
                steps.append({
                    "step": "CLOSE_FULL",
                    "type": "market_reduce_only",
                    "side": "close",  # determined by position side
                    "pct": 100.0
                })
            
            elif action == "PARTIAL_75":
                steps.append({
                    "step": "CLOSE_PARTIAL_75",
                    "type": "market_reduce_only",
                    "side": "close",
                    "pct": 75.0
                })
            
            elif action == "PARTIAL_50":
                steps.append({
                    "step": "CLOSE_PARTIAL_50",
                    "type": "market_reduce_only",
                    "side": "close",
                    "pct": 50.0
                })
            
            elif action == "UPDATE_SL":
                if proposal["new_sl_proposed"]:
                    steps.append({
                        "step": "UPDATE_SL",
                        "type": "stop_loss_order",
                        "price": proposal["new_sl_proposed"]
                    })
                else:
                    decision = Decision.ERROR
                    reason_codes.append("missing_new_sl")
            
            elif action == "HOLD":
                decision = Decision.SKIP
                reason_codes.append("action_hold")
            
            else:
                decision = Decision.ERROR
                reason_codes.append(f"unknown_action_{action}")
        
        return ApplyPlan(
            plan_id=plan_id,
            symbol=symbol,
            action=proposal["harvest_action"],
            kill_score=proposal["kill_score"],
            k_components=k_components,
            new_sl_proposed=proposal.get("new_sl_proposed"),
            R_net=proposal["R_net"],
            last_update_epoch=int(proposal["last_update_epoch"]),
            computed_at_utc=proposal["computed_at_utc"],
            decision=decision.value,
            reason_codes=reason_codes,
            steps=steps,
            close_qty=0.0,  # P3.2: Will be determined at execution
            price=None,  # P3.2: Will be fetched at execution
            timestamp=int(time.time())
        )
    
    def publish_plan(self, plan: ApplyPlan):
        """Publish apply plan to Redis stream"""
        try:
            stream_key = "quantum:stream:apply.plan"
            fields = {
                "plan_id": plan.plan_id,
                "symbol": plan.symbol,
                "action": plan.action,
                "kill_score": str(plan.kill_score),
                "k_regime_flip": str(plan.k_components["regime_flip"]),
                "k_sigma_spike": str(plan.k_components["sigma_spike"]),
                "k_ts_drop": str(plan.k_components["ts_drop"]),
                "k_age_penalty": str(plan.k_components["age_penalty"]),
                "new_sl_proposed": str(plan.new_sl_proposed) if plan.new_sl_proposed else "",
                "R_net": str(plan.R_net),
                "decision": plan.decision,
                "reason_codes": ",".join(plan.reason_codes),
                "steps": json.dumps(plan.steps),
                "close_qty": str(plan.close_qty),
                "price": str(plan.price) if plan.price else "",
                "timestamp": str(plan.timestamp)
            }
            self.redis.xadd(stream_key, fields, maxlen=10000)
            
            # Metrics
            if PROMETHEUS_AVAILABLE:
                self.metric_plan_total.labels(symbol=plan.symbol, decision=plan.decision).inc()
            
            logger.info(f"{plan.symbol}: Plan {plan.plan_id} published (decision={plan.decision}, steps={len(plan.steps)})")
            
        except Exception as e:
            logger.error(f"Error publishing plan: {e}")
    
    def execute_plan(self, plan: ApplyPlan) -> ApplyResult:
        """Execute apply plan (mode-dependent)"""
        if self.mode == ApplyMode.DRY_RUN:
            return self.execute_dry_run(plan)
        elif self.mode == ApplyMode.TESTNET:
            return self.execute_testnet(plan)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def execute_dry_run(self, plan: ApplyPlan) -> ApplyResult:
        """Dry run execution - no actual orders"""
        
        # CHECK GOVERNOR PERMIT (P3.2) - LOG ONLY in dry_run
        permit_key = f"quantum:permit:{plan.plan_id}"
        permit_data = self.redis.get(permit_key)
        
        if not permit_data:
            logger.info(f"{plan.symbol}: [DRY_RUN] No Governor permit (would block in testnet)")
        else:
            try:
                permit = json.loads(permit_data)
                if permit.get('granted'):
                    logger.info(f"{plan.symbol}: [DRY_RUN] Governor permit granted ✓")
                else:
                    logger.info(f"{plan.symbol}: [DRY_RUN] Governor permit denied (would block)")
            except json.JSONDecodeError:
                logger.warning(f"{plan.symbol}: [DRY_RUN] Invalid permit format")
        
        steps_results = []
        
        for step in plan.steps:
            steps_results.append({
                "step": step["step"],
                "status": "would_execute",
                "details": f"DRY_RUN: {step}"
            })
        
        return ApplyResult(
            plan_id=plan.plan_id,
            symbol=plan.symbol,
            decision=plan.decision,
            executed=False,
            would_execute=True,
            steps_results=steps_results,
            error=None,
            timestamp=int(time.time())
        )
    
    def execute_testnet(self, plan: ApplyPlan) -> ApplyResult:
        """Testnet execution - REAL orders to Binance Futures testnet"""
        # Check Binance credentials
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        
        if not api_key or not api_secret:
            logger.error(f"{plan.symbol}: Missing Binance testnet credentials")
            return ApplyResult(
                plan_id=plan.plan_id,
                symbol=plan.symbol,
                decision="ERROR",
                executed=False,
                would_execute=False,
                steps_results=[],
                error="missing_binance_credentials",
                timestamp=int(time.time())
            )
        
        steps_results = []
        
        try:
            # Initialize Binance client
            client = BinanceTestnetClient(api_key, api_secret)
            
            # Test connectivity
            if not client.ping():
                raise Exception("Binance testnet ping failed")
            
            logger.info(f"{plan.symbol}: Binance testnet connected")
            
            # Get current position
            position = client.get_position(plan.symbol)
            if not position or abs(position['positionAmt']) < 0.001:
                logger.warning(f"{plan.symbol}: No position found, skipping execution")
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="SKIP",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "POSITION_CHECK", "status": "no_position", "details": "No position to close"}],
                    error="no_position",
                    timestamp=int(time.time())
                )
            
            pos_amt = position['positionAmt']
            pos_side = position['side']
            logger.info(f"{plan.symbol}: Current position: {pos_amt} ({pos_side})")
            
            # WAIT FOR BOTH PERMITS (Governor P3.2 + P3.3 Position State Brain)
            # Max wait: 1200ms for event-driven P3.3 to issue permit after plan publication
            permit_key = f"quantum:permit:{plan.plan_id}"  # Governor (P3.2)
            p33_permit_key = f"quantum:permit:p33:{plan.plan_id}"  # P3.3 Position State Brain
            
            permits_ready = False
            permit_wait_start = time.time()
            max_wait_sec = 1.2  # 1200ms controlled wait window
            
            # Permit availability check loop
            for attempt in range(12):  # 12 x 100ms = 1200ms max
                try:
                    gov_exists = self.redis.exists(permit_key)
                    p33_exists = self.redis.exists(p33_permit_key)
                    
                    if gov_exists and p33_exists:
                        permits_ready = True
                        wait_time_ms = int((time.time() - permit_wait_start) * 1000)
                        logger.info(f"{plan.symbol}: Both permits ready after {wait_time_ms}ms (Governor + P3.3)")
                        break
                    
                    if attempt == 11:  # Last attempt
                        wait_time_ms = int((time.time() - permit_wait_start) * 1000)
                        missing = []
                        if not gov_exists:
                            missing.append("Governor")
                        if not p33_exists:
                            missing.append("P3.3")
                        logger.warning(f"{plan.symbol}: Permit timeout after {wait_time_ms}ms (missing: {', '.join(missing)})")
                    else:
                        time.sleep(0.1)  # 100ms between checks
                        
                except Exception as e:
                    logger.error(f"{plan.symbol}: Redis error during permit check: {e}")
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision="BLOCKED",
                        executed=False,
                        would_execute=False,
                        steps_results=[{"step": "PERMIT_CHECK", "status": "redis_error", "details": f"Redis error: {e}"}],
                        error="permit_check_redis_error",
                        timestamp=int(time.time())
                    )
            
            # BLOCK if permits not ready
            if not permits_ready:
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "PERMIT_CHECK", "status": "timeout", "details": "Permits not available within 1200ms window"}],
                    error="permit_timeout",
                    timestamp=int(time.time())
                )
            
            # ATOMICALLY CONSUME P3.3 PERMIT (read + delete)
            try:
                p33_permit_data = self.redis.get(p33_permit_key)
                if not p33_permit_data:
                    logger.warning(f"{plan.symbol}: P3.3 permit vanished during consumption")
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision="BLOCKED",
                        executed=False,
                        would_execute=False,
                        steps_results=[{"step": "P33_CONSUME", "status": "vanished", "details": "P3.3 permit disappeared"}],
                        error="missing_or_denied_p33_permit",
                        timestamp=int(time.time())
                    )
            except Exception as e:
                logger.error(f"{plan.symbol}: Redis error reading P3.3 permit: {e}")
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "P33_CONSUME", "status": "redis_error", "details": f"Redis error: {e}"}],
                    error="missing_or_denied_p33_permit",
                    timestamp=int(time.time())
                )
            
            try:
                p33_permit = json.loads(p33_permit_data)
                
                if not p33_permit.get('allow'):
                    reason = p33_permit.get('reason', 'unknown')
                    logger.warning(f"{plan.symbol}: P3.3 permit denied (reason={reason})")
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision="BLOCKED",
                        executed=False,
                        would_execute=False,
                        steps_results=[{"step": "P33_CHECK", "status": "denied", "details": f"P3.3 denied: {reason}"}],
                        error="missing_or_denied_p33_permit",
                        timestamp=int(time.time())
                    )
                
                # Extract safe_close_qty from P3.3 permit
                safe_close_qty = float(p33_permit.get('safe_close_qty', 0))
                exchange_amt = float(p33_permit.get('exchange_position_amt', 0))
                
                if safe_close_qty <= 0:
                    logger.warning(f"{plan.symbol}: P3.3 safe_close_qty is zero or negative")
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision="BLOCKED",
                        executed=False,
                        would_execute=False,
                        steps_results=[{"step": "P33_CHECK", "status": "invalid_qty", "details": f"safe_close_qty={safe_close_qty}"}],
                        error="missing_or_denied_p33_permit",
                        timestamp=int(time.time())
                    )
                
                # DELETE P3.3 permit (single-use)
                self.redis.delete(p33_permit_key)
                logger.info(f"{plan.symbol}: P3.3 permit consumed ✓ (safe_qty={safe_close_qty:.4f}, exchange_amt={exchange_amt:.4f})")
                
            except json.JSONDecodeError as e:
                logger.error(f"{plan.symbol}: Invalid P3.3 permit format: {e}")
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "P33_CHECK", "status": "invalid_format", "details": f"Invalid P3.3 permit: {e}"}],
                    error="missing_or_denied_p33_permit",
                    timestamp=int(time.time())
                )
            
            # Execute each step
            for step in plan.steps:
                logger.info(f"{plan.symbol}: Executing step {step['step']} (TESTNET)")
                
                try:
                    if step['step'] in ['CLOSE_FULL', 'CLOSE_PARTIAL_75', 'CLOSE_PARTIAL_50']:
                        # USE SAFE_CLOSE_QTY FROM P3.3 PERMIT (already clamped and rounded)
                        close_qty = safe_close_qty
                        
                        logger.info(f"{plan.symbol}: Using P3.3 safe_close_qty={close_qty:.4f} (exchange_amt={exchange_amt:.4f})")
                        
                        # Determine order side (opposite of position)
                        order_side = 'SELL' if pos_amt > 0 else 'BUY'
                        
                        logger.info(f"{plan.symbol}: Placing {order_side} order for {close_qty} (reduceOnly)")
                        
                        # Place market order with reduceOnly
                        order_result = client.place_market_order(
                            symbol=plan.symbol,
                            side=order_side,
                            quantity=close_qty,
                            reduce_only=True
                        )
                        
                        steps_results.append({
                            "step": step["step"],
                            "status": "success",
                            "details": f"Order {order_result['orderId']}: {order_side} {order_result['executedQty']} @ MARKET (reduceOnly)",
                            "order_id": str(order_result['orderId']),
                            "side": order_side,
                            "quantity": order_result['quantity'],
                            "executed_qty": order_result['executedQty'],
                            "reduce_only": True
                        })
                        
                        logger.info(f"{plan.symbol}: Order {order_result['orderId']} executed successfully")
                        
                    elif step['step'] == 'UPDATE_SL':
                        # Stop loss modification not implemented yet
                        steps_results.append({
                            "step": step["step"],
                            "status": "not_implemented",
                            "details": "Stop loss modification not yet supported"
                        })
                        logger.warning(f"{plan.symbol}: UPDATE_SL not implemented")
                    
                    else:
                        steps_results.append({
                            "step": step["step"],
                            "status": "unknown_step",
                            "details": f"Unknown step type: {step['step']}"
                        })
                    
                    # Metrics
                    if PROMETHEUS_AVAILABLE:
                        self.metric_execute_total.labels(
                            symbol=plan.symbol,
                            step=step["step"],
                            status="success"
                        ).inc()
                    
                except Exception as step_error:
                    logger.error(f"{plan.symbol}: Step {step['step']} failed: {step_error}")
                    steps_results.append({
                        "step": step["step"],
                        "status": "error",
                        "details": str(step_error)
                    })
                    
                    # Metrics
                    if PROMETHEUS_AVAILABLE:
                        self.metric_execute_total.labels(
                            symbol=plan.symbol,
                            step=step["step"],
                            status="error"
                        ).inc()
            
            # Update last success metric if any steps succeeded
            if any(s['status'] == 'success' for s in steps_results):
                if PROMETHEUS_AVAILABLE:
                    self.metric_last_success.labels(symbol=plan.symbol).set(time.time())
            
            return ApplyResult(
                plan_id=plan.plan_id,
                symbol=plan.symbol,
                decision=plan.decision,
                executed=True,
                would_execute=False,
                steps_results=steps_results,
                error=None,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            logger.error(f"{plan.symbol}: Execution error: {e}", exc_info=True)
            return ApplyResult(
                plan_id=plan.plan_id,
                symbol=plan.symbol,
                decision="ERROR",
                executed=False,
                would_execute=False,
                steps_results=steps_results,
                error=str(e),
                timestamp=int(time.time())
            )
    
    def publish_result(self, result: ApplyResult):
        """Publish apply result to Redis stream"""
        try:
            stream_key = "quantum:stream:apply.result"
            fields = {
                "plan_id": result.plan_id,
                "symbol": result.symbol,
                "decision": result.decision,
                "executed": str(result.executed),
                "would_execute": str(result.would_execute),
                "steps_results": json.dumps(result.steps_results),
                "error": result.error or "",
                "timestamp": str(result.timestamp)
            }
            self.redis.xadd(stream_key, fields, maxlen=10000)
            
            logger.info(f"{result.symbol}: Result published (executed={result.executed}, error={result.error})")
            
        except Exception as e:
            logger.error(f"Error publishing result: {e}")
    
    def run_cycle(self):
        """Single apply cycle"""
        logger.debug("=== Apply Layer Cycle ===")
        
        for symbol in self.symbols:
            try:
                # Read harvest proposal
                proposal = self.get_harvest_proposal(symbol)
                if not proposal:
                    logger.debug(f"{symbol}: No proposal available")
                    continue
                
                # Create apply plan
                plan = self.create_apply_plan(symbol, proposal)
                
                # Publish plan
                self.publish_plan(plan)
                
                # Execute if decision is EXECUTE
                if plan.decision == Decision.EXECUTE.value:
                    result = self.execute_plan(plan)
                    self.publish_result(result)
                else:
                    # Publish skip/blocked result
                    result = ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision=plan.decision,
                        executed=False,
                        would_execute=False,
                        steps_results=[],
                        error=None,
                        timestamp=int(time.time())
                    )
                    self.publish_result(result)
                
            except Exception as e:
                logger.error(f"{symbol}: Error in cycle: {e}", exc_info=True)
    
    def run_loop(self):
        """Main apply loop"""
        logger.info("Starting apply loop")
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down (KeyboardInterrupt)")
                break
            except Exception as e:
                logger.error(f"Error in apply loop: {e}", exc_info=True)
            
            time.sleep(self.poll_interval)


def main():
    logger.info("=== P3 Apply Layer ===")
    
    apply_layer = ApplyLayer()
    apply_layer.run_loop()


if __name__ == "__main__":
    main()
