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


# ---- PRODUCTION HYGIENE: Hard Mode Switch ----
# TESTNET=true: Governor bypass (for development/testing)
# TESTNET=false: Require BOTH permits (production safety)
TESTNET_MODE = os.getenv("TESTNET", "false").lower() in ("true", "1", "yes")
if TESTNET_MODE:
    logger.warning("⚠️  TESTNET MODE ENABLED - Governor bypass active (NO PRODUCTION USAGE)")
else:
    logger.info("✅ PRODUCTION MODE - Both permits required (Governor + P3.3)")

# ---- PRODUCTION HYGIENE: Safety Kill Switch ----
# Set quantum:global:kill_switch = true to halt all execution
SAFETY_KILL_KEY = "quantum:global:kill_switch"

# ---- PRODUCTION HYGIENE: Prometheus Metrics ----
if PROMETHEUS_AVAILABLE:
    # Permit metrics
    p33_permit_deny = Counter('p33_permit_deny_total', 'Total P3.3 denies', ['reason'])
    p33_permit_allow = Counter('p33_permit_allow_total', 'Total P3.3 allows')
    governor_block = Counter('governor_block_total', 'Total Governor blocks', ['reason'])
    
    # Execution metrics
    apply_executed = Counter('apply_executed_total', 'Total executed', ['status'])
    plan_processed = Counter('apply_plan_processed_total', 'Total plans processed', ['decision'])
    
    # Position metrics
    position_mismatch = Gauge('position_mismatch_seconds', 'Seconds since last position match')
    permit_wait_time = Gauge('permit_wait_ms', 'Last permit wait time (ms)')

# ---- Permit wait-loop config (fail-closed) ----
PERMIT_WAIT_MS = int(os.getenv("APPLY_PERMIT_WAIT_MS", "1200"))
PERMIT_STEP_MS = int(os.getenv("APPLY_PERMIT_STEP_MS", "100"))

# Atomic Lua: require BOTH permits then consume both (DEL)
_LUA_CONSUME_BOTH_PERMITS = r"""
-- Atomic: require BOTH permits, then consume (DEL) both.
-- Returns:
--  {1, gov_json, p33_json} on success
--  {0, reason, gov_ttl, p33_ttl} on failure

local gov_key = KEYS[1]
local p33_key = KEYS[2]

local gov = redis.call("GET", gov_key)
local p33 = redis.call("GET", p33_key)

if (not gov) and (not p33) then
  return {0, "missing_both", redis.call("TTL", gov_key), redis.call("TTL", p33_key)}
end
if not gov then
  return {0, "missing_governor", redis.call("TTL", gov_key), redis.call("TTL", p33_key)}
end
if not p33 then
  return {0, "missing_p33", redis.call("TTL", gov_key), redis.call("TTL", p33_key)}
end

redis.call("DEL", gov_key)
redis.call("DEL", p33_key)

return {1, gov, p33}
"""

def _register_consume_script(r):
    """Register Lua script for atomic permit consumption"""
    return r.register_script(_LUA_CONSUME_BOTH_PERMITS)

def wait_and_consume_permits(
    r,
    plan_id: str,
    max_wait_ms: int = PERMIT_WAIT_MS,
    step_ms: int = PERMIT_STEP_MS,
    consume_script=None,
):
    """
    Wait up to max_wait_ms for BOTH permits:
      - quantum:permit:{plan_id}        (Governor P3.2)
      - quantum:permit:p33:{plan_id}    (Position State Brain P3.3)
    Atomically consumes both on success.
    Returns:
      (True, gov_dict, p33_dict) on success
      (False, info_dict, None)  on failure (fail-closed)
    """
    gov_key = f"quantum:permit:{plan_id}"
    p33_key = f"quantum:permit:p33:{plan_id}"

    if consume_script is None:
        consume_script = _register_consume_script(r)

    deadline = time.time() + (max_wait_ms / 1000.0)
    last = {"reason": "init", "gov_ttl": -2, "p33_ttl": -2}

    while time.time() < deadline:
        res = consume_script(keys=[gov_key, p33_key], args=[])
        if int(res[0]) == 1:
            try:
                gov = json.loads(res[1])
            except Exception:
                gov = {"raw": res[1]}
            try:
                p33 = json.loads(res[2])
            except Exception:
                p33 = {"raw": res[2]}
            return True, gov, p33

        # failure details from lua
        last = {
            "reason": str(res[1]),
            "gov_ttl": int(res[2]),
            "p33_ttl": int(res[3]),
        }
        time.sleep(step_ms / 1000.0)

    return False, last, None


class ApplyMode(Enum):
    DRY_RUN = "dry_run"
    TESTNET = "testnet"


class Decision(Enum):
    EXECUTE = "EXECUTE"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    RECONCILE_CLOSE = "RECONCILE_CLOSE"


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
            # Check if plan already published to stream (dedupe)
            stream_published_key = f"quantum:apply:stream_published:{plan.plan_id}"
            if self.redis.exists(stream_published_key):
                logger.debug(f"{plan.symbol}: Plan {plan.plan_id} already published to stream (skipping republish)")
                return
            
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
            
            # Mark as published to stream (5 min TTL - same as apply cycle)
            self.redis.setex(stream_published_key, 300, "1")
            
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
        # ---- PRODUCTION HYGIENE: Safety Kill Switch Check ----
        try:
            kill_switch = self.redis.get(SAFETY_KILL_KEY)
            if kill_switch and kill_switch.lower() in (b"true", b"1", b"yes"):
                logger.critical(f"[KILL_SWITCH] Execution halted - kill switch is ACTIVE")
                if PROMETHEUS_AVAILABLE:
                    apply_executed.labels(status='kill_switch').inc()
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision=plan.decision,
                    executed=False,
                    would_execute=True,
                    steps_results=[],
                    error="kill_switch_active",
                    timestamp=int(time.time())
                )
        except Exception as e:
            logger.warning(f"[KILL_SWITCH] Error checking kill switch: {e}")
        
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
            
            # ---- HARD MODE SWITCH: TESTNET vs PRODUCTION ----
            if TESTNET_MODE:
                # TESTNET: Skip all permits, go straight to execution
                logger.info(f"[TESTNET_BYPASS] Skipping permits for {plan.plan_id}")
                gov_permit = {"granted": True, "mode": "testnet_bypass"}
                p33_permit = {"allow": True, "safe_qty": plan.sell_qty, "mode": "testnet_bypass"}
                ok = True
                wait_ms = 0
                if PROMETHEUS_AVAILABLE:
                    apply_executed.labels(status='testnet_bypass').inc()
            else:
                # PRODUCTION: Require BOTH permits (Governor + P3.3)
                t0 = time.time()
                consume_script = _register_consume_script(self.redis)
                ok, gov_permit, p33_permit = wait_and_consume_permits(
                    self.redis, 
                    plan.plan_id,
                    max_wait_ms=PERMIT_WAIT_MS,
                    step_ms=PERMIT_STEP_MS,
                    consume_script=consume_script
                )
                wait_ms = int((time.time() - t0) * 1000)
                if PROMETHEUS_AVAILABLE:
                    permit_wait_time.set(wait_ms)
            
            if not ok:
                logger.warning(
                    f"[PERMIT_WAIT] BLOCK plan={plan.plan_id} symbol={plan.symbol} "
                    f"wait_ms={wait_ms} info={gov_permit}"
                )
                if PROMETHEUS_AVAILABLE:
                    reason = gov_permit.get('reason', 'unknown')
                    governor_block.labels(reason=reason).inc()
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "PERMIT_WAIT", "status": gov_permit.get('reason', 'unknown'), "details": f"Permits not available: {gov_permit}"}],
                    error=f"permit_timeout_or_missing:{gov_permit.get('reason','unknown')}",
                    timestamp=int(time.time())
                )
            
            # Validate P3.3 permit
            if not isinstance(p33_permit, dict):
                logger.warning(f"[PERMIT_WAIT] BLOCK invalid_p33_format plan={plan.plan_id} symbol={plan.symbol}")
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "P33_PERMIT", "status": "invalid_format", "details": f"Invalid P3.3 permit format"}],
                    error="invalid_p33_permit_format",
                    timestamp=int(time.time())
                )
            
            # Check if P3.3 allow flag is set
            if not p33_permit.get('allow'):
                reason = p33_permit.get('reason', 'unknown')
                logger.warning(
                    f"[PERMIT_WAIT] BLOCK p33_denied plan={plan.plan_id} symbol={plan.symbol} "
                    f"reason={reason}"
                )
                if PROMETHEUS_AVAILABLE:
                    p33_permit_deny.labels(reason=reason).inc()
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "P33_CHECK", "status": "denied", "details": f"P3.3 denied: {reason}"}],
                    error="p33_permit_denied",
                    timestamp=int(time.time())
                )
            
            # Extract safe_close_qty from P3.3 permit
            safe_close_qty = float(p33_permit.get('safe_close_qty', 0))
            exchange_amt = float(p33_permit.get('exchange_position_amt', 0))
            
            if safe_close_qty <= 0:
                logger.warning(
                    f"[PERMIT_WAIT] BLOCK invalid_safe_qty plan={plan.plan_id} symbol={plan.symbol} "
                    f"safe_qty={safe_close_qty}"
                )
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision="BLOCKED",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "P33_VALIDATE", "status": "invalid_qty", "details": f"safe_close_qty={safe_close_qty}"}],
                    error="invalid_safe_close_qty",
                    timestamp=int(time.time())
                )
            
            logger.info(
                f"[PERMIT_WAIT] OK plan={plan.plan_id} symbol={plan.symbol} "
                f"wait_ms={wait_ms} safe_qty={safe_close_qty:.4f}"
            )
            if PROMETHEUS_AVAILABLE:
                p33_permit_allow.inc()
            
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
                    apply_executed.labels(status='success').inc()
            
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
    
    def validate_reconcile_close_guardrails(self, plan_data: Dict) -> tuple[bool, str]:
        """
        Validate guardrails for RECONCILE_CLOSE.
        
        Returns: (is_safe, error_message)
        """
        symbol = plan_data.get('symbol')
        
        # 1. Check HOLD key exists
        hold_key = f"quantum:reconcile:hold:{symbol}"
        if not self.redis.exists(hold_key):
            return False, "no_hold_key_active"
        
        # 2. Check reduceOnly flag
        if not plan_data.get('reduceOnly') or plan_data.get('reduceOnly') not in ('true', 'True', True):
            return False, "reduceOnly_not_set"
        
        # 3. Check type is MARKET
        order_type = plan_data.get('type', 'MARKET')
        if order_type not in ('MARKET', ''):
            return False, "invalid_order_type"
        
        # 4. Check qty is safe (not bigger than exchange position)
        try:
            exchange_amt = float(plan_data.get('exchange_amt', 0))
            qty = float(plan_data.get('qty', 0))
            if qty > abs(exchange_amt):
                return False, "qty_exceeds_exchange_position"
        except (ValueError, TypeError):
            return False, "invalid_qty_or_exchange_amt"
        
        # 5. Check reason is reconcile_drift
        if plan_data.get('reason') != 'reconcile_drift':
            return False, "invalid_reason"
        
        return True, ""
    
    def execute_reconcile_close(self, plan_data: Dict) -> ApplyResult:
        """
        Execute RECONCILE_CLOSE plan with strict guardrails.
        
        Only allowed when:
        - HOLD key is active for symbol
        - reduceOnly=true (mandatory)
        - Order type is MARKET
        - qty <= abs(exchange_amt)
        - reason='reconcile_drift'
        """
        plan_id = plan_data.get('plan_id')
        symbol = plan_data.get('symbol')
        side = plan_data.get('side')
        qty = float(plan_data.get('qty', 0))
        
        logger.warning(f"[RECONCILE_CLOSE] {symbol}: Execution started - plan_id={plan_id[:16]}, side={side}, qty={qty}")
        
        # Validate guardrails
        is_safe, error = self.validate_reconcile_close_guardrails(plan_data)
        if not is_safe:
            logger.error(f"[RECONCILE_CLOSE] {symbol}: Guardrails failed: {error}")
            return ApplyResult(
                plan_id=plan_id,
                symbol=symbol,
                decision="RECONCILE_CLOSE",
                executed=False,
                would_execute=False,
                steps_results=[{"step": "GUARDRAILS", "status": "failed", "reason": error}],
                error=error,
                timestamp=int(time.time())
            )
        
        try:
            # Get API credentials
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
            
            if not api_key or not api_secret:
                raise Exception("Missing Binance credentials")
            
            # Create Binance client
            client = BinanceTestnetClient(api_key, api_secret)
            
            # Place market order with reduceOnly
            logger.warning(f"[RECONCILE_CLOSE] {symbol}: Placing order - side={side}, qty={qty}, reduceOnly=true")
            result = client.place_market_order(symbol, side, qty, reduce_only=True)
            
            if result.get('orderId'):
                order_id = result['orderId']
                logger.warning(f"[RECONCILE_CLOSE_EXECUTED] {symbol}: plan_id={plan_id[:16]}, order_id={order_id}, qty={qty}")
                
                if PROMETHEUS_AVAILABLE:
                    apply_executed.labels(status='reconcile_close_success').inc()
                
                return ApplyResult(
                    plan_id=plan_id,
                    symbol=symbol,
                    decision="RECONCILE_CLOSE",
                    executed=True,
                    would_execute=False,
                    steps_results=[{"step": "PLACE_ORDER", "status": "success", "order_id": order_id, "qty": qty}],
                    error=None,
                    timestamp=int(time.time())
                )
            else:
                error_msg = result.get('msg', 'Unknown error')
                logger.error(f"[RECONCILE_CLOSE] {symbol}: Order placement failed: {error_msg}")
                return ApplyResult(
                    plan_id=plan_id,
                    symbol=symbol,
                    decision="RECONCILE_CLOSE",
                    executed=False,
                    would_execute=False,
                    steps_results=[{"step": "PLACE_ORDER", "status": "failed", "error": error_msg}],
                    error=error_msg,
                    timestamp=int(time.time())
                )
        
        except Exception as e:
            logger.error(f"[RECONCILE_CLOSE] {symbol}: Execution error: {e}", exc_info=True)
            return ApplyResult(
                plan_id=plan_id,
                symbol=symbol,
                decision="RECONCILE_CLOSE",
                executed=False,
                would_execute=False,
                steps_results=[{"step": "EXECUTION", "status": "error", "error": str(e)}],
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
    
    def process_reconcile_close_stream(self):
        """Process RECONCILE_CLOSE plans from dedicated reconcile.close stream"""
        try:
            stream_key = "quantum:stream:reconcile.close"
            consumer_group = "apply_recon"
            consumer_id = f"apply-{os.getpid()}"
            
            # Create consumer group (idempotent)
            try:
                self.redis.xgroup_create(stream_key, consumer_group, id='0-0', mkstream=True)
                logger.warning(f"[RECON] Created consumer group {consumer_group}")
            except:
                pass  # Group already exists
            
            # Read RECONCILE_CLOSE messages
            logger.warning(f"[RECON] Reading from {stream_key}...")
            messages = self.redis.xreadgroup(
                groupname=consumer_group,
                consumername=consumer_id,
                streams={stream_key: '>'},
                count=10,
                block=100  # Short block, don't starve harvest loop
            )
            
            logger.warning(f"[RECON] Got messages: {messages is not None and len(messages) > 0}")
            if not messages:
                return
            
            logger.warning(f"[RECON] Processing {len(messages)} stream(s)")
            for stream_name, stream_messages in messages:
                logger.warning(f"[RECON] Stream {stream_name}: {len(stream_messages)} messages")
                for msg_id, fields in stream_messages:
                    # Convert fields to plain dict
                    plan_data = {}
                    for k, v in fields.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        val = v.decode() if isinstance(v, bytes) else v
                        plan_data[key] = val
                    
                    symbol = plan_data.get('symbol')
                    plan_id = plan_data.get('plan_id', '')
                    
                    logger.warning(f"[RECONCILE_CLOSE] {symbol}: Plan {plan_id[:16]} - Starting execution")
                    
                    # Execute RECONCILE_CLOSE (with strict guardrails)
                    result = self.execute_reconcile_close(plan_data)
                    self.publish_result(result)
                    
                    # ACK the message
                    self.redis.xack(stream_key, consumer_group, msg_id)
                    logger.warning(f"[RECONCILE_CLOSE] {symbol}: Plan {plan_id[:16]} - ACK sent")
        
        except Exception as e:
            logger.error(f"Error processing reconcile.close stream: {e}", exc_info=True)
    
    def run_cycle(self):
        """Single apply cycle"""
        logger.debug("=== Apply Layer Cycle ===")
        
        # HIGHEST PRIORITY: Process RECONCILE_CLOSE plans (self-healing)
        logger.warning("[RECON] Starting process_reconcile_close_stream call...")
        try:
            self.process_reconcile_close_stream()
        except Exception as e:
            logger.error(f"Error processing RECONCILE_CLOSE stream: {e}", exc_info=True)
        
        # NORMAL: Process harvest proposals
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
