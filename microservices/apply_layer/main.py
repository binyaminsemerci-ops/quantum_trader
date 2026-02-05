#!/usr/bin/env python3
"""
P3 Apply Layer - Harvest Proposal Consumer

BUILD_TAG: apply-layer-entry-exit-sep-v1

Reads harvest proposals from Redis, creates apply plans, and optionally executes them.

Modes:
- dry_run (P3.0): Plans published, no execution
- testnet (P3.1): Plans executed against Binance testnet

Safety gates:
- Allowlist (default: BTCUSDT only)
- Entry/Exit separation (open_threshold=0.85, close_threshold=0.65, qty_scale)
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
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Exit ownership enforcement (single exit controller)
try:
    from lib.exit_ownership import EXIT_OWNER, validate_exit_ownership
    EXIT_OWNERSHIP_ENABLED = True
except ImportError:
    EXIT_OWNER = "exitbrain_v3_5"
    EXIT_OWNERSHIP_ENABLED = False
    print("WARN: exit_ownership module not found, exit gate disabled")

# PolicyStore (fail-closed allowlist enforcement)
try:
    from lib.policy_store import load_policy, PolicyData
    POLICY_ENABLED = True
except ImportError:
    POLICY_ENABLED = False
    print("WARN: policy_store module not found, allowlist gate disabled")

# P2.8A Apply Heat Observer
try:
    from microservices.apply_layer import heat_observer
except ImportError:
    heat_observer = None
    print("WARN: heat_observer module not found, P2.8A observability disabled")

# Risk Policy Enforcer (LAYER 0-2 gates)
try:
    from microservices.risk_policy_enforcer import (
        RiskPolicyEnforcer, 
        RiskLimits, 
        SystemState, 
        FailureType,
        log_risk_metrics
    )
    RISK_ENFORCER_AVAILABLE = True
except ImportError:
    RISK_ENFORCER_AVAILABLE = False
    print("WARN: risk_policy_enforcer not found, risk gates disabled")

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
# TESTNET=false: Require THREE permits (production safety)
TESTNET_MODE = os.getenv("TESTNET", "false").lower() in ("true", "1", "yes")

# ---- DEV MODE: Dedupe bypass for rapid testing ----
# APPLY_DEDUPE_BYPASS=true: 5s TTL (dev/QA only)
# APPLY_DEDUPE_BYPASS=false: 5min TTL (production default)
DEDUPE_BYPASS = os.getenv("APPLY_DEDUPE_BYPASS", "false").lower() in ("true", "1", "yes")
DEDUPE_TTL = 5 if DEDUPE_BYPASS else 300  # 5 sec vs 5 min

# ---- Position Guard: Epsilon for floating point comparison ----
POSITION_EPSILON = 1e-12  # abs(position_amt) > epsilon means "has position"

if TESTNET_MODE:
    logger.warning("ΓÜá∩╕Å  TESTNET MODE ENABLED - Governor bypass active (NO PRODUCTION USAGE)")
else:
    logger.info("Γ£à PRODUCTION MODE - Three permits required (Governor + P3.3 + P2.6)")

# ---- PRODUCTION HYGIENE: Safety Kill Switch ----
# Set quantum:global:kill_switch = true to halt all execution
SAFETY_KILL_KEY = "quantum:global:kill_switch"

# ---- LEARNING PLANE HEARTBEAT (fail-closed) ----
RL_FEEDBACK_HEARTBEAT_KEY = "quantum:svc:rl_feedback_v2:heartbeat"
RL_TRAINER_HEARTBEAT_KEY = "quantum:svc:rl_trainer:heartbeat"
LEARNING_PLANE_HEARTBEAT_TTL = int(os.getenv("LEARNING_PLANE_HEARTBEAT_TTL_SEC", "30"))

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

# Atomic Lua: require THREE permits then consume all (DEL)
_LUA_CONSUME_THREE_PERMITS = r"""
-- Atomic: require THREE permits (Governor + P3.3 + P2.6), then consume (DEL) all.
-- Returns:
--  {1, gov_json, p33_json, p26_json} on success
--  {0, reason, gov_ttl, p33_ttl, p26_ttl} on failure

local gov_key = KEYS[1]
local p33_key = KEYS[2]
local p26_key = KEYS[3]

local gov = redis.call("GET", gov_key)
local p33 = redis.call("GET", p33_key)
local p26 = redis.call("GET", p26_key)

if (not gov) and (not p33) and (not p26) then
  return {0, "missing_all", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end
if not gov then
  return {0, "missing_governor", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end
if not p33 then
  return {0, "missing_p33", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end
if not p26 then
  return {0, "missing_p26", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end

redis.call("DEL", gov_key)
redis.call("DEL", p33_key)
redis.call("DEL", p26_key)

return {1, gov, p33, p26}
"""

def _register_consume_script(r):
    """Register Lua script for atomic permit consumption (THREE permits)"""
    return r.register_script(_LUA_CONSUME_THREE_PERMITS)

def wait_and_consume_permits(
    r,
    plan_id: str,
    max_wait_ms: int = PERMIT_WAIT_MS,
    step_ms: int = PERMIT_STEP_MS,
    consume_script=None,
):
    """
    Wait up to max_wait_ms for THREE permits:
      - quantum:permit:{plan_id}        (Governor P3.2)
      - quantum:permit:p33:{plan_id}    (Position State Brain P3.3)
      - quantum:permit:p26:{plan_id}    (Portfolio Gate P2.6)
    Atomically consumes all three on success.
    Returns:
      (True, gov_dict, p33_dict, p26_dict) on success
      (False, info_dict, None, None)  on failure (fail-closed)
    """
    gov_key = f"quantum:permit:{plan_id}"
    p33_key = f"quantum:permit:p33:{plan_id}"
    p26_key = f"quantum:permit:p26:{plan_id}"

    if consume_script is None:
        consume_script = _register_consume_script(r)

    deadline = time.time() + (max_wait_ms / 1000.0)
    last = {"reason": "init", "gov_ttl": -2, "p33_ttl": -2, "p26_ttl": -2}

    while time.time() < deadline:
        res = consume_script(keys=[gov_key, p33_key, p26_key], args=[])
        if int(res[0]) == 1:
            try:
                gov = json.loads(res[1])
            except Exception:
                gov = {"raw": res[1]}
            try:
                p33 = json.loads(res[2])
            except Exception:
                p33 = {"raw": res[2]}
            try:
                p26 = json.loads(res[3]) if len(res) > 3 else {"granted": True}
            except Exception:
                p26 = {"raw": res[3] if len(res) > 3 else "1"}
            return True, gov, p33, p26

        # failure details from lua
        last = {
            "reason": str(res[1]),
            "gov_ttl": int(res[2]),
            "p33_ttl": int(res[3]),
            "p26_ttl": int(res[4]) if len(res) > 4 else -2,
        }
        time.sleep(step_ms / 1000.0)

    return False, last, None, None


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
    source: str  # Source service (exitbrain_v3_5, manual_close, etc.)
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


class ApplyMode(Enum):
    DRY_RUN = "dry_run"
    TESTNET = "testnet"


class Decision(Enum):
    EXECUTE = "EXECUTE"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    RECONCILE_CLOSE = "RECONCILE_CLOSE"


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
# ---- Permit wait-loop config (fail-closed) ----
PERMIT_WAIT_MS = int(os.getenv("APPLY_PERMIT_WAIT_MS", "1200"))
PERMIT_STEP_MS = int(os.getenv("APPLY_PERMIT_STEP_MS", "100"))

# Atomic Lua: require THREE permits then consume all (DEL)
_LUA_CONSUME_THREE_PERMITS = r"""
-- Atomic: require THREE permits (Governor + P3.3 + P2.6), then consume (DEL) all.
-- Returns:
--  {1, gov_json, p33_json, p26_json} on success
--  {0, reason, gov_ttl, p33_ttl, p26_ttl} on failure

local gov_key = KEYS[1]
local p33_key = KEYS[2]
local p26_key = KEYS[3]

local gov = redis.call("GET", gov_key)
local p33 = redis.call("GET", p33_key)
local p26 = redis.call("GET", p26_key)

if (not gov) and (not p33) and (not p26) then
  return {0, "missing_all", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end
if not gov then
  return {0, "missing_governor", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end
if not p33 then
  return {0, "missing_p33", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end
if not p26 then
  return {0, "missing_p26", redis.call("TTL", gov_key), redis.call("TTL", p33_key), redis.call("TTL", p26_key)}
end

redis.call("DEL", gov_key)
redis.call("DEL", p33_key)
redis.call("DEL", p26_key)

return {1, gov, p33, p26}
"""

def _register_consume_script(r):
    """Register Lua script for atomic permit consumption (THREE permits)"""
    return r.register_script(_LUA_CONSUME_THREE_PERMITS)

def wait_and_consume_permits(
    r,
    plan_id: str,
    max_wait_ms: int = PERMIT_WAIT_MS,
    step_ms: int = PERMIT_STEP_MS,
    consume_script=None,
):
    """
    Wait up to max_wait_ms for THREE permits:
      - quantum:permit:{plan_id}        (Governor P3.2)
      - quantum:permit:p33:{plan_id}    (Position State Brain P3.3)
      - quantum:permit:p26:{plan_id}    (Portfolio Gate P2.6)
    Atomically consumes all three on success.
    Returns:
      (True, gov_dict, p33_dict, p26_dict) on success
      (False, info_dict, None, None)  on failure (fail-closed)
    """
    gov_key = f"quantum:permit:{plan_id}"
    p33_key = f"quantum:permit:p33:{plan_id}"
    p26_key = f"quantum:permit:p26:{plan_id}"

    if consume_script is None:
        consume_script = _register_consume_script(r)

    deadline = time.time() + (max_wait_ms / 1000.0)
    last = {"reason": "init", "gov_ttl": -2, "p33_ttl": -2, "p26_ttl": -2}

    while time.time() < deadline:
        res = consume_script(keys=[gov_key, p33_key, p26_key], args=[])
        if int(res[0]) == 1:
            try:
                gov = json.loads(res[1])
            except Exception:
                gov = {"raw": res[1]}
            try:
                p33 = json.loads(res[2])
            except Exception:
                p33 = {"raw": res[2]}
            try:
                p26 = json.loads(res[3]) if len(res) > 3 else {"granted": True}
            except Exception:
                p26 = {"raw": res[3] if len(res) > 3 else "1"}
            return True, gov, p33, p26

        # failure details from lua
        last = {
            "reason": str(res[1]),
            "gov_ttl": int(res[2]),
            "p33_ttl": int(res[3]),
            "p26_ttl": int(res[4]) if len(res) > 4 else -2,
        }
        time.sleep(step_ms / 1000.0)

    return False, last, None, None


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
    source: str  # Source service (exitbrain_v3_5, manual_close, etc.)
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

        # Risk Policy Enforcer (LAYER 0-2 gates)
        if RISK_ENFORCER_AVAILABLE:
            risk_limits = RiskLimits(
                heartbeat_max_age_sec=int(os.getenv("RISK_HEARTBEAT_TTL", "30")),
                max_leverage=float(os.getenv("RISK_MAX_LEVERAGE", "10.0")),
                daily_loss_limit=float(os.getenv("RISK_DAILY_LOSS_LIMIT", "-1000.0")),
                rolling_drawdown_max_pct=float(os.getenv("RISK_DRAWDOWN_MAX_PCT", "15.0")),
                rolling_drawdown_window_days=int(os.getenv("RISK_DRAWDOWN_WINDOW_DAYS", "30")),
                max_consecutive_losses=int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "5")),
                loss_streak_cooldown_minutes=int(os.getenv("RISK_LOSS_COOLDOWN_MIN", "60")),
                vol_min=float(os.getenv("RISK_VOL_MIN", "0.005")),
                vol_max=float(os.getenv("RISK_VOL_MAX", "0.10")),
                max_spread_bps=float(os.getenv("RISK_MAX_SPREAD_BPS", "10.0")),
                symbol_whitelist=os.getenv("RISK_SYMBOL_WHITELIST", "BTCUSDT,ETHUSDT").split(",")
            )
            self.risk_enforcer = RiskPolicyEnforcer(self.redis, risk_limits)
            logger.info("Risk Policy Enforcer initialized (LAYER 0-2 active)")
            try:
                self.redis.set("quantum:risk:boot_ts", str(self.risk_enforcer.boot_ts))
            except Exception:
                pass
        else:
            self.risk_enforcer = None
            logger.warning("Risk Policy Enforcer DISABLED - no risk gates active")

        # Learning plane heartbeat (legacy, now handled by risk enforcer)
        self.rl_feedback_heartbeat_key = RL_FEEDBACK_HEARTBEAT_KEY
        self.rl_trainer_heartbeat_key = RL_TRAINER_HEARTBEAT_KEY
        self.learning_plane_ttl = LEARNING_PLANE_HEARTBEAT_TTL
        
        self.mode = ApplyMode(os.getenv("APPLY_MODE", "dry_run"))
        self.symbols = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
        
        # P2 Universe integration: dynamic allowlist from Universe Service
        self.universe_enable = os.getenv("UNIVERSE_ENABLE", "true").lower() in ("true", "1", "yes")
        self.universe_cache_seconds = int(os.getenv("UNIVERSE_CACHE_SECONDS", "60"))
        self.universe_last_refresh = 0
        self.allowlist = set(self._load_allowlist())
        
        self.poll_interval = int(os.getenv("APPLY_POLL_SEC", 5))
        self.dedupe_ttl = int(os.getenv("APPLY_DEDUPE_TTL_SEC", 21600))  # 6h
        
        # Safety thresholds
        self.k_block_critical = float(os.getenv("K_BLOCK_CRITICAL", 0.80))
        self.k_block_warning = float(os.getenv("K_BLOCK_WARNING", 0.60))
        
        # Entry/Exit Separation (apply-layer-entry-exit-sep-v1)
        self.k_open_threshold = float(os.getenv("K_OPEN_THRESHOLD", 0.85))
        self.k_close_threshold = float(os.getenv("K_CLOSE_THRESHOLD", 0.65))
        self.k_open_critical = float(os.getenv("K_OPEN_CRITICAL", 0.95))
        self.qty_scale_alpha = float(os.getenv("QTY_SCALE_ALPHA", 2.0))
        self.qty_scale_min = float(os.getenv("QTY_SCALE_MIN", 0.25))
        self.kill_switch = os.getenv("APPLY_KILL_SWITCH", "false").lower() == "true"
        
        # Prometheus metrics
        # Single-source via env: set APPLY_METRICS_PORT; disable with 0/empty
        raw_port = os.getenv("APPLY_METRICS_PORT", "8043").strip()
        self.metrics_port = int(raw_port) if raw_port.isdigit() else 0
        self.setup_metrics()
        
        # P2.8A Heat Observer config
        self.p28_enabled = heat_observer and heat_observer.is_enabled(os.getenv("P28_HEAT_OBS_ENABLED", "true"))
        self.p28_stream = os.getenv("P28_HEAT_OBS_STREAM", "quantum:stream:apply.heat.observed")
        self.p28_lookup_prefix = os.getenv("P28_HEAT_LOOKUP_PREFIX", "quantum:harvest:heat:by_plan:")
        self.p28_dedupe_ttl = int(os.getenv("P28_DEDUPE_TTL_SEC", "600"))
        self.p28_max_debug = int(os.getenv("P28_MAX_DEBUG_CHARS", "400"))
        
        # P2.8A.3 Late Observer config (post-publish delayed observation)
        self.p28_late_enabled = heat_observer and heat_observer.is_enabled(os.getenv("P28_LATE_OBS_ENABLED", "true"))
        self.p28_late_max_wait_ms = int(os.getenv("P28_LATE_OBS_MAX_WAIT_MS", "2000"))
        self.p28_late_poll_ms = int(os.getenv("P28_LATE_OBS_POLL_MS", "100"))
        self.p28_late_max_workers = int(os.getenv("P28_LATE_OBS_MAX_WORKERS", "4"))
        self.p28_late_max_inflight = int(os.getenv("P28_LATE_OBS_MAX_INFLIGHT", "200"))
        
        # Log allowlist source (universe or fallback)
        self._log_allowlist_source()
        
        logger.info(f"ApplyLayer initialized:")
        logger.info(f"  Mode: {self.mode.value}")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Allowlist: {self.allowlist}")
        logger.info(f"  Poll interval: {self.poll_interval}s")
        logger.info(f"  K thresholds: critical={self.k_block_critical}, warning={self.k_block_warning}")
        logger.info(f"  Entry/Exit: open_threshold={self.k_open_threshold}, close_threshold={self.k_close_threshold}, "
                    f"open_critical={self.k_open_critical}, qty_scale_alpha={self.qty_scale_alpha}, qty_scale_min={self.qty_scale_min}")
        logger.info(f"  Dedupe: TTL={DEDUPE_TTL}s (bypass={DEDUPE_BYPASS})")
        logger.info(f"  Kill switch: {self.kill_switch}")
        logger.info(f"  Metrics port: {self.metrics_port}")
        logger.info(f"  P2.8A Heat Observer: {self.p28_enabled}")
        logger.info(f"  P2.8A.3 Late Observer: enabled={self.p28_late_enabled}, max_wait={self.p28_late_max_wait_ms}ms, "
                   f"poll={self.p28_late_poll_ms}ms, max_workers={self.p28_late_max_workers}, "
                   f"max_inflight={self.p28_late_max_inflight}, lookup_prefix={self.p28_lookup_prefix}")
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # If metrics port disabled (<=0), skip starting server
        if getattr(self, "metrics_port", 0) <= 0:
            logger.info("Prometheus metrics disabled (APPLY_METRICS_PORT<=0)")
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
        
        # RECONCILE_CLOSE specific metrics (initialized early, not lazy)
        self.reconcile_close_consumed = Counter(
            'reconcile_close_consumed_total',
            'RECONCILE_CLOSE plans consumed',
            ['symbol']
        )
        self.reconcile_close_executed = Counter(
            'reconcile_close_executed_total',
            'RECONCILE_CLOSE plans executed',
            ['symbol', 'status']
        )
        self.reconcile_close_rejected = Counter(
            'reconcile_close_rejected_total',
            'RECONCILE_CLOSE plans rejected',
            ['symbol', 'reason']
        )
        self.reconcile_guardrail_fail = Counter(
            'reconcile_close_guardrail_fail_total',
            'RECONCILE_CLOSE guardrail failures',
            ['symbol', 'rule']
        )
        
        # Start metrics server (single owner)
        try:
            start_http_server(self.metrics_port)
            logger.info(f"metrics_listen=:{self.metrics_port}")
            logger.info(f"Prometheus metrics available on :{self.metrics_port}/metrics")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    def _load_universe_symbols(self) -> list:
        """Load symbols from Universe Service Redis key (P2 Universe integration)"""
        try:
            meta = self.redis.hgetall("quantum:cfg:universe:meta")
            if not meta:
                logger.warning("Universe meta not found, using fallback")
                return []
            
            stale = int(meta.get("stale", "1"))
            count = int(meta.get("count", "0"))
            asof_epoch = int(meta.get("asof_epoch", "0"))
            
            if stale == 1:
                logger.warning("Universe is stale, using fallback")
                return []
            
            # Get active symbols list
            universe_json = self.redis.get("quantum:cfg:universe:active")
            if not universe_json:
                logger.warning("Universe active key missing, using fallback")
                return []
            
            try:
                universe = json.loads(universe_json)
                symbols = universe.get("symbols", [])
                logger.debug(f"Universe loaded: {len(symbols)} symbols (asof_epoch={asof_epoch})")
                return symbols
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse universe JSON: {e}")
                return []
                
        except Exception as e:
            logger.warning(f"Error loading universe: {e}")
            return []
    
    def _load_allowlist(self) -> list:
        """Load allowlist from Universe or fallback to APPLY_ALLOWLIST env (P2 Universe integration)"""
        if self.universe_enable:
            symbols = self._load_universe_symbols()
            if symbols:
                return symbols
            else:
                logger.warning("Universe unavailable, falling back to APPLY_ALLOWLIST")
        
        # Fallback to env var
        allowlist_str = os.getenv("APPLY_ALLOWLIST", "BTCUSDT")
        return [s.strip() for s in allowlist_str.split(",") if s.strip()]
    
    def _get_policy_universe(self) -> set:
        """
        Get policy universe from PolicyStore (fail-closed gate before placing order).
        Returns intersection of policy symbols and tradable symbols.
        """
        if not POLICY_ENABLED:
            logger.error("🔥 POLICY GATE: PolicyStore not enabled, cannot get allowlist")
            return set()
        
        try:
            policy = load_policy()
            if not policy or policy.is_stale():
                logger.error("🔥 POLICY GATE: Policy unavailable or stale")
                return set()
            
            policy_symbols = set(policy.universe_symbols)
            if not policy_symbols:
                logger.error("🔥 POLICY GATE: Policy universe is empty")
                return set()
            
            logger.debug(f"POLICY_GATE: Loaded {len(policy_symbols)} symbols from policy")
            return policy_symbols
            
        except Exception as e:
            logger.error(f"🔥 POLICY_GATE: Failed to load policy: {e}")
            return set()
    
    def _check_symbol_allowlist(self, symbol: str) -> bool:
        """
        HARD GATE: Check if symbol is in policy universe before order placement.
        Fail-closed: if symbol not in policy → REJECT order.
        """
        policy_universe = self._get_policy_universe()
        
        if not policy_universe:
            logger.error(f"🔥 DENY_SYMBOL_NOT_IN_ALLOWLIST symbol={symbol} reason=empty_policy_universe")
            return False
        
        if symbol not in policy_universe:
            logger.warning(
                f"🔥 DENY_SYMBOL_NOT_IN_ALLOWLIST symbol={symbol} "
                f"reason=symbol_not_in_policy policy_count={len(policy_universe)} "
                f"policy_sample={sorted(list(policy_universe))}"
            )
            return False
        
        logger.debug(f"SYMBOL_ALLOWED: {symbol} in policy universe")
        return True
    
    def _refresh_allowlist_if_needed(self):
        """Refresh allowlist cache if expired (P2 Universe integration)"""
        now = time.time()
        if now - self.universe_last_refresh >= self.universe_cache_seconds:
            new_allowlist = set(self._load_allowlist())
            if new_allowlist != self.allowlist:
                logger.info(f"Allowlist updated: {len(self.allowlist)} → {len(new_allowlist)} symbols")
                self.allowlist = new_allowlist
                self._log_allowlist_source()
            self.universe_last_refresh = now
    
    def _log_allowlist_source(self):
        """Log allowlist source (universe or fallback) with metadata (P2 Universe integration)"""
        if not self.universe_enable:
            logger.info(f"Apply allowlist source=env count={len(self.allowlist)}")
            return
        
        try:
            meta = self.redis.hgetall("quantum:cfg:universe:meta")
            if meta:
                stale = int(meta.get("stale", "1"))
                count = int(meta.get("count", "0"))
                asof_epoch = int(meta.get("asof_epoch", "0"))
                
                if stale == 0 and count > 0:
                    logger.info(f"Apply allowlist source=universe stale=0 count={count} asof_epoch={asof_epoch}")
                else:
                    logger.warning(f"Apply allowlist source=fallback reason=stale={stale} count={count}")
            else:
                logger.warning(f"Apply allowlist source=fallback reason=meta_missing count={len(self.allowlist)}")
        except Exception as e:
            logger.warning(f"Error logging allowlist source: {e}")
    
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
            
            # P0.FIX: Read calibrated action if available (from P2.6 Heat Gate)
            # If calibrated=1, use "action" field (calibrated), else fall back to "harvest_action"
            is_calibrated = data.get("calibrated") == "1"
            if is_calibrated and data.get("action"):
                action = data.get("action")
                logger.debug(f"{symbol}: Using calibrated action={action} (original={data.get('original_action', 'N/A')})")
            else:
                action = data.get("harvest_action")
            
            proposal = {
                "harvest_action": action,  # Use calibrated action if available
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
                "p26_calibrated": is_calibrated,  # Track if calibrated
                "p26_original_action": data.get("original_action") if is_calibrated else None,
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
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if symbol has an active position (fail-soft UPDATE_SL guard)
        
        Checks Redis position snapshot (quantum:position:snapshot:<symbol>)
        Returns: True if position exists with abs(position_amt) > POSITION_EPSILON
        
        Handles:
        - SHORT positions (negative position_amt)
        - Empty string or missing field (treated as 0.0)
        - Redis errors (fail-soft to False)
        """
        try:
            key = f"quantum:position:snapshot:{symbol}"
            data = self.redis.hgetall(key)
            if not data:
                return False
            
            # Handle missing field, empty string, or None
            position_amt_raw = data.get("position_amt", "0")
            if not position_amt_raw or position_amt_raw == "":
                position_amt = 0.0
            else:
                position_amt = float(position_amt_raw)
            
            # Use abs() to handle SHORT positions (negative amounts)
            has_pos = abs(position_amt) > POSITION_EPSILON
            return has_pos
        except (ValueError, TypeError) as e:
            logger.warning(f"{symbol}: Position parse failed: {e}")
            return False  # Fail-soft: assume no position
        except Exception as e:
            logger.warning(f"{symbol}: Position check failed: {e}")
            return False  # Fail-soft: assume no position
    
    def normalize_action(self, action: str, proposal: Dict[str, Any], symbol: str) -> tuple[str, Optional[str]]:
        """
        Normalize action field to standard values (apply-layer-entry-exit-sep-v1)
        
        Handles:
        - UNKNOWN action → infer from proposal fields
        - Synonyms (ENTRY/ENTER/OPEN → OPEN, EXIT/REDUCE/CLOSE → CLOSE)
        - Missing action → default to HOLD
        
        Returns: (normalized_action, reason_code_or_none)
        """
        original_action = action
        reason_code = None
        
        # Handle UNKNOWN or None
        if not action or action == "UNKNOWN":
            # Infer from proposal fields
            if proposal.get("new_sl_proposed"):
                # Has SL update → CHECK if position exists (fail-soft guard)
                if self.has_position(symbol):
                    action = "UPDATE_SL"
                    reason_code = f"action_normalized_unknown_to_update_sl"
                    logger.info(f"ACTION_NORMALIZED {symbol}: from={original_action} to={action} (has new_sl_proposed={proposal['new_sl_proposed']:.2f})")
                else:
                    # No position → HOLD (fail-soft)
                    action = "HOLD"
                    reason_code = f"update_sl_no_position_skip"
                    logger.info(f"UPDATE_SL_SKIP_NO_POSITION {symbol}: proposed_sl={proposal['new_sl_proposed']:.2f} (no_position)")
            else:
                # No action indicators → HOLD (safe default)
                action = "HOLD"
                reason_code = f"action_normalized_unknown_to_hold"
                logger.info(f"ACTION_NORMALIZED {symbol}: from={original_action} to={action} (no_indicators)")
            return action, reason_code
        
        # Normalize synonyms
        action_upper = action.upper()
        
        # CLOSE synonyms
        if action_upper in ["EXIT", "REDUCE", "CLOSE", "FULL_CLOSE"]:
            action = "FULL_CLOSE_PROPOSED"
            reason_code = f"action_normalized_{original_action.lower()}_to_close"
            logger.info(f"ACTION_NORMALIZED {symbol}: from={original_action} to={action} (close_synonym)")
        
        # OPEN synonyms (not used in harvest context, but handle for robustness)
        elif action_upper in ["ENTRY", "ENTER", "OPEN"]:
            action = "HOLD"  # Harvest layer doesn't open positions, map to HOLD
            reason_code = f"action_normalized_{original_action.lower()}_to_hold"
            logger.info(f"ACTION_NORMALIZED {symbol}: from={original_action} to={action} (open_synonym_ignored)")
        
        # Already standard actions (pass through)
        elif action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50", "UPDATE_SL", "HOLD"]:
            pass  # No normalization needed, reason_code=None
        
        else:
            # Unknown action variant → default to HOLD (fail-soft)
            logger.warning(f"ACTION_NORMALIZED {symbol}: from={original_action} to=HOLD (unknown_variant_fail_soft)")
            action = "HOLD"
            reason_code = f"action_normalized_{original_action.lower()}_unknown_variant"
        
        return action, reason_code
    
    def check_idempotency(self, plan_id: str) -> bool:
        """Check if plan already executed (returns True if duplicate)"""
        key = f"quantum:apply:dedupe:{plan_id}"
        result = self.redis.setnx(key, int(time.time()))
        if result == 1:
            # New plan, set TTL (use DEDUPE_TTL for dev bypass support)
            self.redis.expire(key, DEDUPE_TTL)
            return False
        else:
            # Duplicate
            if PROMETHEUS_AVAILABLE:
                self.metric_dedupe_hits.inc()
            return True
    
    def create_apply_plan(self, symbol: str, proposal: Dict[str, Any]) -> ApplyPlan:
        """Create apply plan from harvest proposal"""
        plan_id = self.create_plan_id(symbol, proposal)
        
        # Extract source (for exit ownership enforcement)
        # Default to exitbrain_v3_5 if not specified (all harvest proposals come from exitbrain)
        source = proposal.get("source", EXIT_OWNER)
        
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
            # Entry/Exit Separation (apply-layer-entry-exit-sep-v1)
            action = proposal["harvest_action"]
            
            # Detect action type
            is_close_action = action in ["FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50"]
            
            if is_close_action:
                # CLOSE: use stricter threshold (fail-closed)
                threshold = self.k_close_threshold
                if proposal["kill_score"] >= threshold:
                    decision = Decision.BLOCKED
                    reason_codes.append("kill_score_close_blocked")
                    logger.warning(f"{symbol}: CLOSE blocked kill_score={proposal['kill_score']:.3f} >= threshold={threshold:.3f} (regime_flip={proposal.get('k_regime_flip',0):.2f})")
                else:
                    decision = Decision.EXECUTE
                    reason_codes.append("kill_score_close_ok")
                    logger.info(f"{symbol}: CLOSE allowed kill_score={proposal['kill_score']:.3f} < threshold={threshold:.3f}")
            else:
                # OPEN: use permissive threshold + qty_scale (fail-soft)
                threshold = self.k_open_threshold
                
                if proposal["kill_score"] >= self.k_open_critical:
                    # Extreme kill_score → hard block
                    decision = Decision.BLOCKED
                    reason_codes.append("kill_score_open_critical")
                    logger.warning(f"{symbol}: OPEN blocked (critical) kill_score={proposal['kill_score']:.3f} >= critical={self.k_open_critical:.3f}")
                elif proposal["kill_score"] >= threshold:
                    # High kill_score → allow with qty_scale
                    excess = proposal["kill_score"] - threshold
                    qty_scale = math.exp(-self.qty_scale_alpha * excess)
                    qty_scale = max(self.qty_scale_min, min(1.0, qty_scale))
                    
                    proposal["qty_scale"] = qty_scale  # Store for downstream
                    decision = Decision.EXECUTE
                    reason_codes.append("kill_score_open_scaled")
                    logger.info(f"{symbol}: OPEN scaled kill_score={proposal['kill_score']:.3f} threshold={threshold:.3f} qty_scale={qty_scale:.2f} (regime_flip={proposal.get('k_regime_flip',0):.2f})")
                else:
                    # Normal kill_score → allow full size
                    decision = Decision.EXECUTE
                    reason_codes.append("kill_score_open_ok")
                    logger.info(f"{symbol}: OPEN allowed kill_score={proposal['kill_score']:.3f} < threshold={threshold:.3f}")
        
        # Safety gate 4: Idempotency
        if decision == Decision.EXECUTE:
            if self.check_idempotency(plan_id):
                decision = Decision.SKIP
                reason_codes.append("duplicate_plan")
                logger.info(f"{symbol}: Plan {plan_id} already executed (duplicate)")
        
        # Build execution steps
        if decision == Decision.EXECUTE:
            # Normalize action before building steps (apply-layer-entry-exit-sep-v1)
            action, norm_reason = self.normalize_action(proposal["harvest_action"], proposal, symbol)
            if norm_reason:
                reason_codes.append(norm_reason)
            
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
        
        plan = ApplyPlan(
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
            source=source,
            timestamp=int(time.time())
        )
        
        # P2.8A: Shadow observation (read HeatBridge, emit observed stream)
        # FAIL-OPEN: Never blocks, never changes decisions, errors logged
        if self.p28_enabled and heat_observer:
            try:
                heat_observer.observe(
                    redis_client=self.redis,
                    plan_id=plan_id,
                    symbol=symbol,
                    apply_decision=decision.value,
                    obs_point="create_apply_plan",
                    enabled=self.p28_enabled,
                    stream_name=self.p28_stream,
                    lookup_prefix=self.p28_lookup_prefix,
                    dedupe_ttl_sec=self.p28_dedupe_ttl,
                    max_debug_chars=self.p28_max_debug
                )
            except Exception as e:
                # Catch-all to protect Apply (fail-open)
                logger.warning(f"{symbol}: Heat observer error (ignored): {e}")
        
        return plan
    
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
            
            # P2.8A.3: Late observer (post-publish delayed observation)
            # WHY: Observer at create_apply_plan runs BEFORE publish ΓåÆ HeatBridge hasn't written by_plan yet
            #      This late observer runs AFTER publish ΓåÆ HeatBridge has time to write by_plan key
            if self.p28_late_enabled and heat_observer:
                try:
                    heat_observer.observe_late_async(
                        redis_client=self.redis,
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        lookup_prefix=self.p28_lookup_prefix,  # REQUIRED: same prefix as early observer
                        apply_decision=plan.decision,
                        obs_stream=self.p28_stream,
                        max_wait_ms=self.p28_late_max_wait_ms,
                        poll_ms=self.p28_late_poll_ms,
                        dedupe_ttl_sec=self.p28_dedupe_ttl,
                        max_debug_chars=self.p28_max_debug,
                        obs_point="publish_plan_post",
                        logger=logger,
                        max_workers=self.p28_late_max_workers,
                        max_inflight=self.p28_late_max_inflight
                    )
                except Exception as e:
                    # Fail-open: don't crash Apply if late observer fails
                    logger.debug(f"{plan.symbol}: Late observer spawn failed: {e}")
            
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
                    logger.info(f"{plan.symbol}: [DRY_RUN] Governor permit granted Γ£ô")
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

    def _learning_plane_ok(self) -> Tuple[bool, str]:
        """Legacy method - now redirects to risk enforcer if available."""
        if self.risk_enforcer:
            # Use risk enforcer for comprehensive LAYER 0 check
            metrics = self.risk_enforcer.compute_system_state()
            if metrics.system_state == SystemState.NO_GO:
                return False, metrics.failure_reason or "layer0_fail"
            return True, "ok"
        
        # Fallback: original heartbeat check if enforcer not available
        try:
            self.redis.ping()
        except Exception:
            return False, "redis_inactive"

        now = int(time.time())

        for key, label in (
            (self.rl_feedback_heartbeat_key, "rl_feedback"),
            (self.rl_trainer_heartbeat_key, "rl_trainer"),
        ):
            val = self.redis.get(key)
            if not val:
                return False, f"{label}_heartbeat_missing"
            try:
                ts = int(float(val))
            except Exception:
                return False, f"{label}_heartbeat_invalid"
            if now - ts > self.learning_plane_ttl:
                return False, f"{label}_heartbeat_stale"

        return True, "ok"
    
    def execute_testnet(self, plan: ApplyPlan) -> ApplyResult:
        """Testnet execution - REAL orders to Binance Futures testnet"""
        logger.error(
            "[APPLY_TRACE][ENTER] plan_id=%s symbol=%s side=%s qty=%s has_metadata=%s",
            getattr(plan, "id", None),
            getattr(plan, "symbol", None),
            getattr(plan, "side", None),
            getattr(plan, "qty", None),
            hasattr(plan, "metadata"),
        )
        # ---- RISK POLICY ENFORCEMENT (LAYER 0-2) ----
        if self.risk_enforcer:
            # Get requested leverage from plan metadata
            logger.error(
                "[APPLY_TRACE][PRE-METADATA] about to access plan.metadata, dir(plan)=%s",
                dir(plan),
            )
            # --- D1 FAIL-SAFE METADATA ---
            metadata = getattr(plan, "metadata", None)
            if metadata is None:
                metadata = {}
                logger.warning(
                    "[APPLY_TRACE][D1] plan.metadata missing → using defaults "
                    f"(plan_id={plan.plan_id}, symbol={plan.symbol})"
                )
            requested_leverage = metadata.get("target_leverage", 1.0)
            
            # Comprehensive risk gate
            allowed, system_state, risk_reason = self.risk_enforcer.allow_trade(
                symbol=plan.symbol,
                requested_leverage=requested_leverage,
                volatility=metadata.get("volatility"),
                spread_bps=metadata.get("spread_bps")
            )
            
            if not allowed:
                logger.critical(f"[RISK_POLICY] Execution blocked - {system_state.value}: {risk_reason}")
                
                # Handle failure by type
                if system_state == SystemState.BOOTING:
                    logger.info("System booting – trading disabled")
                    try:
                        self.redis.incr("quantum:metrics:trades_blocked")
                    except Exception:
                        pass
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision=plan.decision,
                        executed=False,
                        would_execute=False,
                        steps_results=[],
                        error=f"risk_booting:{risk_reason}",
                        timestamp=int(time.time())
                    )

                if system_state == SystemState.NO_GO:
                    try:
                        self.redis.incr("quantum:metrics:trades_blocked")
                    except Exception:
                        pass
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision=plan.decision,
                        executed=False,
                        would_execute=False,
                        steps_results=[],
                        error=f"risk_layer0_fail:{risk_reason}",
                        timestamp=int(time.time())
                    )
                
                elif system_state == SystemState.PAUSED:
                    # LAYER 1/2 failure - trading paused
                    logger.warning(f"PAUSED: {risk_reason}")
                    try:
                        self.redis.incr("quantum:metrics:paused_events")
                        self.redis.incr("quantum:metrics:trades_blocked")
                    except Exception:
                        pass
                    return ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision=plan.decision,
                        executed=False,
                        would_execute=False,
                        steps_results=[],
                        error=f"risk_paused:{risk_reason}",
                        timestamp=int(time.time())
                    )
        else:
            # Fallback: legacy learning plane check if enforcer not available
            ok, reason = self._learning_plane_ok()
            if not ok:
                logger.critical(f"[LEARNING_PLANE] Execution halted - {reason}")
                try:
                    self.redis.incr("quantum:metrics:trades_blocked")
                except Exception:
                    pass
                return ApplyResult(
                    plan_id=plan.plan_id,
                    symbol=plan.symbol,
                    decision=plan.decision,
                    executed=False,
                    would_execute=False,
                    steps_results=[],
                    error=f"learning_plane_down:{reason}",
                    timestamp=int(time.time())
                )

        # ---- PRODUCTION HYGIENE: Safety Kill Switch Check ----
        try:
            kill_switch = self.redis.get(SAFETY_KILL_KEY)
            if kill_switch and kill_switch.lower() in (b"true", b"1", b"yes"):
                logger.critical(f"[KILL_SWITCH] Execution halted - kill switch is ACTIVE")
                try:
                    self.redis.incr("quantum:metrics:kill_switch_events")
                    self.redis.incr("quantum:metrics:trades_blocked")
                except Exception:
                    pass
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

        try:
            self.redis.incr("quantum:metrics:trades_allowed")
        except Exception:
            pass
        
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
                p26_permit = {"granted": True, "mode": "testnet_bypass"}
                ok = True
                wait_ms = 0
                if PROMETHEUS_AVAILABLE:
                    apply_executed.labels(status='testnet_bypass').inc()
            else:
                # PRODUCTION: Require THREE permits (Governor + P3.3 + P2.6)
                t0 = time.time()
                consume_script = _register_consume_script(self.redis)
                ok, gov_permit, p33_permit, p26_permit = wait_and_consume_permits(
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
                        
                        # EXIT OWNERSHIP GATE: Only authorized services can place reduceOnly orders
                        if EXIT_OWNERSHIP_ENABLED:
                            if plan.source != EXIT_OWNER:
                                logger.warning(
                                    f"DENY_NOT_EXIT_OWNER symbol={plan.symbol} source={plan.source} "
                                    f"plan_id={plan.plan_id} action={step['step']} expected={EXIT_OWNER}"
                                )
                                return ApplyResult(
                                    plan_id=plan.plan_id,
                                    symbol=plan.symbol,
                                    decision="DENIED",
                                    executed=False,
                                    would_execute=False,
                                    steps_results=[{
                                        "step": step["step"],
                                        "status": "denied",
                                        "details": f"DENY_NOT_EXIT_OWNER: source={plan.source} not authorized (expected={EXIT_OWNER})"
                                    }],
                                    error="DENY_NOT_EXIT_OWNER",
                                    timestamp=int(time.time())
                                )
                            else:
                                logger.info(f"ALLOW_EXIT_OWNER symbol={plan.symbol} source={plan.source} (authorized)")
                        
                        logger.info(f"{plan.symbol}: Placing {order_side} order for {close_qty} (reduceOnly)")
                        
                        logger.error(
                            "[APPLY_TRACE][PRE-EXCHANGE] sending order symbol=%s side=%s qty=%s",
                            plan.symbol,
                            order_side,
                            close_qty,
                        )
                        # Place market order with reduceOnly
                        order_result = client.place_market_order(
                            symbol=plan.symbol,
                            side=order_side,
                            quantity=close_qty,
                            reduce_only=True
                        )
                        logger.error("[APPLY_TRACE][POST-EXCHANGE] order submitted OK")
                        
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
            logger.exception(
                "[APPLY_TRACE][EXCEPTION] type=%s msg=%s",
                type(e).__name__,
                str(e),
            )
            raise
    
    def validate_reconcile_close_guardrails(self, plan_data: Dict) -> tuple[bool, str]:
        """
        STRICT INVARIANT VALIDATION - Trading backdoor protection.
        
        All checks are HARD FAIL - any violation prevents execution.
        This ensures RECONCILE_CLOSE cannot become a trading backdoor.
        
        Invariants:
        0. HMAC signature valid (prevents unauthorized Redis writes)
        1. decision == "RECONCILE_CLOSE"
        2. source == "p3.4" (only P3.4 Reconcile Engine can publish)
        3. HOLD key == 1 for symbol (prevents trading when not in drift)
        4. reduceOnly == true (string/boolean handled robustly)
        5. type == "MARKET" only
        6. qty > 0 and qty <= abs(exchange_amt)
        7. reason == "reconcile_drift"
        
        Returns: (is_safe, error_message)
        """
        symbol = plan_data.get('symbol')
        decision = plan_data.get('decision')
        source = plan_data.get('source')
        reduce_only = plan_data.get('reduceOnly')
        order_type = plan_data.get('type')
        reason = plan_data.get('reason')
        plan_hmac = plan_data.get('hmac', '')
        plan_id = plan_data.get('plan_id')
        qty = plan_data.get('qty', '0')
        signature = plan_data.get('signature', '')
        ts = plan_data.get('ts', '')
        
        # INVARIANT 0: HMAC signature must be valid (SECURITY FINAL BOSS)
        # Prefer RECONCILE_HMAC_SECRET; fallback to legacy RECONCILE_CLOSE_SECRET
        secret = (
            os.getenv("RECONCILE_HMAC_SECRET")
            or os.getenv("RECONCILE_CLOSE_SECRET", "quantum_reconcile_secret_change_in_prod")
        )
        
        # Extract timestamp from plan_id if not in separate field
        if not ts and ':' in plan_id:
            parts = plan_id.split(':')
            if len(parts) >= 4:
                ts = parts[3]
        
        # Compute expected HMAC
        hmac_payload = f"{plan_id}|{symbol}|{qty}|{ts}|{signature}"
        expected_hmac = hmac.new(secret.encode(), hmac_payload.encode(), hashlib.sha256).hexdigest()
        
        if not plan_hmac or plan_hmac != expected_hmac:
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='hmac').inc()
            return False, f"INVARIANT: HMAC signature invalid or missing"
        
        # INVARIANT 1: decision must be RECONCILE_CLOSE
        if decision != 'RECONCILE_CLOSE':
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='decision').inc()
            return False, f"INVARIANT: decision must be RECONCILE_CLOSE, got {decision}"
        
        # INVARIANT 2: source must be p3.4 (security: prevent arbitrary Redis writes)
        if source != 'p3.4':
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='source').inc()
            return False, f"INVARIANT: source must be 'p3.4', got {source}"
        
        # INVARIANT 3: HOLD key must be active (1 = active)
        hold_key = f"quantum:reconcile:hold:{symbol}"
        hold_value = self.redis.get(hold_key)
        if not hold_value:
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='hold_key').inc()
            return False, f"INVARIANT: HOLD key not active for {symbol}"
        # Handle both bytes and str
        hold_str = hold_value.decode('utf-8') if isinstance(hold_value, bytes) else str(hold_value)
        if hold_str != '1':
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='hold_key').inc()
            return False, f"INVARIANT: HOLD key not active for {symbol} (value={hold_str})"
        
        # INVARIANT 3: reduceOnly must be true (handle string/boolean robustly)
        if isinstance(reduce_only, str):
            reduce_only = reduce_only.lower() in ('true', '1')
        elif not isinstance(reduce_only, bool):
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='reduce_only_type').inc()
            return False, f"INVARIANT: reduceOnly invalid type {type(reduce_only)}"
        if not reduce_only:
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='reduce_only').inc()
            return False, "INVARIANT: reduceOnly must be true"
        
        # INVARIANT 5: Order type must be MARKET
        if order_type != 'MARKET':
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='order_type').inc()
            return False, f"INVARIANT: type must be MARKET, got {order_type}"
        
        # INVARIANT 6: qty must be safe (positive and <= exchange position)
        try:
            exchange_amt = float(plan_data.get('exchange_amt', 0))
            qty = float(plan_data.get('qty', 0))
            if qty <= 0:
                self.reconcile_guardrail_fail.labels(symbol=symbol, rule='qty_positive').inc()
                return False, f"INVARIANT: qty must be positive, got {qty}"
            if exchange_amt != 0 and qty > abs(exchange_amt):
                self.reconcile_guardrail_fail.labels(symbol=symbol, rule='qty_safe').inc()
                return False, f"INVARIANT: qty {qty} exceeds exchange position {exchange_amt}"
        except (ValueError, TypeError) as e:
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='qty_parse').inc()
            return False, f"INVARIANT: invalid qty or exchange_amt: {e}"
        
        # INVARIANT 7: reason must be reconcile_drift
        if reason != 'reconcile_drift':
            self.reconcile_guardrail_fail.labels(symbol=symbol, rule='reason').inc()
            return False, f"INVARIANT: reason must be 'reconcile_drift', got {reason}"
        
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
        exchange_amt = plan_data.get('exchange_amt', '0')
        signature = plan_data.get('signature', 'unknown')
        
        self.reconcile_close_consumed.labels(symbol=symbol).inc()
        
        logger.warning(f"[RECONCILE_CLOSE] {symbol}: Execution started - plan_id={plan_id[:16]}, side={side}, qty={qty}")
        
        # EXACTLY-ONCE: Check deduplication to prevent replays
        dedupe_key = f"quantum:apply:dedupe:{plan_id}"
        dedupe_set = self.redis.set(dedupe_key, "1", ex=86400, nx=True)
        dedupe_hit = not dedupe_set
        
        if dedupe_hit:
            logger.warning(f"[RECONCILE_CLOSE] {symbol}: Duplicate plan_id {plan_id[:16]} - dropping")
            self.reconcile_close_rejected.labels(symbol=symbol, reason='duplicate').inc()
            return ApplyResult(
                plan_id=plan_id,
                symbol=symbol,
                decision="RECONCILE_CLOSE",
                executed=False,
                would_execute=False,
                steps_results=[{
                    "step": "DEDUPLICATION",
                    "status": "DROPPED_DUPLICATE",
                    "dedupe_hit": True
                }],
                error="Duplicate plan_id",
                timestamp=int(time.time())
            )
        
        # LEASE RENEWAL: Extend HOLD lease now that we're processing
        # This prevents HOLD from expiring during rate limits or backlog
        lease_sec = 900
        self.redis.expire(f"quantum:reconcile:hold:{symbol}", lease_sec)
        self.redis.expire(f"quantum:reconcile:hold_reason:{symbol}", lease_sec)
        self.redis.expire(f"quantum:reconcile:hold_sig:{symbol}", lease_sec)
        logger.info(f"[RECONCILE_CLOSE] {symbol}: HOLD lease renewed (TTL={lease_sec}s)")
        
        # Validate guardrails (STRICT INVARIANTS)
        is_safe, error = self.validate_reconcile_close_guardrails(plan_data)
        if not is_safe:
            logger.error(f"[RECONCILE_CLOSE] {symbol}: INVARIANT VIOLATION: {error}")
            self.reconcile_close_rejected.labels(symbol=symbol, reason='guardrails').inc()
            return ApplyResult(
                plan_id=plan_id,
                symbol=symbol,
                decision="RECONCILE_CLOSE",
                executed=False,
                would_execute=False,
                steps_results=[{
                    "step": "GUARDRAILS",
                    "status": "REJECTED_GUARDRAIL",
                    "reason": error,
                    "dedupe_hit": dedupe_hit
                }],
                error=error,
                timestamp=int(time.time())
            )
        
        try:
            # Get API credentials
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
            
            if not api_key or not api_secret:
                raise Exception("Missing Binance credentials")
            
            # Create Binance client
            client = BinanceTestnetClient(api_key, api_secret)
            
            # Place market order with reduceOnly
            logger.warning(f"[RECONCILE_CLOSE] {symbol}: Placing order - side={side}, qty={qty}, reduceOnly=true")
            result = client.place_market_order(symbol, side, qty, reduce_only=True)
            
            if result.get('orderId'):
                order_id = result['orderId']
                
                # AUDIT LOG: Critical reconcile_close execution
                hold_key = f"quantum:reconcile:hold:{symbol}"
                logger.warning(
                    f"[RECONCILE_CLOSE_AUDIT] reconcile_close=true symbol={symbol} qty={qty} "
                    f"side={side} order_id={order_id} plan_id={plan_id[:16]} "
                    f"hold_key={hold_key} exchange_amt={exchange_amt}"
                )
                
                logger.warning(f"[RECONCILE_CLOSE_EXECUTED] {symbol}: plan_id={plan_id[:16]}, order_id={order_id}, qty={qty}")
                
                self.reconcile_close_executed.labels(symbol=symbol, status='success').inc()
                
                if PROMETHEUS_AVAILABLE:
                    apply_executed.labels(status='reconcile_close_success').inc()
                
                return ApplyResult(
                    plan_id=plan_id,
                    symbol=symbol,
                    decision="RECONCILE_CLOSE",
                    executed=True,
                    would_execute=False,
                    steps_results=[{
                        "step": "PLACE_ORDER",
                        "status": "EXECUTED",
                        "order_id": order_id,
                        "qty": qty,
                        "dedupe_hit": dedupe_hit
                    }],
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
            error_str = str(e)
            
            # Handle rate limiting (429) specially - don't spam, HOLD remains
            if '429' in error_str or 'too many requests' in error_str.lower() or 'banned' in error_str.lower():
                logger.error(f"[RECONCILE_CLOSE] {symbol}: RATE LIMITED (429) - HOLD remains, will retry on next cooldown bucket")
                self.reconcile_close_executed.labels(symbol=symbol, status='rate_limit').inc()
                return ApplyResult(
                    plan_id=plan_id,
                    symbol=symbol,
                    decision="RECONCILE_CLOSE",
                    executed=False,
                    would_execute=True,
                    steps_results=[{"step": "EXECUTION", "status": "RETRYABLE_RATE_LIMIT", "error": error_str}],
                    error=f"RETRYABLE_RATE_LIMIT: {error_str}",
                    timestamp=int(time.time())
                )
            
            # Other execution errors
            logger.error(f"[RECONCILE_CLOSE] {symbol}: Execution error: {e}", exc_info=True)
            self.reconcile_close_executed.labels(symbol=symbol, status='error').inc()
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
    
    def process_apply_plan_stream(self):
        """Process entry intents from intent_bridge apply.plan stream"""
        try:
            stream_key = "quantum:stream:apply.plan"
            consumer_group = "apply_layer_entry"
            consumer_id = f"apply-entry-{os.getpid()}"
            
            # Create consumer group (idempotent)
            try:
                self.redis.xgroup_create(stream_key, consumer_group, id='0-0', mkstream=True)
                logger.info(f"[ENTRY] Created consumer group {consumer_group}")
            except:
                pass  # Group already exists
            
            # Read apply.plan messages (entry intents from intent_bridge)
            messages = self.redis.xreadgroup(
                groupname=consumer_group,
                consumername=consumer_id,
                streams={stream_key: '>'},
                count=10,
                block=100  # Short block
            )
            
            if not messages:
                return
            
            logger.info(f"[ENTRY] Processing {len(messages)} stream(s) with apply.plan messages")
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    try:
                        # Convert fields to plain dict
                        plan_data = {}
                        for k, v in fields.items():
                            key = k.decode() if isinstance(k, bytes) else k
                            val = v.decode() if isinstance(v, bytes) else v
                            plan_data[key] = val
                        
                        symbol = plan_data.get('symbol', '')
                        action = plan_data.get('action', '').upper()
                        plan_id = plan_data.get('plan_id', '')
                        
                        # ========== CLOSE HANDLER ==========
                        # Process FULL_CLOSE_PROPOSED, PARTIAL_* actions (reduceOnly market orders)
                        if action in ['FULL_CLOSE_PROPOSED', 'PARTIAL_25', 'PARTIAL_50', 'PARTIAL_75']:
                            logger.info(f"[CLOSE] {symbol}: Processing {action} plan_id={plan_id[:8]}")
                            
                            # Idempotency: Check if already executed
                            dedupe_key = f"quantum:apply:done:{plan_id}"
                            if self.redis.exists(dedupe_key):
                                logger.warning(f"[CLOSE] {symbol}: SKIP_DUPLICATE plan_id={plan_id[:8]} (already executed)")
                                self.redis.xack(stream_key, consumer_group, msg_id)
                                # Publish skip result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'False',
                                    'error': 'duplicate_plan',
                                    'timestamp': str(int(time.time()))
                                })
                                continue
                            
                            # Get current position
                            pos_key = f"quantum:position:{symbol}"
                            existing_pos = self.redis.hgetall(pos_key)
                            if not existing_pos:
                                logger.warning(f"[CLOSE] {symbol}: SKIP_NO_POSITION plan_id={plan_id[:8]} (no position exists)")
                                self.redis.xack(stream_key, consumer_group, msg_id)
                                # Set dedupe marker
                                self.redis.setex(dedupe_key, 600, "1")  # 10 min TTL
                                # Publish skip result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'False',
                                    'error': 'no_position',
                                    'timestamp': str(int(time.time()))
                                })
                                continue
                            
                            # Parse position data
                            position_side = existing_pos.get(b'side', existing_pos.get('side', b'')).decode() if isinstance(existing_pos.get(b'side', existing_pos.get('side', b'')), bytes) else existing_pos.get(b'side', existing_pos.get('side', ''))
                            position_qty_str = existing_pos.get(b'quantity', existing_pos.get('quantity', b'0')).decode() if isinstance(existing_pos.get(b'quantity', existing_pos.get('quantity', b'0')), bytes) else existing_pos.get(b'quantity', existing_pos.get('quantity', '0'))
                            position_qty = float(position_qty_str)
                            
                            # Parse steps from plan to get close percentage
                            steps_str = plan_data.get('steps', '[]')
                            try:
                                steps = json.loads(steps_str)
                            except:
                                steps = []
                            
                            # Find market_reduce_only step
                            close_pct = 100.0  # Default to full close
                            for step in steps:
                                if step.get('type') == 'market_reduce_only':
                                    close_pct = float(step.get('pct', 100.0))
                                    break
                            
                            # Compute close_qty
                            close_qty = abs(position_qty) * (close_pct / 100.0)
                            
                            if close_qty <= 0.0:
                                logger.warning(f"[CLOSE] {symbol}: SKIP_CLOSE_QTY_ZERO plan_id={plan_id[:8]} pct={close_pct} pos_qty={position_qty}")
                                self.redis.xack(stream_key, consumer_group, msg_id)
                                # Set dedupe marker
                                self.redis.setex(dedupe_key, 600, "1")
                                # Publish skip result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'False',
                                    'error': 'close_qty_zero',
                                    'close_qty': '0.0',
                                    'timestamp': str(int(time.time()))
                                })
                                continue
                            
                            # Determine close side (opposite of position)
                            if position_side == 'LONG':
                                close_side = 'SELL'  # Close LONG with SELL
                            elif position_side == 'SHORT':
                                close_side = 'BUY'   # Close SHORT with BUY
                            else:
                                logger.error(f"[CLOSE] {symbol}: Invalid position_side={position_side}")
                                self.redis.xack(stream_key, consumer_group, msg_id)
                                continue
                            
                            # Execute reduceOnly market order
                            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
                            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
                            
                            if not api_key or not api_secret:
                                logger.warning(f"[CLOSE] {symbol}: Missing Binance testnet credentials")
                                self.redis.xack(stream_key, consumer_group, msg_id)
                                continue
                            
                            try:
                                client = BinanceTestnetClient(api_key, api_secret)
                                
                                logger.info(f"[CLOSE_EXECUTE] plan_id={plan_id[:8]} symbol={symbol} side={close_side} qty={close_qty:.6f} reduceOnly=true pct={close_pct}%")
                                
                                # Place reduceOnly market order
                                order_result = client.place_market_order(
                                    symbol=symbol,
                                    side=close_side,
                                    quantity=close_qty,
                                    reduce_only=True
                                )
                                
                                filled_qty = float(order_result.get('executedQty', close_qty))
                                order_id = order_result.get('orderId', '')
                                status = order_result.get('status', 'UNKNOWN')
                                
                                logger.info(f"[CLOSE_DONE] plan_id={plan_id[:8]} symbol={symbol} filled={filled_qty:.6f} order_id={order_id} status={status}")
                                
                                # Update Redis position
                                new_qty = position_qty - filled_qty
                                if abs(new_qty) < 0.0001 or close_pct >= 100.0:
                                    # Full close: delete position
                                    self.redis.delete(pos_key)
                                    logger.info(f"[CLOSE] {symbol}: Position deleted (full close)")
                                else:
                                    # Partial close: update quantity
                                    self.redis.hset(pos_key, 'quantity', str(new_qty))
                                    logger.info(f"[CLOSE] {symbol}: Position updated qty={position_qty:.6f} -> {new_qty:.6f}")
                                
                                # Set dedupe marker
                                self.redis.setex(dedupe_key, 600, "1")  # 10 min TTL
                                
                                # Publish success result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'True',
                                    'reduceOnly': 'True',
                                    'close_qty': str(close_qty),
                                    'filled_qty': str(filled_qty),
                                    'order_id': str(order_id),
                                    'status': status,
                                    'side': close_side,
                                    'close_pct': str(close_pct),
                                    'timestamp': str(int(time.time()))
                                })
                                
                            except Exception as e:
                                logger.error(f"[CLOSE] {symbol}: Failed to execute close: {e}", exc_info=True)
                                # Set dedupe marker (prevent retry storm)
                                self.redis.setex(dedupe_key, 600, "1")
                                # Publish error result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'False',
                                    'error': f"execution_failed: {str(e)[:200]}",
                                    'timestamp': str(int(time.time()))
                                })
                            
                            # ACK the message
                            self.redis.xack(stream_key, consumer_group, msg_id)
                            logger.info(f"[CLOSE] {symbol}: Message ACK'd (msg_id={msg_id})")
                            continue
                        
                        # ========== ENTRY HANDLER (existing code) ==========
                        side = plan_data.get('side', '').upper()
                        leverage = float(plan_data.get('leverage', '1'))
                        stop_loss = plan_data.get('stop_loss')
                        take_profit = plan_data.get('take_profit')
                        qty = float(plan_data.get('qty', '0'))
                        
                        # Extract ATR/volatility data for risk computation
                        atr_value = float(plan_data.get('atr_value', 0.0))
                        volatility_factor = float(plan_data.get('volatility_factor', 0.0))
                        entry_price = float(plan_data.get('entry_price', 0.0))
                        
                        # Process both BUY and SELL entry signals
                        if side not in ['BUY', 'SELL']:
                            logger.debug(f"[ENTRY] {symbol}: Skipping {side} (not BUY or SELL)")
                            self.redis.xack(stream_key, consumer_group, msg_id)
                            continue
                        
                        position_side = 'LONG' if side == 'BUY' else 'SHORT'
                        logger.info(f"[ENTRY] {symbol}: Processing {side} intent (→{position_side}, leverage={leverage}, qty={qty}, plan_id={plan_id[:8]})")
                        
                        # 🔥 HARD GATE: Check symbol is in policy universe (fail-closed)
                        if not self._check_symbol_allowlist(symbol):
                            logger.warning(f"[ENTRY] {symbol}: Order REJECTED - symbol not in policy allowlist")
                            self.redis.xack(stream_key, consumer_group, msg_id)
                            continue
                        
                        # 🔥 STRICT ANTI-DUPLICATE GATE: Check if position already exists
                        pos_key = f"quantum:position:{symbol}"
                        existing_pos = self.redis.hgetall(pos_key)
                        if existing_pos:
                            existing_side = existing_pos.get(b'side', existing_pos.get('side', b'')).decode() if isinstance(existing_pos.get(b'side', existing_pos.get('side', b'')), bytes) else existing_pos.get(b'side', existing_pos.get('side', ''))
                            existing_qty = existing_pos.get(b'quantity', existing_pos.get('quantity', b'0')).decode() if isinstance(existing_pos.get(b'quantity', existing_pos.get('quantity', b'0')), bytes) else existing_pos.get(b'quantity', existing_pos.get('quantity', '0'))
                            
                            # Block if trying to open same side (pyramiding)
                            if existing_side == position_side:
                                logger.warning(f"[ENTRY] {symbol}: SKIP_OPEN_DUPLICATE - Position already exists (side={position_side}, qty={existing_qty})")
                                self.redis.xack(stream_key, consumer_group, msg_id)
                                continue
                        
                        # 🔥 COOLDOWN GATE: Prevent rapid re-opening (fail-closed)
                        cooldown_key = f"quantum:cooldown:open:{symbol}"
                        if self.redis.exists(cooldown_key):
                            ttl = self.redis.ttl(cooldown_key)
                            logger.warning(f"[ENTRY] {symbol}: SKIP_OPEN_COOLDOWN - Recently opened (cooldown={ttl}s remaining)")
                            self.redis.xack(stream_key, consumer_group, msg_id)
                            continue
                        
                        # Try to execute the entry order only if credentials are available
                        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
                        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
                        
                        if not api_key or not api_secret:
                            logger.warning(f"[ENTRY] {symbol}: Missing Binance testnet credentials")
                            self.redis.xack(stream_key, consumer_group, msg_id)
                            continue
                        
                        # Place market order (BUY or SELL)
                        try:
                            client = BinanceTestnetClient(api_key, api_secret)
                            
                            # Place market order (NOT reduce-only)
                            order_result = client.place_market_order(
                                symbol=symbol,
                                side=side,  # BUY or SELL
                                quantity=qty,
                                reduce_only=False
                            )
                            
                            logger.info(f"[ENTRY] {symbol}: {side} order placed: {order_result}")
                            
                            # Store position reference
                            pos_key = f"quantum:position:{symbol}"
                            
                            # Compute entry_risk using ATR (dynamic, no hardcoded %)
                            # risk_price = atr_value * volatility_factor
                            # entry_risk_usdt = abs(qty) * risk_price
                            risk_price = atr_value * volatility_factor if (atr_value > 0 and volatility_factor > 0) else 0.0
                            entry_risk_usdt = abs(qty) * risk_price if risk_price > 0 else 0.0
                            risk_missing = 1 if entry_risk_usdt == 0 else 0
                            
                            position_mapping = {
                                "symbol": symbol,
                                "side": position_side,  # LONG or SHORT
                                "quantity": str(qty),
                                "entry_price": str(entry_price),
                                "leverage": str(leverage),
                                "stop_loss": stop_loss or "0",
                                "take_profit": take_profit or "0",
                                "plan_id": plan_id,
                                "created_at": str(int(time.time())),
                                "atr_value": str(atr_value),
                                "volatility_factor": str(volatility_factor),
                                "entry_risk_usdt": str(entry_risk_usdt),
                                "risk_price": str(risk_price),
                                "risk_missing": str(risk_missing)
                            }
                            
                            self.redis.hset(pos_key, mapping=position_mapping)
                            logger.info(f"[ENTRY] {symbol}: Position reference stored (entry_risk_usdt={entry_risk_usdt:.4f}, atr={atr_value}, vol_factor={volatility_factor})")
                            
                            # 🔥 Set cooldown to prevent rapid re-opening (180s = 3 minutes)
                            cooldown_key = f"quantum:cooldown:open:{symbol}"
                            self.redis.setex(cooldown_key, 180, "1")
                            logger.info(f"[ENTRY] {symbol}: Cooldown set (180s)")
                        
                        except Exception as e:
                            logger.error(f"[ENTRY] {symbol}: Failed to place order: {e}", exc_info=True)
                        
                        # ACK the message
                        self.redis.xack(stream_key, consumer_group, msg_id)
                        logger.info(f"[ENTRY] {symbol}: Message ACK'd (msg_id={msg_id})")
                    
                    except Exception as e:
                        logger.error(f"[ENTRY] Error processing message {msg_id}: {e}", exc_info=True)
                        # ACK to prevent reprocessing
                        try:
                            self.redis.xack(stream_key, consumer_group, msg_id)
                        except:
                            pass
        
        except Exception as e:
            logger.error(f"[ENTRY] Error processing apply.plan stream: {e}", exc_info=True)
    
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
                    
                    # P2.8A.1: Shadow observation on apply.plan consumption (deterministic proof support)
                    # FAIL-OPEN: Never blocks, never changes execution
                    if self.p28_enabled and heat_observer and plan_id and symbol:
                        try:
                            decision = plan_data.get('decision', 'UNKNOWN')
                            heat_observer.observe(
                                redis_client=self.redis,
                                plan_id=plan_id,
                                symbol=symbol,
                                apply_decision=decision,
                                obs_point="reconcile_close_consume",
                                enabled=self.p28_enabled,
                                stream_name=self.p28_stream,
                                lookup_prefix=self.p28_lookup_prefix,
                                dedupe_ttl_sec=self.p28_dedupe_ttl,
                                max_debug_chars=self.p28_max_debug
                            )
                        except Exception as e:
                            # Catch-all to protect Apply (fail-open)
                            logger.warning(f"{symbol}: Heat observer error on apply.plan (ignored): {e}")
                    
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
        print("RUN_CYCLE_CALLED_NOW", flush=True)
        logger.debug("=== Apply Layer Cycle ===")
        
        # P2 Universe: Refresh allowlist cache if expired
        self._refresh_allowlist_if_needed()
        
        # HIGHEST PRIORITY: Process entry intents from intent_bridge
        try:
            logger.info("[ENTRY_CYCLE_START] Calling process_apply_plan_stream...")
            self.process_apply_plan_stream()
            logger.info("[ENTRY_CYCLE_END] process_apply_plan_stream completed")
        except Exception as e:
            logger.error(f"[ENTRY_CYCLE_ERROR] Error processing apply.plan stream: {e}", exc_info=True)
        
        # HIGH PRIORITY: Process RECONCILE_CLOSE plans (self-healing)
        try:
            logger.warning("[RECON_CYCLE_START] Calling process_reconcile_close_stream...")
            self.process_reconcile_close_stream()
            logger.warning("[RECON_CYCLE_END] process_reconcile_close_stream completed")
        except Exception as e:
            logger.warning(f"[RECON_CYCLE_ERROR] {type(e).__name__}: {e}")
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
                    # Extract reason from reason_codes if available
                    error_reason = plan.reason_codes[0] if plan.reason_codes else plan.decision
                    result = ApplyResult(
                        plan_id=plan.plan_id,
                        symbol=plan.symbol,
                        decision=plan.decision,
                        executed=False,
                        would_execute=False,
                        steps_results=[],
                        error=error_reason,  # Set error to skip reason
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
