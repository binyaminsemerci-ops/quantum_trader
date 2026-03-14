#!/usr/bin/env python3
"""
Intent Executor - Executes intent_bridge plans after P3.3 permit
==================================================================

Reads quantum:stream:apply.plan (source=intent_bridge)
Waits for P3.3 permit (quantum:permit:p33:<plan_id>)
Executes Binance testnet orders (reduceOnly=true)
Writes results to quantum:stream:apply.result

Author: Quantum Trader Team
Date: 2026-01-26
"""
import os
import sys
import json
import time
import socket
import hmac
import hashlib
import urllib.parse
import urllib.request
import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import redis

# Exit ownership
try:
    from lib.exit_ownership import EXIT_OWNER
    EXIT_OWNERSHIP_ENABLED = False  # 🔥 DISABLED: Allow all services to harvest profits
except ImportError:
    EXIT_OWNER = "exitbrain_v3_5"
    EXIT_OWNERSHIP_ENABLED = False
    
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [INTENT-EXEC] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Config from env
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
BINANCE_BASE_URL = "https://testnet.binancefuture.com"

# P2 Universe integration: ALLOWLIST now loaded dynamically in __init__
# Legacy INTENT_EXECUTOR_ALLOWLIST used as fallback

# Source allowlist (audit-safe: default allows intent_bridge + exit_management_agent)
SOURCE_ALLOWLIST_STR = os.getenv("INTENT_EXECUTOR_SOURCE_ALLOWLIST", "intent_bridge,apply_layer,exit_management_agent")
SOURCE_ALLOWLIST = set([s.strip() for s in SOURCE_ALLOWLIST_STR.split(",") if s.strip()])

# Manual lane configuration (separate stream, TTL-guarded)
MANUAL_STREAM = os.getenv("INTENT_EXECUTOR_MANUAL_STREAM", "quantum:stream:apply.plan.manual")
MANUAL_GROUP = os.getenv("INTENT_EXECUTOR_MANUAL_GROUP", "intent_executor_manual")
MANUAL_LANE_REDIS_KEY = "quantum:manual_lane:enabled"

# Harvest intent configuration (autonomous exits)
HARVEST_STREAM = os.getenv("INTENT_EXECUTOR_HARVEST_STREAM", "quantum:stream:harvest.intent")
HARVEST_GROUP = os.getenv("INTENT_EXECUTOR_HARVEST_GROUP", "intent_executor_harvest")
METRICS_REDIS_HASH = "quantum:metrics:intent_executor"
HEARTBEAT_INTERVAL_SEC = 60

# Optional: Update ledger after execution (timer handles sync, this is redundant but can be enabled)
UPDATE_LEDGER_AFTER_EXEC = os.getenv("INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC", "false").lower() == "true"

# Exchange-aware sizing
ALLOW_UPSIZE = os.getenv("INTENT_EXECUTOR_ALLOW_UPSIZE", "false").lower() == "true"
MIN_NOTIONAL_OVERRIDE = float(os.getenv("INTENT_EXECUTOR_MIN_NOTIONAL_USDT", "0"))

# Binance futures quantity precision (decimal places) - DEPRECATED, use exchangeInfo
SYMBOL_PRECISION = {
    "BTCUSDT": 3,
    "ETHUSDT": 3,
    "TRXUSDT": 0,  # Integer only
}
APPLY_PLAN_STREAM = "quantum:stream:apply.plan"
APPLY_RESULT_STREAM = "quantum:stream:apply.result"
CONSUMER_GROUP = "intent_executor"
CONSUMER_NAME = f"{socket.gethostname()}_{os.getpid()}"

PERMIT_TIMEOUT_SEC = 8.0
PERMIT_POLL_INTERVAL = 0.2
IDEMPOTENCY_TTL = 86400  # 24h


class IntentExecutor:
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        
        # P2 Universe integration: dynamic allowlist from Universe Service
        self.universe_enable = os.getenv("UNIVERSE_ENABLE", "true").lower() in ("true", "1", "yes")
        self.universe_cache_seconds = int(os.getenv("UNIVERSE_CACHE_SECONDS", "60"))
        self.universe_last_refresh = 0
        self.allowlist = set(self._load_allowlist())
        
        # Exchange info cache (filters)
        self.exchange_filters = {}  # symbol -> {minNotional, minQty, stepSize}
        
        # Heartbeat tracking
        self.last_heartbeat_ts = 0
        
        # Log allowlist source (universe or fallback)
        self._log_allowlist_source()
        
        logger.info("=" * 80)
        logger.info("Intent Executor - apply.plan → P3.3 permit → Binance")
        logger.info("=" * 80)
        logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
        logger.info(f"Consumer: {CONSUMER_GROUP} / {CONSUMER_NAME}")
        logger.info(f"Symbol allowlist: {sorted(self.allowlist)}")
        logger.info(f"Source allowlist: {sorted(SOURCE_ALLOWLIST)}")
        logger.info(f"Binance: {BINANCE_BASE_URL}")
        logger.info(f"Apply plan stream: {APPLY_PLAN_STREAM}")
        logger.info(f"Apply result stream: {APPLY_RESULT_STREAM}")
        logger.info(f"Manual stream: {MANUAL_STREAM}")
        logger.info(f"Manual group: {MANUAL_GROUP}")
        logger.info(f"Manual lane TTL guard: {MANUAL_LANE_REDIS_KEY}")
        logger.info(f"Harvest stream: {HARVEST_STREAM}")
        logger.info(f"Harvest group: {HARVEST_GROUP}")
        logger.info(f"Allow upsize: {ALLOW_UPSIZE}")
        logger.info(f"Min notional override: {MIN_NOTIONAL_OVERRIDE}")
        
        # Log manual lane status at startup
        manual_ttl = self._get_manual_lane_ttl()
        if manual_ttl > 0:
            logger.info(f"🔓 MANUAL_LANE_ACTIVE ttl_remaining={manual_ttl}s")
        else:
            logger.info(f"🔒 MANUAL_LANE_OFF")
        
        logger.info("=" * 80)
        
        # Verify Binance keys
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise ValueError("BINANCE_TESTNET_API_KEY/SECRET not set!")
        
        # Fetch exchange info
        self._fetch_exchange_info()
        
        # Create consumer group
        self._ensure_consumer_group()
        
        # Create manual consumer group
        self._ensure_manual_consumer_group()
        
        # Create harvest consumer group
        self._ensure_harvest_consumer_group()
    
    def _load_universe_symbols(self) -> list:
        """Load symbols from Universe Service Redis key (P2 Universe integration)"""
        try:
            # Use decode_responses temporarily for this call
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
            meta = r.hgetall("quantum:cfg:universe:meta")
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
            universe_json = r.get("quantum:cfg:universe:active")
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
        """Load allowlist from Universe or fallback to INTENT_EXECUTOR_ALLOWLIST env (P2 Universe integration)"""
        if self.universe_enable:
            symbols = self._load_universe_symbols()
            if symbols:
                return symbols
            else:
                logger.warning("Universe unavailable, falling back to INTENT_EXECUTOR_ALLOWLIST")
        
        # Fallback to env var
        allowlist_str = os.getenv("INTENT_EXECUTOR_ALLOWLIST", "BTCUSDT,ETHUSDT,TRXUSDT")
        return [s.strip().upper() for s in allowlist_str.split(",") if s.strip()]
    
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
            logger.info(f"Intent Executor allowlist source=env count={len(self.allowlist)}")
            return
        
        try:
            # Use decode_responses temporarily for this call
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
            meta = r.hgetall("quantum:cfg:universe:meta")
            if meta:
                stale = int(meta.get("stale", "1"))
                count = int(meta.get("count", "0"))
                asof_epoch = int(meta.get("asof_epoch", "0"))
                
                if stale == 0 and count > 0:
                    logger.info(f"Intent Executor allowlist source=universe stale=0 count={count} asof_epoch={asof_epoch}")
                else:
                    logger.warning(f"Intent Executor allowlist source=fallback reason=stale={stale} count={count}")
            else:
                logger.warning(f"Intent Executor allowlist source=fallback reason=meta_missing count={len(self.allowlist)}")
        except Exception as e:
            logger.warning(f"Error logging allowlist source: {e}")
    
    def _get_manual_lane_ttl(self) -> int:
        """Get remaining TTL for manual lane enablement key"""
        try:
            ttl = self.redis.ttl(MANUAL_LANE_REDIS_KEY)
            return max(0, ttl) if ttl != -2 else 0  # -2 = key doesn't exist
        except Exception as e:
            logger.warning(f"Failed to get manual lane TTL: {e}")
            return 0
    
    def _is_manual_lane_enabled(self) -> bool:
        """Check if manual lane is enabled (Redis key exists with TTL)"""
        try:
            value = self.redis.get(MANUAL_LANE_REDIS_KEY)
            if value and value.decode() == "1":
                ttl = self._get_manual_lane_ttl()
                return ttl > 0
            return False
        except Exception as e:
            logger.warning(f"Failed to check manual lane status: {e}")
            return False
    
    def _emit_heartbeat(self):
        """Emit periodic heartbeat log with manual lane status and metrics"""
        now = time.time()
        if now - self.last_heartbeat_ts < HEARTBEAT_INTERVAL_SEC:
            return
        
        self.last_heartbeat_ts = now
        
        # Get manual lane status
        ttl = self._get_manual_lane_ttl()
        if ttl > 0:
            logger.info(f"🔓 MANUAL_LANE_ACTIVE ttl_remaining={ttl}s")
        else:
            logger.info(f"🔒 MANUAL_LANE_OFF")
        
        # Get metrics from Redis (best-effort)
        try:
            metrics = self.redis.hgetall(METRICS_REDIS_HASH)
            if metrics:
                metrics_str = " ".join([f"{k.decode()}={v.decode()}" for k, v in metrics.items()])
                logger.info(f"📊 Metrics: {metrics_str}")
        except Exception as e:
            logger.debug(f"Failed to fetch metrics for heartbeat: {e}")
        
    def _fetch_exchange_info(self):
        """Fetch exchangeInfo for filters (minNotional, LOT_SIZE)"""
        try:
            url = f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            for symbol_info in data.get("symbols", []):
                symbol = symbol_info["symbol"]
                if symbol not in self.allowlist:
                    continue
                
                filters = {}
                for filt in symbol_info.get("filters", []):
                    if filt["filterType"] == "MIN_NOTIONAL":
                        filters["minNotional"] = float(filt.get("notional", 0))
                    elif filt["filterType"] == "LOT_SIZE":
                        filters["minQty"] = float(filt.get("minQty", 0))
                        filters["stepSize"] = float(filt.get("stepSize", 0))
                
                if filters:
                    self.exchange_filters[symbol] = filters
                    logger.info(f"📊 {symbol}: minNotional={filters.get('minNotional', 0)}, "
                              f"minQty={filters.get('minQty', 0)}, stepSize={filters.get('stepSize', 0)}")
            
            if not self.exchange_filters:
                logger.warning("⚠️  No exchange filters loaded! All symbols may fail minNotional checks.")
        except Exception as e:
            logger.error(f"Failed to fetch exchangeInfo: {e}")
            logger.warning("⚠️  Continuing without exchange filters (orders may fail)")
    
    def _ensure_consumer_group(self):
        """Create consumer group if not exists (starts at '$' = latest)"""
        try:
            self.redis.xgroup_create(
                APPLY_PLAN_STREAM,
                CONSUMER_GROUP,
                id="$",  # Start at latest, not '0' (skip old messages)
                mkstream=True
            )
            logger.info(f"✅ Consumer group created: {CONSUMER_GROUP} (starting at latest)")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Consumer group exists: {CONSUMER_GROUP}")
            else:
                raise
    
    def _ensure_manual_consumer_group(self):
        """Create manual consumer group"""
        try:
            self.redis.xgroup_create(
                MANUAL_STREAM,
                MANUAL_GROUP,
                id="$",
                mkstream=True
            )
            logger.info(f"✅ Manual consumer group created: {MANUAL_GROUP} on {MANUAL_STREAM}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Manual consumer group exists: {MANUAL_GROUP}")
            else:
                raise
    
    def _ensure_harvest_consumer_group(self):
        """Create harvest consumer group for autonomous exits"""
        try:
            self.redis.xgroup_create(
                HARVEST_STREAM,
                HARVEST_GROUP,
                id="$",
                mkstream=True
            )
            logger.info(f"✅ Harvest consumer group created: {HARVEST_GROUP} on {HARVEST_STREAM}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Harvest consumer group exists: {HARVEST_GROUP}")
            else:
                raise
    
    def _is_done(self, plan_id: str) -> bool:
        """Check if plan already executed (idempotency)"""
        key = f"quantum:intent_executor:done:{plan_id}"
        return bool(self.redis.get(key))
    
    def _mark_done(self, plan_id: str):
        """Mark plan as executed"""
        key = f"quantum:intent_executor:done:{plan_id}"
        self.redis.setex(key, IDEMPOTENCY_TTL, "1")
    
    def _inc_redis_counter(self, field: str):
        """Increment Redis hash counter for metrics"""
        try:
            self.redis.hincrby(METRICS_REDIS_HASH, field, 1)
        except Exception as e:
            logger.debug(f"Failed to increment Redis counter {field}: {e}")
    
    def _read_permit_safe(self, key: str) -> Optional[Dict]:
        """Read P3.3 permit key — handles both STRING (json) and HASH types.
        P3.3 position_state_brain uses: setex(key, ttl, json.dumps(data)) -> STRING.
        auto_permit_p33.py / legacy code used: hset() -> HASH.
        """
        # Try HASH first
        try:
            data = self.redis.hgetall(key)
            if data:
                return {
                    k.decode() if isinstance(k, bytes) else k:
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in data.items()
                }
        except Exception:
            # WRONGTYPE: key is STRING (P3.3 uses setex with json) — fall through
            pass
        # Fallback: STRING / json
        try:
            raw = self.redis.get(key)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"Error reading permit {key}: {e}")
        return None

    def _wait_for_permit(self, plan_id: str) -> Optional[Dict]:
        """Wait for P3.3 permit with timeout. Handles STRING (json) and HASH permit types."""
        key = f"quantum:permit:p33:{plan_id}"
        start_time = time.time()

        while time.time() - start_time < PERMIT_TIMEOUT_SEC:
            permit = self._read_permit_safe(key)
            if permit:
                return permit
            time.sleep(PERMIT_POLL_INTERVAL)

        return None
    
    def _validate_and_adjust_qty(self, symbol: str, qty: float, mark_price: float) -> tuple:
        """
        Validate and adjust qty to meet exchange filters.
        Returns: (adjusted_qty, reason) or (0, reason) if order should be blocked.
        """
        filters = self.exchange_filters.get(symbol, {})
        min_notional = MIN_NOTIONAL_OVERRIDE or filters.get("minNotional", 0)
        min_qty = filters.get("minQty", 0)
        step_size = filters.get("stepSize", 0.001)
        
        # Round to stepSize
        qty = round(qty / step_size) * step_size
        
        # Check minQty
        if qty < min_qty:
            return 0, f"qty {qty} < minQty {min_qty}"
        
        # Check minNotional
        notional = qty * mark_price
        
        if notional < min_notional:
            if ALLOW_UPSIZE:
                # Upsize to meet minNotional - ceil() to ensure we meet threshold
                required_qty = min_notional / mark_price
                qty_upsized = math.ceil(required_qty / step_size) * step_size
                logger.info(f"📈 Upsizing: {qty:.4f} → {qty_upsized:.4f} to meet minNotional {min_notional}")
                return qty_upsized, "upsized_to_meet_min_notional"
            else:
                # FAIL-CLOSED: block order
                return 0, f"notional {notional:.2f} < minNotional {min_notional:.2f} (ALLOW_UPSIZE=false)"
        
        # 🔥 TESTNET MAX NOTIONAL: Enforce $500 position cap (defense layer 3)
        if os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true":
            MAX_POSITION_NOTIONAL = 500.0
            if notional > MAX_POSITION_NOTIONAL:
                logger.error(
                    f"[TESTNET_CAP] Position size rejected: "
                    f"${notional:.2f} exceeds ${MAX_POSITION_NOTIONAL} cap "
                    f"(symbol={symbol}, qty={qty:.4f}, price={mark_price:.2f})"
                )
                return 0, f"position_too_large:${notional:.0f}>$500_cap"
        
        return qty, "ok"
    
    def _get_mark_price(self, symbol: str) -> float:
        """Get current mark price from Binance"""
        try:
            url = f"{BINANCE_BASE_URL}/fapi/v1/premiumIndex?symbol={symbol}"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                return float(data.get("markPrice", 0))
        except Exception as e:
            logger.warning(f"Failed to fetch mark price for {symbol}: {e}")
            return 0
    
    def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """POST /fapi/v1/leverage with params in request BODY (form-encoded).
        Non-fatal. Binance -4028 (Leverage not changed) = already correct.
        """
        try:
            params = {
                "symbol": symbol,
                "leverage": leverage,
                "timestamp": int(time.time() * 1000)
            }
            body_str = urllib.parse.urlencode(params)
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                body_str.encode(),
                hashlib.sha256
            ).hexdigest()
            body_str += f"&signature={signature}"
            body_bytes = body_str.encode()
            url = f"{BINANCE_BASE_URL}/fapi/v1/leverage"
            req = urllib.request.Request(url, data=body_bytes, method="POST")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                actual = result.get("leverage", leverage)
                logger.info(f"[LEVERAGE] {symbol}: set to {actual}x OK")
                return True
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err  = json.loads(body)
                code = err.get("code", 0)
                msg  = err.get("msg", body)
            except Exception:
                code, msg = 0, body
            # -4028 = "Leverage not changed" — already at correct value, treat as OK
            if code == -4028:
                logger.info(f"[LEVERAGE] {symbol}: already at {leverage}x (no change needed)")
                return True
            logger.warning(f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal) code={code} msg={msg}")
            return False
        except Exception as e:
            logger.warning(f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal): {e}")
            return False

    def _execute_binance_order(self, symbol: str, side: str, qty: float, reduce_only: bool = True) -> Dict:
        """Execute Binance futures market order with exchange-aware sizing"""
        # Get filters
        filters = self.exchange_filters.get(symbol, {})
        step_size = filters.get("stepSize", 0.001)
        
        # Round qty to stepSize (not legacy precision dict)
        qty = round(qty / step_size) * step_size
        
        # Determine precision for string formatting from stepSize
        if step_size >= 1:
            precision = 0
        elif step_size >= 0.1:
            precision = 1
        elif step_size >= 0.01:
            precision = 2
        elif step_size >= 0.001:
            precision = 3
        else:
            precision = 4
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{qty:.{precision}f}",
            "reduceOnly": "true" if reduce_only else "false",
            "timestamp": int(time.time() * 1000)
        }
        
        # Sign request
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            BINANCE_API_SECRET.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        
        # Build URL
        url = f"{BINANCE_BASE_URL}/fapi/v1/order?{urllib.parse.urlencode(params)}"
        
        # Send request
        req = urllib.request.Request(url, method="POST")
        req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return {
                    "success": True,
                    "order_id": result.get("orderId"),
                    "client_order_id": result.get("clientOrderId"),
                    "status": result.get("status"),
                    "filled_qty": float(result.get("executedQty", 0))
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            logger.error(f"Binance HTTP error: {e.code} {error_body}")
            return {
                "success": False,
                "error": f"HTTPError {e.code}: {error_body}"
            }
        except Exception as e:
            logger.error(f"Binance request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _poll_order_fill(self, symbol: str, order_id: int, max_attempts: int = 3) -> Dict:
        """Poll order status to get fill confirmation (3 attempts: 0.5s, 1s, 2s)"""
        delays = [0.5, 1.0, 2.0]
        
        for attempt in range(max_attempts):
            if attempt > 0:
                time.sleep(delays[attempt - 1])
            
            try:
                params = {
                    "symbol": symbol,
                    "orderId": order_id,
                    "timestamp": int(time.time() * 1000)
                }
                
                query_string = urllib.parse.urlencode(params)
                signature = hmac.new(
                    BINANCE_API_SECRET.encode(),
                    query_string.encode(),
                    hashlib.sha256
                ).hexdigest()
                params["signature"] = signature
                
                url = f"{BINANCE_BASE_URL}/fapi/v1/order?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, method="GET")
                req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    order = json.loads(response.read().decode())
                    status = order.get("status")
                    filled_qty = float(order.get("executedQty", 0))
                    
                    if status in ["FILLED", "PARTIALLY_FILLED"] and filled_qty > 0:
                        return {"status": status, "filled_qty": filled_qty, "attempts": attempt + 1}
                    elif status in ["CANCELED", "REJECTED", "EXPIRED"]:
                        return {"status": status, "filled_qty": filled_qty, "attempts": attempt + 1}
            except Exception as e:
                logger.debug(f"Fill poll attempt {attempt + 1} failed: {e}")
        
        return {"status": "UNKNOWN", "filled_qty": 0.0, "attempts": max_attempts}
    
    def _update_ledger(self, symbol: str):
        """Update ledger after order execution by fetching fresh positionRisk"""
        try:
            params = {"timestamp": int(time.time() * 1000)}
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            
            url = f"{BINANCE_BASE_URL}/fapi/v2/positionRisk?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                positions = json.loads(response.read().decode())
                pos = next((p for p in positions if p["symbol"] == symbol), None)
                
                if pos:
                    amt = float(pos.get("positionAmt", 0))
                    side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else "FLAT")
                    abs_amt = abs(amt)
                    
                    ledger_key = f"quantum:position:ledger:{symbol}"
                    self.redis.hset(ledger_key, mapping={
                        # reconcile_engine fields
                        "ledger_amt":     str(amt),   # SIGNED (neg=SHORT)
                        "ledger_side":    side,
                        # P3.3 fields (last_known_amt must be signed)
                        "last_known_amt": str(amt),   # SIGNED (neg=SHORT)
                        "last_side":      side,
                        "position_amt":   str(amt),
                        "updated_at":     str(int(time.time())),
                    })
                    logger.info(f"📊 Ledger updated: {symbol} {side} {amt:.4f}")
                else:
                    logger.warning(f"Position not found for {symbol} in positionRisk")
        except Exception as e:
            logger.error(f"Failed to update ledger for {symbol}: {e}")
    
    def _commit_ledger_exactly_once(self, symbol: str, order_id: int, filled_qty: float, side: str):
        """
        Exactly-once ledger commit on FILLED orders (idempotent on order_id)
        
        This ensures quantum:position:ledger:<symbol> reflects exchange truth
        with atomic incremental updates (Bug #11 fix).
        
        Dedup: Uses quantum:ledger:seen_orders set to prevent double-counting
        Source: Atomic calculation from current ledger + order delta (BUY=+qty, SELL=-qty)
        
        Args:
            symbol: Trading pair
            order_id: Binance order ID (unique)
            filled_qty: Quantity filled on this order
            side: Order side (BUY or SELL)
        """
        logger.info(f"🔍 LEDGER_COMMIT_START symbol={symbol} order_id={order_id} filled_qty={filled_qty}")
        try:
            # Dedup check: Skip if order already processed
            seen_orders_key = "quantum:ledger:seen_orders"
            is_duplicate = self.redis.sismember(seen_orders_key, str(order_id))
            
            if is_duplicate:
                logger.debug(f"LEDGER_COMMIT_SKIP symbol={symbol} order_id={order_id} (duplicate)")
                return
            
            # Mark order as seen (atomic, idempotent)
            self.redis.sadd(seen_orders_key, str(order_id))
            
            # Fetch current ledger state for atomic update (Bug #11 fix)
            ledger_key = f"quantum:position:ledger:{symbol}"
            current_ledger = self.redis.hgetall(ledger_key)
            
            # Calculate position delta from order
            # BUY increases position (LONG), SELL decreases (SHORT)
            order_delta = filled_qty if side == "BUY" else -filled_qty
            
            # Calculate new position atomically
            current_position = float(current_ledger.get(b"position_amt", b"0").decode()) if current_ledger else 0.0
            new_position = current_position + order_delta
            
            logger.info(
                f"🔧 ATOMIC UPDATE: {symbol} {side} {filled_qty:.4f} | "
                f"{current_position:.4f} → {new_position:.4f} (delta={order_delta:+.4f})"
            )
            
            # Get snapshot for price/pnl data (best-effort, may be stale)
            snapshot_key = f"quantum:position:snapshot:{symbol}"
            snapshot = self.redis.hgetall(snapshot_key)
            
            # Use snapshot price data if available, otherwise keep ledger values
            if snapshot:
                entry_price = float(snapshot.get(b"entry_price", b"0").decode())
                unrealized_pnl = float(snapshot.get(b"unrealized_pnl", b"0").decode())
                leverage = int(float(snapshot.get(b"leverage", b"1").decode()))
            elif current_ledger:
                # Keep existing ledger price data
                entry_price = float(current_ledger.get(b"entry_price", b"0").decode())
                unrealized_pnl = float(current_ledger.get(b"unrealized_pnl", b"0").decode())
                leverage = int(float(current_ledger.get(b"leverage", b"5").decode()))
            else:
                # First order - use zero values (P3.3 will update from exchange)
                entry_price = 0.0
                unrealized_pnl = 0.0
                leverage = 5
            
            # Use atomically calculated position
            position_amt = new_position
            
            # Derive ledger side from SIGNED position_amt (P3.3 contract)
            eps = 1e-12
            if abs(position_amt) <= eps:
                ledger_side = "FLAT"

                # Capture pre-FLAT data for trade.closed before zeroing
                pre_flat_entry_price = entry_price
                pre_flat_side = "LONG" if current_position > 0 else "SHORT"

                # Zero out all fields when FLAT (clean state)
                entry_price = 0.0
                unrealized_pnl = 0.0
                filled_qty = 0.0
                # Clean up stale snapshot/position keys to unblock intent-bridge
                try:
                    snap_key = f"quantum:position:snapshot:{symbol}"
                    pos_key = f"quantum:position:{symbol}"
                    d_snap = self.redis.delete(snap_key)
                    d_pos = self.redis.delete(pos_key)
                    if d_snap or d_pos:
                        logger.info(f"🧹 FLAT CLEANUP: {symbol} deleted snapshot={d_snap} position={d_pos}")
                except Exception as ce:
                    logger.error(f"❌ Flat cleanup failed for {symbol}: {ce}")

                # 📤 Publish trade.closed event for SimpleCLM/calibration pipeline
                # Main lane closes must also feed CLM (same as harvest lane)
                try:
                    exit_price = self._get_mark_price(symbol)
                    pnl_percent = 0.0
                    if pre_flat_entry_price > 0 and exit_price > 0:
                        if pre_flat_side == "LONG":
                            pnl_percent = ((exit_price - pre_flat_entry_price) / pre_flat_entry_price) * 100
                        else:
                            pnl_percent = ((pre_flat_entry_price - exit_price) / pre_flat_entry_price) * 100

                    close_event = {
                        "event_type": "trade.closed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "side": pre_flat_side,
                        "entry_price": str(pre_flat_entry_price),
                        "exit_price": str(exit_price),
                        "pnl_percent": str(round(pnl_percent, 2)),
                        "confidence": "0.5",
                        "model_id": "main_lane_close",
                        "R_net": "0.0",
                        "pnl_usd": "0.0",
                        "reason": "main_lane_flat",
                        "order_id": str(order_id),
                        "source": "intent_executor_main"
                    }
                    self.redis.xadd(
                        "quantum:stream:trade.closed",
                        close_event,
                        maxlen=1000
                    )
                    logger.info(
                        f"📤 MAIN LANE trade.closed: {symbol} {pre_flat_side} "
                        f"entry={pre_flat_entry_price:.2f} exit={exit_price:.2f} PnL={pnl_percent:.1f}%"
                    )
                except Exception as tc_e:
                    logger.error(f"❌ Failed to publish main lane trade.closed: {tc_e}")
            elif position_amt > 0:
                ledger_side = "LONG"
            else:
                ledger_side = "SHORT"
            
            # Build ledger payload (source: exchange snapshot)
            # CRITICAL: position_amt MUST be signed (negative=SHORT, positive=LONG)
            # Both reconcile_engine (ledger_amt) and P3.3 (last_known_amt) must agree.
            ledger_key = f"quantum:position:ledger:{symbol}"
            ledger_payload = {
                # reconcile_engine fields
                "ledger_amt":         str(position_amt),   # SIGNED
                "ledger_side":        ledger_side,
                # P3.3 fields (last_known_amt signed: neg=SHORT)
                "position_amt":       str(position_amt),
                "last_known_amt":     str(position_amt),
                "qty":                str(abs(position_amt)),
                "side":               ledger_side,
                "last_side":          ledger_side,
                "avg_entry_price":    str(entry_price),
                "entry_price":        str(entry_price),
                "unrealized_pnl":     str(unrealized_pnl),
                "leverage":           str(leverage),
                "last_order_id":      str(order_id),
                "last_filled_qty":    str(filled_qty),
                "last_executed_qty":  str(filled_qty),
                "last_update_ts":     str(int(time.time())),
                "updated_at":         str(int(time.time())),
                "synced_at":          str(int(time.time())),
                "source":             "intent_executor_exactly_once"
            }
            
            # Atomic ledger write
            self.redis.hset(ledger_key, mapping=ledger_payload)
            
            logger.info(
                f"LEDGER_COMMIT symbol={symbol} order_id={order_id} "
                f"amt={position_amt:.4f} side={ledger_side} "
                f"entry_price={entry_price:.2f} unrealized_pnl={unrealized_pnl:.2f} "
                f"filled_qty={filled_qty:.4f}"
            )
            
        except Exception as e:
            logger.error(
                f"LEDGER_COMMIT_FAILED symbol={symbol} order_id={order_id}: {e}",
                exc_info=True
            )
            # Non-blocking: Log failure but don't crash execution flow
    
    def _write_result(self, plan_id: str, symbol: str, executed: bool, **kwargs):
        """Write execution result to apply.result stream"""
        result = {
            "plan_id": plan_id,
            "symbol": symbol,
            "executed": executed,
            "source": "intent_executor",
            "timestamp": int(time.time())
        }
        result.update(kwargs)
        
        # Write to Redis stream
        self.redis.xadd(
            APPLY_RESULT_STREAM,
            {
                b"event_type": b"apply.result",
                b"plan_id": plan_id.encode(),
                b"symbol": symbol.encode(),
                b"executed": str(executed).lower().encode(),
                b"source": b"intent_executor",
                b"details": json.dumps(result).encode(),
                b"timestamp": str(result["timestamp"]).encode()
            }
        )
        
        logger.info(f"📝 Result written: plan={plan_id[:8]} executed={executed}")
    
    def process_plan(self, stream_id: bytes, event_data: Dict, lane: str = "main"):
        """Process single apply.plan message - with guaranteed ACK on all paths
        
        Args:
            stream_id: Redis stream entry ID
            event_data: Plan data from stream
            lane: 'main' or 'manual' for metrics/logging
        """
        stream_id_str = stream_id.decode()
        plan_id = ""
        symbol = ""
        
        # Parse event
        try:
            # Skip old nested payload format (has 'payload' field instead of flat fields)
            if b"payload" in event_data and b"plan_id" not in event_data:
                logger.debug(f"Skip old nested payload format: {stream_id_str}")
                return True  # ACK but skip
            
            # Get plan_id and source early for logging
            plan_id = event_data.get(b"plan_id", b"").decode()
            source = event_data.get(b"source", b"").decode()
            symbol = event_data.get(b"symbol", b"").decode().upper()
            
            # P3.5 GUARD: Check decision field - NEVER execute BLOCKED/SKIP plans
            plan_decision = event_data.get(b"decision", event_data.get(b"plan_decision", b"")).decode().upper()
            if plan_decision in ("BLOCKED", "SKIP"):
                # Extract reason from plan (prefer error, then reason, then reason_codes, then default)
                reason = (
                    event_data.get(b"error", b"").decode() or
                    event_data.get(b"reason", b"").decode() or
                    event_data.get(b"reason_codes", b"").decode() or
                    plan_decision.lower()
                )
                
                logger.info(f"P3.5_GUARD decision={plan_decision} plan_id={plan_id[:8]} symbol={symbol} reason={reason}")
                
                # Write result with decision preserved, executed=False, would_execute=False
                self._write_result(
                    plan_id, symbol, executed=False,
                    decision=plan_decision,
                    would_execute=False,
                    error=reason
                )
                self._mark_done(plan_id)
                self._inc_redis_counter("p35_guard_blocked")
                return True  # ACK and skip execution
            
            # MAIN LANE: Check source allowlist (but allow empty source for P3.3 bypass)
            # P3.3 permits = plans with empty source that come via Apply Layer
            if lane == "main":
                # Only reject if source is NOT in allowlist AND source is NOT empty
                # Empty source = P3.3 bypass, will be validated by permit check later
                if source and source not in SOURCE_ALLOWLIST:
                    logger.info(f"🚫 [lane={lane}] Skip plan (source_not_allowed): plan_id={plan_id[:8]} source={source} allowlist={sorted(SOURCE_ALLOWLIST)}")
                    self._inc_redis_counter("blocked_source")
                    # IMPORTANT: Extract fields for result before returning
                    side = event_data.get(b"side", b"").decode().upper()
                    qty_str = event_data.get(b"qty", b"0").decode()
                    try:
                        qty = float(qty_str)
                    except:
                        qty = 0
                    self._write_result(
                        plan_id, symbol, executed=False,
                        error=f"source_not_allowed:{source}",
                        side=side, qty=qty
                    )
                    self._mark_done(plan_id)
                    return True  # ACK and mark done
            
            # MANUAL LANE: Check TTL-guarded enablement
            elif lane == "manual":
                if not self._is_manual_lane_enabled():
                    ttl = self._get_manual_lane_ttl()
                    logger.info(f"🚫 [lane={lane}] Skip plan (manual_lane_disabled): plan_id={plan_id[:8]} ttl={ttl}")
                    self._inc_redis_counter("manual_blocked_disabled")
                    # IMPORTANT: Extract fields for result before returning
                    side = event_data.get(b"side", b"").decode().upper()
                    qty_str = event_data.get(b"qty", b"0").decode()
                    try:
                        qty = float(qty_str)
                    except:
                        qty = 0
                    self._write_result(
                        plan_id, symbol, executed=False,
                        error=f"manual_lane_disabled",
                        side=side, qty=qty
                    )
                    self._mark_done(plan_id)
                    return True  # ACK and mark done
                
                # Manual lane enabled - log consumption
                logger.info(f"🔓 [lane={lane}] Consuming manual plan: {plan_id[:8]}")
                self._inc_redis_counter("manual_consumed")
            
            # Extract plan details (FLAT format)
            side = event_data.get(b"side", b"").decode().upper()
            qty_str = event_data.get(b"qty", b"0").decode()
            
            # CRITICAL FIX (Feb 8, 2026): Default to TRUE (safer) if field missing
            # Previous default "false" caused churning disaster (~1600 USDT loss)
            # Missing field = assume CLOSE (reduce_only=true) not OPEN
            reduce_only_str = event_data.get(b"reduceOnly", b"true").decode().lower()
            reduce_only = reduce_only_str in ("true", "1", "yes")


            
            # Extract leverage from plan (LeverageEngine via intent-bridge)
            _lev_str = event_data.get(b"leverage", b"1").decode()
            try:
                plan_leverage = max(1, min(125, int(float(_lev_str))))
            except (ValueError, TypeError):
                plan_leverage = 1

            # Log warning if field was missing (indicates old/malformed plan)
            if b"reduceOnly" not in event_data:
                logger.warning(f"⚠️  P3.3 plan {plan_id[:8]} missing reduceOnly field, defaulting to TRUE (safe mode)")
            
            # Validate required fields (FLAT format must have all these)
            if not plan_id or not symbol or not side or not qty_str:
                logger.debug(f"Skip plan with missing fields: {stream_id_str}")
                # Try to write result with minimal info
                try:
                    side_val = event_data.get(b"side", b"").decode().upper() or "UNKNOWN"
                    qty_val = float(qty_str or "0")
                    self._write_result(
                        plan_id, symbol, executed=False,
                        error="missing_required_fields",
                        side=side_val, qty=qty_val
                    )
                    self._mark_done(plan_id)
                except:
                    pass  # Couldn't write result, but still mark done
                return True  # ACK and mark done
            
            qty = float(qty_str)
            
            # Allowlist check
            if symbol not in self.allowlist:
                logger.debug(f"Symbol not in allowlist: {symbol}")
                self._write_result(
                    plan_id, symbol, executed=False,
                    error=f"symbol_not_in_allowlist:{symbol}",
                    side=side, qty=qty
                )
                self._mark_done(plan_id)
                return True  # ACK and mark done
            
            # Idempotency check
            if self._is_done(plan_id):
                logger.debug(f"Plan already executed: {plan_id[:8]}")
                return True  # Already done
            
            logger.info(f"▶️  Processing plan: {plan_id[:8]} | {symbol} {side} qty={qty:.4f}")
            
            # Get P3.3 permit
            # For empty source (P3.3 bypass), wait for permit (it will be created after plan arrives)
            # For other sources, try cached permit first, then wait if needed
            permit_key = f"quantum:permit:p33:{plan_id}"
            # _read_permit_safe handles both STRING (P3.3 setex json) and HASH types
            permit = self._read_permit_safe(permit_key)
            
            if permit:
                if source == '':
                    logger.info(f"✅ Permit found (P3.3 bypass): {plan_id[:8]}")
                else:
                    logger.info(f"✅ Permit found: {plan_id[:8]}")
            else:
                # Permit not in cache yet - wait for it
                # This handles race condition where P3.3 permit not created yet
                if source == '':
                    logger.info(f"⏳ P3.3 permit not yet created, waiting: {plan_id[:8]}")
                else:
                    logger.info(f"⏳ Waiting for P3.3 permit: {plan_id[:8]}")
                permit = self._wait_for_permit(plan_id)
            
            if not permit:
                logger.warning(f"❌ No P3.3 permit: {plan_id[:8]}")
                self._write_result(
                    plan_id, symbol, executed=False,
                    error="p33_permit_missing_or_denied",
                    side=side, qty=qty
                )
                self._mark_done(plan_id)
                return True
            
            # Check permit status
            allow = permit.get("allow", False) or permit.get("granted", False)
            if not allow:
                reason = permit.get("reason", "unknown")
                logger.warning(f"❌ P3.3 denied: {plan_id[:8]} reason={reason}")
                self._write_result(
                    plan_id, symbol, executed=False,
                    error=f"p33_permit_denied:{reason}",
                    side=side, qty=qty,
                    permit=permit
                )
                self._mark_done(plan_id)
                return True
            
            # Get safe_qty from permit
            safe_qty = permit.get("safe_close_qty") or permit.get("safe_qty")
            if safe_qty is not None:
                safe_qty = abs(float(safe_qty))
                # For OPEN permits, safe_qty=0 means "use plan qty"
                # For CLOSE permits, safe_qty>0 is the clamped position size
                if safe_qty > 0:
                    qty_to_use = min(abs(qty), safe_qty)
                    logger.info(f"✅ P3.3 permit granted: safe_qty={safe_qty:.4f} → using {qty_to_use:.4f}")
                else:
                    # OPEN permit: use full plan qty
                    qty_to_use = abs(qty)
                    logger.info(f"✅ P3.3 permit granted (OPEN): safe_qty=0 → using plan qty={qty_to_use:.4f}")
            else:
                qty_to_use = abs(qty)
                logger.info(f"✅ P3.3 permit granted: using original qty={qty_to_use:.4f}")
            
            # Exchange-aware sizing validation (for OPEN orders only)
            if not reduce_only:
                mark_price = self._get_mark_price(symbol)
                if mark_price == 0:
                    logger.error(f"❌ Failed to get mark price for {symbol}")
                    self._write_result(
                        plan_id, symbol, executed=False,
                        error="mark_price_unavailable",
                        side=side, qty=qty_to_use
                    )
                    self._mark_done(plan_id)
                    return True
                
                qty_validated, reason = self._validate_and_adjust_qty(symbol, qty_to_use, mark_price)
                
                if qty_validated == 0:
                    # BLOCKED: order doesn't meet exchange requirements
                    logger.warning(f"🚫 Order blocked: {symbol} {side} {qty_to_use:.4f} - {reason}")
                    self._write_result(
                        plan_id, symbol, executed=False,
                        error=f"min_notional_check_failed:{reason}",
                        side=side, qty=qty_to_use
                    )
                    self._mark_done(plan_id)
                    return True
                
                qty_to_use = qty_validated
                notional = qty_to_use * mark_price
                logger.info(f"✅ Sizing validated: qty={qty_to_use:.4f}, price={mark_price:.2f}, notional={notional:.2f} USDT")
            
            # Exit ownership gate: only EXIT_OWNER can place reduceOnly orders
            if reduce_only and EXIT_OWNERSHIP_ENABLED:
                if source != EXIT_OWNER:
                    logger.warning(
                        f"🚫 DENY_NOT_EXIT_OWNER exec_boundary plan_id={plan_id[:8]} "
                        f"source={source} expected={EXIT_OWNER} symbol={symbol}"
                    )
                    self._write_result(
                        plan_id, symbol, executed=False,
                        decision="DENIED",
                        error=f"NOT_EXIT_OWNER:source={source}",
                        side=side, qty=qty_to_use
                    )
                    self._mark_done(plan_id)
                    return True
            
            # Set leverage on Binance before entry orders (LeverageEngine value)
            if not reduce_only:
                self._set_leverage(symbol, plan_leverage)

            # Execute Binance order
            logger.info(f"🚀 Executing Binance order: {symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only} leverage={plan_leverage}x")
            
            # Set inflight tracking (30s TTL as safety net)
            inflight_key = f"quantum:inflight:{symbol}"
            self.redis.setex(inflight_key, 30, order_id if 'order_id' in locals() else "pending")
            
            order_result = self._execute_binance_order(symbol, side, qty_to_use, reduce_only)
            
            if order_result["success"]:
                order_id = order_result.get("order_id")
                
                # Poll for fill confirmation
                fill_result = self._poll_order_fill(symbol, order_id, max_attempts=3)
                final_status = fill_result.get("status", order_result.get("status"))
                final_filled = fill_result.get("filled_qty", order_result.get("filled_qty", 0))
                
                logger.info(
                    f"✅ ORDER FILLED: {symbol} {side} qty={qty_to_use:.4f} "
                    f"order_id={order_id} status={final_status} filled={final_filled:.4f}"
                )
                
                logger.info(f"🔍 DEBUG: final_status='{final_status}' type={type(final_status)} equals_FILLED={final_status == 'FILLED'}")
                
                # Set execution-based cooldown timestamp (ms)
                cooldown_key = f"quantum:cooldown:last_exec_ts:{symbol}"
                self.redis.set(cooldown_key, str(int(time.time() * 1000)), ex=3600)  # 1h TTL — prevents permanent cooldown
                
                # Clear inflight tracking if order is terminal state
                if final_status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                    inflight_key = f"quantum:inflight:{symbol}"
                    self.redis.delete(inflight_key)
                    logger.debug(f"Cleared inflight tracking: {symbol}")
                
                # Update ledger with fresh position after order (optional, timer already syncs)
                if UPDATE_LEDGER_AFTER_EXEC:
                    self._update_ledger(symbol)
                
                # Exactly-once ledger commit (idempotent on order_id)
                if final_status == "FILLED":
                    logger.info(f"🔍 DEBUG: Calling ledger commit for {symbol} order_id={order_id}")
                    self._commit_ledger_exactly_once(symbol, order_id, final_filled, side)
                
                self._write_result(
                    plan_id, symbol, executed=True,
                    side=side,
                    qty=qty_to_use,
                    order_id=order_id,
                    filled_qty=final_filled,
                    order_status=final_status,
                    permit=permit
                )
                self._inc_redis_counter("processed")
                self._inc_redis_counter("executed_true")
                if lane == "manual":
                    self._inc_redis_counter("manual_processed")
            else:
                logger.error(f"❌ ORDER FAILED: {order_result.get('error')}")
                self._write_result(
                    plan_id, symbol, executed=False,
                    error=f"binance_error:{order_result.get('error')}",
                    side=side,
                    qty=qty_to_use,
                    permit=permit
                )
                self._inc_redis_counter("processed")
                self._inc_redis_counter("executed_false")
            
            # Mark as done
            self._mark_done(plan_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to process plan: {e}", exc_info=True)
            # Guaranteed result write on error
            try:
                plan_id = event_data.get(b"plan_id", b"").decode()
                symbol = event_data.get(b"symbol", b"").decode().upper()
                side = event_data.get(b"side", b"").decode().upper()
                qty = float(event_data.get(b"qty", b"0").decode() or 0)
                self._write_result(
                    plan_id, symbol, executed=False,
                    decision="ERROR",
                    error=f"exception:{str(e)[:100]}",
                    side=side, qty=qty
                )
                self._mark_done(plan_id)
            except Exception as write_err:
                logger.error(f"Failed to write error result: {write_err}")
            
            self._inc_redis_counter("processed")
            self._inc_redis_counter("executed_false")
            return True  # ACK to avoid blocking stream
    
    def _get_position_info(self, symbol: str) -> dict:
        """Get current position information from Binance for harvest processing"""
        try:
            timestamp = int(time.time() * 1000)
            query_params = {
                "symbol": symbol,
                "timestamp": timestamp
            }
            query_string = urllib.parse.urlencode(query_params)
            
            # Sign the request
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            query_params["signature"] = signature
            
            url = f"{BINANCE_BASE_URL}/fapi/v2/positionRisk?{urllib.parse.urlencode(query_params)}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                positions = json.loads(response.read().decode())
                pos = next((p for p in positions if p["symbol"] == symbol), None)
                
                if pos:
                    return {
                        "position_amt": float(pos.get("positionAmt", 0)),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "mark_price": float(pos.get("markPrice", 0)),
                        "unrealized_pnl": float(pos.get("unRealizedProfit", 0))
                    }
                else:
                    return {"position_amt": 0}
                    
        except Exception as e:
            logger.error(f"Failed to get position info for {symbol}: {e}")
            return {"position_amt": 0}
    
    def process_harvest_intent(self, stream_id: bytes, event_data: Dict):
        """Process autonomous harvest/exit intent from Autonomous Trader
        
        Harvest intents are simpler than apply plans:
        - No P3.3 permit required (already evaluated by ExitManager)
        - Direct position close (reduceOnly=True)
        - Symbol allowlist still applies for safety
        """
        stream_id_str = stream_id.decode()
        
        try:
            # Parse harvest intent
            symbol = event_data.get(b"symbol", b"").decode().upper()
            action = event_data.get(b"action", b"").decode().upper()
            percentage = float(event_data.get(b"percentage", b"1.0").decode() or 1.0)
            reason = event_data.get(b"reason", b"").decode()
            R_net = float(event_data.get(b"R_net", b"0").decode() or 0)
            pnl_usd = float(event_data.get(b"pnl_usd", b"0").decode() or 0)
            entry_price_intent = float(event_data.get(b"entry_price", b"0").decode() or 0)
            exit_price_intent = float(event_data.get(b"exit_price", b"0").decode() or 0)
            
            logger.info(f"🌾 HARVEST INTENT: {symbol} {action} ({percentage:.0%}) R={R_net:.2f} PnL=${pnl_usd:.2f} reason={reason}")
            
            # Validate symbol allowlist
            if symbol not in self.allowlist:
                logger.warning(f"❌ HARVEST BLOCKED: {symbol} not in allowlist")
                return True  # ACK but skip
            
            # Accept CLOSE and PARTIAL_CLOSE actions
            if action not in ["CLOSE", "PARTIAL_CLOSE"]:
                logger.warning(f"❌ HARVEST BLOCKED: unsupported action {action}")
                return True  # ACK but skip
            
            # For partial closes, treat as full close for now (simplicity)
            # TODO: Implement true partial close logic in future
            if action == "PARTIAL_CLOSE":
                logger.info(f"🔄 Converting PARTIAL_CLOSE to CLOSE (full exit)")
                percentage = 1.0  # Force 100% close for now
            
            # Get current position to calculate close quantity
            position_info = self._get_position_info(symbol)
            if not position_info or position_info["position_amt"] == 0:
                logger.info(f"ℹ️  HARVEST SKIP: {symbol} no position found")
                return True  # ACK - position already closed
            
            position_amt = float(position_info["position_amt"])
            
            # Use prices from intent (more reliable than position_info which may be stale after close)
            entry_price = entry_price_intent if entry_price_intent > 0 else float(position_info.get("entry_price", 0))
            exit_price = exit_price_intent if exit_price_intent > 0 else float(position_info.get("mark_price", 0))
            
            close_qty = abs(position_amt * percentage)
            side = "SELL" if position_amt > 0 else "BUY"
            
            # Calculate pnl_percent for CLM
            if entry_price > 0 and exit_price > 0:
                if position_amt > 0:  # LONG
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            else:
                pnl_percent = 0.0
            
            logger.info(f"🚀 HARVEST CLOSE: {symbol} {side} qty={close_qty:.4f} (pos={position_amt:.4f}) entry={entry_price} exit={exit_price}")
            
            # Execute close order
            order_result = self._execute_binance_order(
                symbol=symbol,
                side=side,
                qty=close_qty,
                reduce_only=True
            )
            
            if order_result.get("success"):
                order_id = order_result.get("order_id")
                final_filled = order_result.get("filled_qty", close_qty)
                logger.info(f"✅ HARVEST SUCCESS: {symbol} closed {final_filled:.4f} orderId={order_id}")
                self._inc_redis_counter("harvest_executed")

                # Clean up stale position keys to unblock intent-bridge
                try:
                    _hc_snap = f"quantum:position:snapshot:{symbol}"
                    _hc_pos = f"quantum:position:{symbol}"
                    _hc_ds = self.redis.delete(_hc_snap)
                    _hc_dp = self.redis.delete(_hc_pos)
                    if _hc_ds or _hc_dp:
                        logger.info(f"HARVEST CLEANUP: {symbol} deleted snapshot={_hc_ds} position={_hc_dp}")
                except Exception as _hc_e:
                    logger.error(f"Harvest cleanup failed for {symbol}: {_hc_e}")

                # 📤 Publish trade.closed event for SimpleCLM/calibration pipeline
                try:
                    close_event = {
                        "event_type": "trade.closed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "side": "LONG" if side == "SELL" else "SHORT",  # Position side, not order side
                        "entry_price": str(entry_price),
                        "exit_price": str(exit_price),
                        "pnl_percent": str(round(pnl_percent, 2)),
                        "confidence": "0.7",  # Default (harvest exits don't have explicit confidence)
                        "model_id": "autonomous_exit",
                        "R_net": str(round(R_net, 2)),
                        "pnl_usd": str(round(pnl_usd, 2)),
                        "reason": reason,
                        "order_id": str(order_id),
                        "source": "autonomous_trader"
                    }
                    self.redis.xadd(
                        "quantum:stream:trade.closed",
                        close_event,
                        maxlen=1000  # Keep last 1000
                    )
                    logger.info(f"📤 Published trade.closed: {symbol} PnL={pnl_percent:.1f}% R={R_net:.2f}")
                except Exception as e:
                    logger.error(f"❌ Failed to publish trade.closed: {e}")
                
            else:
                error = order_result.get("error", "unknown")
                logger.error(f"❌ HARVEST FAILED: {symbol} - {error}")
                self._inc_redis_counter("harvest_failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing harvest intent: {e}", exc_info=True)
            return True  # ACK to avoid blocking
    
    def run(self):
        """Main processing loop - consumes main and manual lanes"""
        logger.info("✅ Redis connected")
        logger.info("🚀 Intent Executor started")
        logger.info(f"📨 Consuming MAIN: {APPLY_PLAN_STREAM}")
        logger.info(f"📨 Consuming MANUAL: {MANUAL_STREAM}")
        logger.info(f"📨 Consuming HARVEST: {HARVEST_STREAM}")
        
        last_id_main = ">"  # Only new messages
        last_id_manual = ">"
        last_id_harvest = ">"
        
        while True:
            try:
                # P2 Universe: Refresh allowlist cache if expired
                self._refresh_allowlist_if_needed()
                
                # Emit periodic heartbeat
                self._emit_heartbeat()
                
                # Read from MAIN lane (XREADGROUP)
                messages = self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {APPLY_PLAN_STREAM: last_id_main},
                    count=10,
                    block=1000  # 1 sec timeout
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, event_data in stream_messages:
                            # Process plan (main lane)
                            ack = self.process_plan(message_id, event_data, lane="main")
                            
                            # ACK message
                            if ack:
                                self.redis.xack(APPLY_PLAN_STREAM, CONSUMER_GROUP, message_id)
                
                # Read from MANUAL lane (always check, TTL guard inside process_plan)
                try:
                    manual_messages = self.redis.xreadgroup(
                        MANUAL_GROUP,
                        f"{CONSUMER_NAME}-manual",
                        {MANUAL_STREAM: last_id_manual},
                        count=5,
                        block=500  # 0.5 sec timeout
                    )
                    
                    if manual_messages:
                        for stream_name, stream_messages in manual_messages:
                            for message_id, event_data in stream_messages:
                                # Process plan (manual lane - TTL guard inside)
                                ack = self.process_plan(message_id, event_data, lane="manual")
                                
                                # ACK message
                                if ack:
                                    self.redis.xack(MANUAL_STREAM, MANUAL_GROUP, message_id)
                except Exception as e:
                    logger.error(f"Error reading manual lane: {e}")
                
                # Read from HARVEST lane (autonomous exits)
                try:
                    harvest_messages = self.redis.xreadgroup(
                        HARVEST_GROUP,
                        f"{CONSUMER_NAME}-harvest",
                        {HARVEST_STREAM: last_id_harvest},
                        count=5,
                        block=200  # 0.2 sec timeout
                    )
                    
                    if harvest_messages:
                        for stream_name, stream_messages in harvest_messages:
                            for message_id, event_data in stream_messages:
                                # Process harvest intent (autonomous exit)
                                ack = self.process_harvest_intent(message_id, event_data)
                                
                                # ACK message
                                if ack:
                                    self.redis.xack(HARVEST_STREAM, HARVEST_GROUP, message_id)
                except Exception as e:
                    logger.error(f"Error reading harvest lane: {e}")
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)


def main():
    executor = IntentExecutor()
    executor.run()


if __name__ == "__main__":
    main()
