#!/usr/bin/env python3
"""
P3.4 Position Reconciliation Engine

Continuously reconciles:
- Exchange snapshot (from P3.3)
- Internal ledger
- Apply result evidence

Auto-repairs when safe (fail-closed design).
Sets HOLDs when unsafe.
Never places orders - only repairs internal state.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Install with: pip install redis")
    sys.exit(1)

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("WARN: prometheus_client not installed, metrics disabled")


logging.basicConfig(
    level=os.getenv("P34_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---- Configuration ----
REDIS_HOST = os.getenv("QUANTUM_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("QUANTUM_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("QUANTUM_REDIS_DB", "0"))

ALLOWLIST = os.getenv("P34_ALLOWLIST", "BTCUSDT").split(",")
LOOP_CADENCE_SEC = float(os.getenv("P34_LOOP_CADENCE_SEC", "1.0"))
PROMETHEUS_PORT = int(os.getenv("P34_PROMETHEUS_PORT", "8046"))

# Drift detection thresholds
EXCHANGE_FRESHNESS_SEC = int(os.getenv("P34_EXCHANGE_FRESHNESS_SEC", "10"))
QTY_DRIFT_TOLERANCE_PCT = float(os.getenv("P34_QTY_DRIFT_TOLERANCE_PCT", "1.0"))  # 1%
EVIDENCE_WINDOW_SEC = int(os.getenv("P34_EVIDENCE_WINDOW_SEC", "300"))  # 5 min
HOLD_TTL_SEC = int(os.getenv("P34_HOLD_TTL_SEC", "300"))  # 5 min

# Prometheus Metrics
if PROMETHEUS_AVAILABLE:
    p34_drift_detected = Counter('p34_reconcile_drift_total', 'Total drifts detected', ['symbol', 'reason'])
    p34_auto_repair = Counter('p34_reconcile_auto_repair_total', 'Total auto-repairs', ['symbol'])
    p34_hold_active = Gauge('p34_reconcile_hold_active', 'Hold active (1=yes)', ['symbol'])
    p34_diff_amt = Gauge('p34_reconcile_diff_amt', 'Position diff amount', ['symbol'])
    p34_last_fix_age_sec = Gauge('p34_reconcile_last_fix_age_sec', 'Seconds since last fix', ['symbol'])


class ReconcileStatus(str, Enum):
    OK = "OK"
    DRIFT = "DRIFT"
    HOLD = "HOLD"
    REPAIRED = "REPAIRED"
    MANUAL_REQUIRED = "MANUAL_REQUIRED"


@dataclass
class ExchangeSnapshot:
    """Exchange position snapshot (from P3.3)"""
    position_amt: float
    side: str
    entry_price: float
    mark_price: float
    ts_epoch: int
    
    @classmethod
    def from_redis_hash(cls, data: dict):
        if not data:
            return None
        return cls(
            position_amt=float(data.get(b'position_amt', 0)),
            side=data.get(b'side', b'').decode('utf-8'),
            entry_price=float(data.get(b'entry_price', 0)),
            mark_price=float(data.get(b'mark_price', 0)),
            ts_epoch=int(data.get(b'ts_epoch', 0))
        )


@dataclass
class Ledger:
    """Internal ledger state"""
    ledger_amt: float
    ledger_side: str
    ts_epoch: int
    source: str  # p34_auto, p34_manual, apply, init
    version: int
    
    @classmethod
    def from_redis_hash(cls, data: dict):
        if not data:
            return None
        return cls(
            ledger_amt=float(data.get(b'ledger_amt', 0)),
            ledger_side=data.get(b'ledger_side', b'').decode('utf-8'),
            ts_epoch=int(data.get(b'ts_epoch', 0)),
            source=data.get(b'source', b'init').decode('utf-8'),
            version=int(data.get(b'version', 0))
        )


@dataclass
class ApplyResult:
    """Apply result from stream"""
    plan_id: str
    symbol: str
    executed: bool
    filled_qty: float
    side: str
    reduce_only: bool
    order_id: str
    ts: int


class ReconcileEngine:
    """P3.4 Position Reconciliation Engine"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        
        logger.info(f"P3.4 Reconcile Engine initialized")
        logger.info(f"Allowlist: {ALLOWLIST}")
        logger.info(f"Loop cadence: {LOOP_CADENCE_SEC}s")
        logger.info(f"Exchange freshness: {EXCHANGE_FRESHNESS_SEC}s")
        logger.info(f"Drift tolerance: {QTY_DRIFT_TOLERANCE_PCT}%")
    
    def run(self):
        """Main reconciliation loop"""
        logger.info("Starting reconciliation loop...")
        
        while True:
            try:
                for symbol in ALLOWLIST:
                    symbol = symbol.strip()
                    if not symbol:
                        continue
                    
                    self.reconcile_symbol(symbol)
                
                time.sleep(LOOP_CADENCE_SEC)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(LOOP_CADENCE_SEC)
    
    def reconcile_symbol(self, symbol: str):
        """Reconcile single symbol"""
        try:
            # 1. Read exchange snapshot
            exchange = self._read_exchange_snapshot(symbol)
            if not exchange:
                self._handle_missing_exchange(symbol)
                return
            
            # 2. Check exchange freshness
            age_sec = int(time.time()) - exchange.ts_epoch
            if age_sec > EXCHANGE_FRESHNESS_SEC:
                self._handle_stale_exchange(symbol, age_sec)
                return
            
            # 3. Read or initialize ledger
            ledger = self._read_ledger(symbol)
            if not ledger:
                # Initialize from exchange if fresh
                ledger = self._initialize_ledger(symbol, exchange)
                if not ledger:
                    return
            
            # 4. Detect drift
            drift = self._detect_drift(symbol, exchange, ledger)
            if not drift:
                # All OK
                self._update_state(symbol, ReconcileStatus.OK, "", exchange, ledger, 0)
                self._clear_hold(symbol)
                return
            
            # 5. Analyze drift and decide action
            reason, diff_amt, diff_pct = drift
            
            # Side mismatch is always HOLD (not auto-fixable)
            if reason == "side_mismatch":
                self._handle_side_mismatch(symbol, exchange, ledger)
                return
            
            # Qty mismatch: try auto-repair if evidence exists
            if reason == "qty_mismatch":
                self._handle_qty_mismatch(symbol, exchange, ledger, diff_amt, diff_pct)
                return
            
        except Exception as e:
            logger.error(f"{symbol}: Reconcile error: {e}", exc_info=True)
    
    def _read_exchange_snapshot(self, symbol: str) -> Optional[ExchangeSnapshot]:
        """Read exchange snapshot from P3.3"""
        key = f"quantum:position:snapshot:{symbol}"
        data = self.redis.hgetall(key)
        return ExchangeSnapshot.from_redis_hash(data)
    
    def _read_ledger(self, symbol: str) -> Optional[Ledger]:
        """Read internal ledger"""
        key = f"quantum:position:ledger:{symbol}"
        data = self.redis.hgetall(key)
        return Ledger.from_redis_hash(data)
    
    def _initialize_ledger(self, symbol: str, exchange: ExchangeSnapshot) -> Optional[Ledger]:
        """Initialize ledger from fresh exchange snapshot"""
        logger.info(f"{symbol}: Initializing ledger from exchange snapshot")
        
        ledger = Ledger(
            ledger_amt=exchange.position_amt,
            ledger_side=exchange.side,
            ts_epoch=int(time.time()),
            source="p34_init",
            version=1
        )
        
        key = f"quantum:position:ledger:{symbol}"
        self.redis.hset(key, mapping={
            "ledger_amt": ledger.ledger_amt,
            "ledger_side": ledger.ledger_side,
            "ts_epoch": ledger.ts_epoch,
            "source": ledger.source,
            "version": ledger.version
        })
        
        self._emit_event("LEDGER_INIT", symbol, {
            "ledger_amt": ledger.ledger_amt,
            "ledger_side": ledger.ledger_side,
            "source": "exchange_snapshot"
        })
        
        return ledger
    
    def _detect_drift(self, symbol: str, exchange: ExchangeSnapshot, ledger: Ledger):
        """
        Detect drift between exchange and ledger.
        Returns (reason, diff_amt, diff_pct) or None if OK.
        """
        # Side mismatch
        if exchange.side != ledger.ledger_side:
            if abs(exchange.position_amt) > 0.0001 or abs(ledger.ledger_amt) > 0.0001:
                return ("side_mismatch", 0, 0)
        
        # Qty mismatch
        diff_amt = abs(exchange.position_amt - ledger.ledger_amt)
        if diff_amt < 0.0001:
            return None  # OK
        
        # Calculate diff percentage
        base = max(abs(exchange.position_amt), abs(ledger.ledger_amt), 0.0001)
        diff_pct = (diff_amt / base) * 100
        
        if diff_pct > QTY_DRIFT_TOLERANCE_PCT:
            return ("qty_mismatch", diff_amt, diff_pct)
        
        return None  # OK
    
    def _handle_missing_exchange(self, symbol: str):
        """Handle missing exchange snapshot"""
        logger.warning(f"{symbol}: Missing exchange snapshot")
        self._set_hold(symbol, "missing_exchange", None, None)
        self._emit_event("MANUAL_REQUIRED", symbol, {"reason": "missing_exchange"})
        
        if PROMETHEUS_AVAILABLE:
            p34_drift_detected.labels(symbol=symbol, reason="missing_exchange").inc()
            p34_hold_active.labels(symbol=symbol).set(1)
    
    def _handle_stale_exchange(self, symbol: str, age_sec: int):
        """Handle stale exchange snapshot"""
        logger.warning(f"{symbol}: Stale exchange snapshot ({age_sec}s old)")
        self._set_hold(symbol, "stale_exchange", None, None)
        self._emit_event("MANUAL_REQUIRED", symbol, {"reason": "stale_exchange", "age_sec": age_sec})
        
        if PROMETHEUS_AVAILABLE:
            p34_drift_detected.labels(symbol=symbol, reason="stale_exchange").inc()
            p34_hold_active.labels(symbol=symbol).set(1)
    
    def _handle_side_mismatch(self, symbol: str, exchange: ExchangeSnapshot, ledger: Ledger):
        """Handle side mismatch (not auto-fixable)"""
        logger.error(
            f"{symbol}: SIDE MISMATCH - exchange={exchange.side}({exchange.position_amt}) "
            f"ledger={ledger.ledger_side}({ledger.ledger_amt})"
        )
        
            self._set_hold(symbol, "side_mismatch", exchange, ledger)
        self._update_state(
            symbol, ReconcileStatus.HOLD, "side_mismatch",
            exchange, ledger, 0
        )
        self._emit_event("MANUAL_REQUIRED", symbol, {
            "reason": "side_mismatch",
            "exchange_side": exchange.side,
            "exchange_amt": exchange.position_amt,
            "ledger_side": ledger.ledger_side,
            "ledger_amt": ledger.ledger_amt
        })
        
        if PROMETHEUS_AVAILABLE:
            p34_drift_detected.labels(symbol=symbol, reason="side_mismatch").inc()
            p34_hold_active.labels(symbol=symbol).set(1)
    
    def _handle_qty_mismatch(self, symbol: str, exchange: ExchangeSnapshot, ledger: Ledger, diff_amt: float, diff_pct: float):
        """Handle qty mismatch - try auto-repair with evidence"""
        logger.warning(
            f"{symbol}: QTY MISMATCH - exchange={exchange.position_amt} "
            f"ledger={ledger.ledger_amt} diff={diff_amt:.4f} ({diff_pct:.2f}%)"
        )
        
        # Look for evidence in apply.result stream
        evidence = self._find_evidence(symbol, diff_amt)
        
        if evidence:
            # Auto-repair possible
            self._auto_repair(symbol, exchange, ledger, diff_amt, diff_pct, evidence)
        else:
            # No evidence - set HOLD
            logger.warning(f"{symbol}: No evidence for drift - setting HOLD")
                self._set_hold(symbol, "qty_mismatch_no_evidence", exchange, ledger)
            self._update_state(
                symbol, ReconcileStatus.HOLD, "qty_mismatch_no_evidence",
                exchange, ledger, diff_amt
            )
            self._emit_event("MANUAL_REQUIRED", symbol, {
                "reason": "qty_mismatch_no_evidence",
                "exchange_amt": exchange.position_amt,
                "ledger_amt": ledger.ledger_amt,
                "diff_amt": diff_amt,
                "diff_pct": diff_pct
            })
            
            if PROMETHEUS_AVAILABLE:
                p34_drift_detected.labels(symbol=symbol, reason="qty_mismatch_no_evidence").inc()
                p34_hold_active.labels(symbol=symbol).set(1)
                p34_diff_amt.labels(symbol=symbol).set(diff_amt)
    
    def _find_evidence(self, symbol: str, diff_amt: float) -> Optional[Dict]:
        """
        Look for evidence in apply.result stream.
        Returns most recent executed=True result that explains drift.
        """
        stream_key = "quantum:stream:apply.result"
        
        # Read last 50 results
        try:
            results = self.redis.xrevrange(stream_key, count=50)
        except Exception as e:
            logger.error(f"{symbol}: Error reading apply.result stream: {e}")
            return None
        
        now = int(time.time())
        
        for result_id, result_data in results:
            try:
                # Decode result
                res_symbol = result_data.get(b'symbol', b'').decode('utf-8')
                if res_symbol != symbol:
                    continue
                
                res_ts = int(result_data.get(b'ts', 0))
                age = now - res_ts
                if age > EVIDENCE_WINDOW_SEC:
                    continue  # Too old
                
                res_executed = result_data.get(b'executed', b'false').decode('utf-8') == 'true'
                if not res_executed:
                    continue  # Not executed
                
                res_filled_qty = float(result_data.get(b'filled_qty', 0))
                
                # Check if filled_qty explains drift
                if abs(res_filled_qty - diff_amt) < 0.0001:
                    # This result explains the drift
                    return {
                        "result_id": result_id.decode('utf-8'),
                        "plan_id": result_data.get(b'plan_id', b'').decode('utf-8'),
                        "filled_qty": res_filled_qty,
                        "side": result_data.get(b'side', b'').decode('utf-8'),
                        "order_id": result_data.get(b'order_id', b'').decode('utf-8'),
                        "ts": res_ts,
                        "age_sec": age
                    }
                
            except Exception as e:
                logger.error(f"{symbol}: Error parsing result: {e}")
                continue
        
        return None
    
    def _auto_repair(self, symbol: str, exchange: ExchangeSnapshot, ledger: Ledger, diff_amt: float, diff_pct: float, evidence: Dict):
        """Auto-repair ledger to match exchange"""
        logger.info(
            f"{symbol}: AUTO-REPAIR - aligning ledger to exchange "
            f"(evidence: plan={evidence['plan_id']} filled={evidence['filled_qty']:.4f})"
        )
        
        # Update ledger
        new_ledger = Ledger(
            ledger_amt=exchange.position_amt,
            ledger_side=exchange.side,
            ts_epoch=int(time.time()),
            source="p34_auto",
            version=ledger.version + 1
        )
        
        key = f"quantum:position:ledger:{symbol}"
        self.redis.hset(key, mapping={
            "ledger_amt": new_ledger.ledger_amt,
            "ledger_side": new_ledger.ledger_side,
            "ts_epoch": new_ledger.ts_epoch,
            "source": new_ledger.source,
            "version": new_ledger.version
        })
        
        # Update state
        self._update_state(
            symbol, ReconcileStatus.REPAIRED, "auto_fixed",
            exchange, new_ledger, 0
        )
        
        # Clear hold
        self._clear_hold(symbol)
        
        # Emit event
        self._emit_event("AUTO_REPAIR", symbol, {
            "old_ledger_amt": ledger.ledger_amt,
            "new_ledger_amt": new_ledger.ledger_amt,
            "diff_amt": diff_amt,
            "diff_pct": diff_pct,
            "evidence_plan_id": evidence['plan_id'],
            "evidence_filled_qty": evidence['filled_qty'],
            "evidence_order_id": evidence['order_id'],
            "evidence_age_sec": evidence['age_sec']
        })
        
        if PROMETHEUS_AVAILABLE:
            p34_auto_repair.labels(symbol=symbol).inc()
            p34_hold_active.labels(symbol=symbol).set(0)
            p34_diff_amt.labels(symbol=symbol).set(0)
            p34_last_fix_age_sec.labels(symbol=symbol).set(0)
    
    def _set_hold(self, symbol: str, reason: str):
            def _set_hold(self, symbol: str, reason: str, exchange=None, ledger=None):
        """Set reconcile hold"""
            """Set reconcile hold and publish RECONCILE_CLOSE plan"""
        key = f"quantum:reconcile:hold:{symbol}"
        self.redis.setex(key, HOLD_TTL_SEC, "1")
        logger.warning(f"{symbol}: HOLD set (reason={reason}, TTL={HOLD_TTL_SEC}s)")
        
        self._emit_event("HOLD_SET", symbol, {"reason": reason, "ttl_sec": HOLD_TTL_SEC})
            if exchange and ledger:
                self._publish_reconcile_close_plan(symbol, exchange, ledger, reason)
    
        def _publish_reconcile_close_plan(self, symbol: str, exchange: ExchangeSnapshot, ledger: Ledger, reason: str):
            """Publish RECONCILE_CLOSE plan when drift detected (Patch A)"""
            import time
        
            exchange_amt = exchange.position_amt
            ledger_amt = ledger.ledger_amt
            exchange_side = exchange.side
        
            if exchange_amt == 0:
                return
        
            close_side = "SELL" if exchange_amt > 0 else "BUY"
        
            qty = min(abs(exchange_amt), abs(exchange_amt - ledger_amt))
            if qty <= 0:
                return
        
            sig_key = f"{symbol}:{exchange_side}:{round(abs(exchange_amt), 6)}:{round(abs(ledger_amt), 6)}"
            signature = hashlib.md5(sig_key.encode()).hexdigest()[:12]
        
            cooldown_bucket = int(time.time() / 120)
            cooldown_key = f"quantum:reconcile:close:cooldown:{symbol}:{signature}:{cooldown_bucket}"
        
            if self.redis.exists(cooldown_key):
                return
        
            now_ms = int(time.time() * 1000)
            plan_id = f"reconclose:{symbol}:{signature}:{now_ms}"
        
            plan = {
                "plan_id": plan_id,
                "decision": "RECONCILE_CLOSE",
                "symbol": symbol,
                "side": close_side,
                "type": "MARKET",
                "qty": qty,
                "reduceOnly": True,
                "reason": "reconcile_drift",
                "source": "p3.4",
                "exchange_amt": exchange_amt,
                "ledger_amt": ledger_amt,
                "ts": now_ms,
            }
        
            try:
                self.redis.xadd("quantum:stream:trading.plan", plan, id="*")
                self.redis.setex(cooldown_key, 120, "1")
                logger.info(f"{symbol}: RECONCILE_CLOSE plan published - plan_id={plan_id}, qty={qty}, reason={reason}")
                if PROMETHEUS_AVAILABLE:
                    p34_drift_detected.labels(symbol=symbol, reason="reconcile_close_plan").inc()
            except Exception as e:
                logger.error(f"{symbol}: Error publishing RECONCILE_CLOSE plan: {e}")
    
    def _clear_hold(self, symbol: str):
        """Clear reconcile hold"""
        key = f"quantum:reconcile:hold:{symbol}"
        deleted = self.redis.delete(key)
        if deleted:
            logger.info(f"{symbol}: HOLD cleared")
            self._emit_event("HOLD_CLEARED", symbol, {})
        
        if PROMETHEUS_AVAILABLE:
            p34_hold_active.labels(symbol=symbol).set(0)
    
    def _update_state(self, symbol: str, status: ReconcileStatus, reason: str, exchange: ExchangeSnapshot, ledger: Ledger, diff_amt: float):
        """Update reconcile state"""
        key = f"quantum:reconcile:state:{symbol}"
        
        self.redis.hset(key, mapping={
            "status": status.value,
            "reason": reason,
            "exchange_amt": exchange.position_amt,
            "ledger_amt": ledger.ledger_amt,
            "diff_amt": diff_amt,
            "diff_pct": (diff_amt / max(abs(exchange.position_amt), 0.0001)) * 100 if diff_amt > 0 else 0,
            "last_seen_ts": int(time.time())
        })
        
        if PROMETHEUS_AVAILABLE:
            p34_diff_amt.labels(symbol=symbol).set(diff_amt)
    
    def _emit_event(self, event: str, symbol: str, details: Dict):
        """Emit reconcile event to stream"""
        stream_key = "quantum:stream:reconcile.events"
        
        data = {
            "event": event,
            "symbol": symbol,
            "ts": int(time.time()),
            **details
        }
        
        try:
            self.redis.xadd(stream_key, data, maxlen=10000)
        except Exception as e:
            logger.error(f"{symbol}: Error emitting event: {e}")


def main():
    """Main entry point"""
    logger.info("═══════════════════════════════════════════════════════════════")
    logger.info("P3.4 Position Reconciliation Engine - Starting")
    logger.info("═══════════════════════════════════════════════════════════════")
    logger.info(f"Version: 1.0.0")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT} DB={REDIS_DB}")
    logger.info(f"Allowlist: {ALLOWLIST}")
    logger.info(f"Prometheus: port {PROMETHEUS_PORT}")
    logger.info("═══════════════════════════════════════════════════════════════")
    
    # Start Prometheus metrics server
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(PROMETHEUS_PORT)
            logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    # Create and run engine
    engine = ReconcileEngine()
    engine.run()


if __name__ == "__main__":
    main()
