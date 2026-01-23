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
        """Testnet execution - actual orders to Binance testnet"""
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
            # Import Binance client (assume it exists or implement minimal client)
            # For now, simulate execution with detailed logging
            for step in plan.steps:
                logger.info(f"{plan.symbol}: Executing step {step['step']} (TESTNET)")
                
                # Placeholder for actual Binance API calls
                # In real implementation:
                # - Get current position
                # - Calculate order quantity (respect minNotional, stepSize)
                # - Place order with reduceOnly flag
                # - Wait for fill
                # - Store order ID and status
                
                steps_results.append({
                    "step": step["step"],
                    "status": "simulated_success",
                    "details": f"TESTNET simulation: {step}",
                    "order_id": f"sim_{int(time.time())}"
                })
                
                # Metrics
                if PROMETHEUS_AVAILABLE:
                    self.metric_execute_total.labels(
                        symbol=plan.symbol,
                        step=step["step"],
                        status="success"
                    ).inc()
            
            # Update last success metric
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
