#!/usr/bin/env python3
"""
P3.1 Integration - Allocation Target Shadow Proposer

Reads existing allocation targets from quantum:allocation:target:{symbol}
and P3.1 Capital Efficiency scores from quantum:capital:efficiency:{symbol},
computes proposed targets with efficiency-based multiplier, and writes to
shadow stream/keys WITHOUT overwriting live targets.

Architecture:
- Port: 8065 (Prometheus metrics)
- Mode: SHADOW ONLY (never writes to quantum:allocation:target:{symbol})
- Loop: every 10s
- Fail-open: missing/stale/low-conf efficiency → multiplier = 1.0 (no-op)

Redis Outputs:
1. Stream: quantum:stream:allocation.target.proposed (always)
2. Shadow key: quantum:allocation:target:proposed:{symbol} (TTL 600s)

Author: Quantum Trading OS
Date: 2026-01-28
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import redis
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
# CONFIGURATION
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# P3.1 Integration parameters
P31_MIN_CONF = float(os.getenv("P31_MIN_CONF", "0.65"))
P31_STALE_SEC = int(os.getenv("P31_STALE_SEC", "600"))
P31_MIN_MULT = float(os.getenv("P31_MIN_MULT", "0.5"))
P31_MAX_MULT = float(os.getenv("P31_MAX_MULT", "1.5"))
P31_SCALE = float(os.getenv("P31_SCALE", "1.0"))
P31_BASE = float(os.getenv("P31_BASE", "1.0"))

# Stream/Key names
PROPOSE_STREAM_NAME = os.getenv("PROPOSE_STREAM_NAME", "quantum:stream:allocation.target.proposed")
PROPOSE_KEY_TTL = int(os.getenv("PROPOSE_KEY_TTL", "600"))

# Service parameters
LOOP_INTERVAL_SEC = int(os.getenv("LOOP_INTERVAL_SEC", "10"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8065"))

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
METRIC_LOOPS = Counter('p29_shadow_loops_total', 'Processing loops')
METRIC_PROPOSED = Counter('p29_shadow_proposed_total', 'Proposed targets', ['reason'])
METRIC_MULTIPLIER = Gauge('p29_shadow_multiplier', 'Applied multiplier', ['symbol'])
METRIC_EFF_CONF = Gauge('p29_shadow_eff_confidence', 'Efficiency confidence', ['symbol'])
METRIC_EFF_SCORE = Gauge('p29_shadow_eff_score', 'Efficiency score', ['symbol'])

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AllocationTarget:
    """Existing allocation target from P2.9"""
    symbol: str
    target_usd: float
    confidence: float
    timestamp: int
    mode: str

@dataclass
class EfficiencyData:
    """P3.1 Capital Efficiency data"""
    score: float
    confidence: float
    timestamp: int
    stale: bool

@dataclass
class ProposedTarget:
    """Shadow proposed target with explainability"""
    symbol: str
    base_target: float
    proposed_target: float
    multiplier: float
    eff_score: Optional[float]
    eff_confidence: Optional[float]
    eff_stale: bool
    reason: str
    ts_epoch: int
    mode: str = "shadow"

# ============================================================================
# REDIS CLIENT
# ============================================================================

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        # Test connection
        self.client.ping()
        logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    
    def get_allocation_target(self, symbol: str) -> Optional[AllocationTarget]:
        """Read existing allocation target from quantum:allocation:target:{symbol}"""
        try:
            key = f"quantum:allocation:target:{symbol}"
            data = self.client.hgetall(key)
            
            if not data:
                return None
            
            return AllocationTarget(
                symbol=symbol,
                target_usd=float(data.get('target_usd', 0)),
                confidence=float(data.get('confidence', 0)),
                timestamp=int(data.get('timestamp', 0)),
                mode=data.get('mode', 'shadow')
            )
        except Exception as e:
            logger.error(f"Error reading allocation target for {symbol}: {e}")
            return None
    
    def get_efficiency_data(self, symbol: str, now_ts: int) -> Optional[EfficiencyData]:
        """Read P3.1 efficiency data from quantum:capital:efficiency:{symbol}"""
        try:
            key = f"quantum:capital:efficiency:{symbol}"
            data = self.client.hgetall(key)
            
            if not data:
                return None
            
            score = float(data.get('efficiency_score', 0))
            confidence = float(data.get('confidence', 0))
            timestamp = int(data.get('ts', 0))
            
            # Check staleness
            age = now_ts - timestamp
            stale = age > P31_STALE_SEC
            
            return EfficiencyData(
                score=score,
                confidence=confidence,
                timestamp=timestamp,
                stale=stale
            )
        except Exception as e:
            logger.error(f"Error reading efficiency data for {symbol}: {e}")
            return None
    
    def get_active_symbols(self) -> list:
        """Get list of active symbols from allocation target keys"""
        try:
            pattern = "quantum:allocation:target:*"
            keys = self.client.keys(pattern)
            symbols = [k.split(':')[-1] for k in keys]
            return symbols
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            return []
    
    def write_proposed_stream(self, proposed: ProposedTarget):
        """Write proposed target to stream"""
        try:
            event = {
                "ts_epoch": str(proposed.ts_epoch),
                "symbol": proposed.symbol,
                "base_target": str(proposed.base_target),
                "proposed_target": str(proposed.proposed_target),
                "multiplier": f"{proposed.multiplier:.4f}",
                "eff_score": str(proposed.eff_score) if proposed.eff_score is not None else "",
                "eff_confidence": str(proposed.eff_confidence) if proposed.eff_confidence is not None else "",
                "eff_stale": "1" if proposed.eff_stale else "0",
                "reason": proposed.reason,
                "mode": proposed.mode
            }
            self.client.xadd(PROPOSE_STREAM_NAME, event, maxlen=1000)
        except Exception as e:
            logger.error(f"Error writing proposed stream: {e}")
    
    def write_proposed_key(self, proposed: ProposedTarget):
        """Write proposed target to shadow key"""
        try:
            key = f"quantum:allocation:target:proposed:{proposed.symbol}"
            data = {
                "ts_epoch": str(proposed.ts_epoch),
                "base_target": f"{proposed.base_target:.2f}",
                "proposed_target": f"{proposed.proposed_target:.2f}",
                "multiplier": f"{proposed.multiplier:.4f}",
                "eff_score": f"{proposed.eff_score:.4f}" if proposed.eff_score is not None else "",
                "eff_confidence": f"{proposed.eff_confidence:.3f}" if proposed.eff_confidence is not None else "",
                "eff_stale": "1" if proposed.eff_stale else "0",
                "reason": proposed.reason,
                "mode": proposed.mode,
                "last_update_ts_epoch": str(proposed.ts_epoch)
            }
            self.client.hset(key, mapping=data)
            self.client.expire(key, PROPOSE_KEY_TTL)
        except Exception as e:
            logger.error(f"Error writing proposed key for {proposed.symbol}: {e}")

# ============================================================================
# ALLOCATION SHADOW PROPOSER
# ============================================================================

class AllocationShadowProposer:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        logger.info("Allocation Shadow Proposer initialized")
        logger.info(f"P3.1 params: MIN_CONF={P31_MIN_CONF}, STALE_SEC={P31_STALE_SEC}")
        logger.info(f"Multiplier range: [{P31_MIN_MULT}, {P31_MAX_MULT}], BASE={P31_BASE}, SCALE={P31_SCALE}")
    
    def compute_multiplier(self, eff: Optional[EfficiencyData], now_ts: int) -> Tuple[float, str]:
        """
        Compute efficiency-based multiplier with fail-open logic.
        
        Returns:
            (multiplier: float, reason: str)
        
        Fail-open cases (multiplier=1.0):
        - missing_eff: No efficiency data
        - stale_eff: Data older than P31_STALE_SEC
        - low_conf: Confidence < P31_MIN_CONF
        - redis_error: Exception reading data
        
        Success case (reason=ok):
        - multiplier = clamp(MIN_MULT, MAX_MULT, BASE + SCALE*(score-0.5))
        """
        # Fail-open case 1: missing efficiency data
        if eff is None:
            return 1.0, "missing_eff"
        
        # Fail-open case 2: stale efficiency data
        if eff.stale:
            return 1.0, "stale_eff"
        
        # Fail-open case 3: low confidence
        if eff.confidence < P31_MIN_CONF:
            return 1.0, "low_conf"
        
        # Success case: compute multiplier
        # Formula: BASE + SCALE * (score - 0.5)
        # Example: BASE=1.0, SCALE=1.0
        #   score=0.0 → 1.0 + 1.0*(-0.5) = 0.5 → clamp to MIN_MULT=0.5
        #   score=0.5 → 1.0 + 1.0*(0) = 1.0 (neutral)
        #   score=1.0 → 1.0 + 1.0*(0.5) = 1.5 → clamp to MAX_MULT=1.5
        raw_mult = P31_BASE + P31_SCALE * (eff.score - 0.5)
        mult = max(P31_MIN_MULT, min(P31_MAX_MULT, raw_mult))
        
        return mult, "ok"
    
    def process_symbol(self, symbol: str, now_ts: int):
        """Process one symbol: read targets, compute proposal, write shadow outputs"""
        try:
            # Read base allocation target
            target = self.redis.get_allocation_target(symbol)
            if target is None:
                logger.debug(f"{symbol}: No base allocation target found, skipping")
                return
            
            # Read efficiency data
            eff = self.redis.get_efficiency_data(symbol, now_ts)
            
            # Compute multiplier
            mult, reason = self.compute_multiplier(eff, now_ts)
            
            # Compute proposed target
            proposed_target = target.target_usd * mult
            
            # Create proposed target object
            proposed = ProposedTarget(
                symbol=symbol,
                base_target=target.target_usd,
                proposed_target=proposed_target,
                multiplier=mult,
                eff_score=eff.score if eff else None,
                eff_confidence=eff.confidence if eff else None,
                eff_stale=eff.stale if eff else False,
                reason=reason,
                ts_epoch=now_ts,
                mode="shadow"
            )
            
            # Write outputs
            self.redis.write_proposed_stream(proposed)
            self.redis.write_proposed_key(proposed)
            
            # Update metrics
            METRIC_PROPOSED.labels(reason=reason).inc()
            METRIC_MULTIPLIER.labels(symbol=symbol).set(mult)
            if eff:
                METRIC_EFF_CONF.labels(symbol=symbol).set(eff.confidence)
                METRIC_EFF_SCORE.labels(symbol=symbol).set(eff.score)
            
            # Log
            if reason == "ok":
                logger.info(
                    f"{symbol}: base=${target.target_usd:.2f} eff={eff.score:.3f} "
                    f"conf={eff.confidence:.2f} mult={mult:.3f} → proposed=${proposed_target:.2f}"
                )
            else:
                logger.info(
                    f"{symbol}: base=${target.target_usd:.2f} mult=1.000 reason={reason} "
                    f"→ proposed=${proposed_target:.2f}"
                )
        
        except Exception as e:
            logger.error(f"{symbol}: Error processing: {e}")
            METRIC_PROPOSED.labels(reason="redis_error").inc()
    
    def run_loop(self):
        """Main processing loop"""
        logger.info("Starting allocation shadow proposer loop...")
        
        while True:
            try:
                loop_start = time.time()
                now_ts = int(loop_start)
                
                # Get active symbols from existing allocation targets
                symbols = self.redis.get_active_symbols()
                
                if not symbols:
                    logger.debug("No active symbols found")
                else:
                    logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
                    
                    for symbol in symbols:
                        self.process_symbol(symbol, now_ts)
                
                METRIC_LOOPS.inc()
                
                # Sleep until next interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, LOOP_INTERVAL_SEC - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(5)

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("P3.1 Integration - Allocation Target Shadow Proposer")
    logger.info("=" * 60)
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    logger.info(f"Loop interval: {LOOP_INTERVAL_SEC}s")
    logger.info(f"Metrics port: {METRICS_PORT}")
    logger.info(f"Output stream: {PROPOSE_STREAM_NAME}")
    logger.info(f"P3.1 MIN_CONF: {P31_MIN_CONF}")
    logger.info(f"P3.1 STALE_SEC: {P31_STALE_SEC}")
    logger.info(f"Multiplier range: [{P31_MIN_MULT}, {P31_MAX_MULT}]")
    logger.info("=" * 60)
    
    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    logger.info(f"Prometheus metrics: http://localhost:{METRICS_PORT}/metrics")
    
    # Initialize Redis
    redis_client = RedisClient()
    
    # Create proposer and run
    proposer = AllocationShadowProposer(redis_client)
    proposer.run_loop()

if __name__ == "__main__":
    main()
