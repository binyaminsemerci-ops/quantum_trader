#!/usr/bin/env python3
"""
P3.0 Performance Attribution Brain (Hedge Fund Grade)

Computes alpha attribution by breaking down P&L into regime, cluster, signal,
and time bucket contributions. Uses rolling EWMA for performance factor.

Architecture:
- Port: 8060
- Mode: shadow | enforce (P30_MODE)
- Loop: every 5s
- Fail-safe: missing data → LKG fallback (15min)

Author: Quantum Trading OS
Date: 2026-01-28
"""

import os
import sys
import time
import json
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

import redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# ============================================================================
# CONFIGURATION
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

P30_MODE = os.getenv("P30_MODE", "shadow")  # shadow | enforce
P30_INTERVAL_SEC = int(os.getenv("P30_INTERVAL_SEC", "5"))
P30_METRICS_PORT = int(os.getenv("P30_METRICS_PORT", "8060"))

# EWMA parameters
EWMA_ALPHA = float(os.getenv("EWMA_ALPHA", "0.3"))  # Smoothing factor
LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", "20"))  # Number of trades to analyze

# LKG (Last Known Good) settings
MAX_LKG_AGE_SEC = int(os.getenv("MAX_LKG_AGE_SEC", "900"))  # 15 minutes

# Attribution TTL
ATTRIBUTION_TTL_SEC = int(os.getenv("ATTRIBUTION_TTL_SEC", "300"))  # 5 minutes

# Time buckets (hours)
TIME_BUCKETS = ["00-06", "06-12", "12-18", "18-24"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

metrics_attributions_computed = Counter(
    "p30_attributions_computed_total",
    "Total performance attributions computed",
    ["symbol"],
)

metrics_alpha_score = Gauge(
    "p30_alpha_score",
    "Alpha score for symbol",
    ["symbol"],
)

metrics_performance_factor = Gauge(
    "p30_performance_factor",
    "Performance factor (EWMA)",
    ["symbol"],
)

metrics_confidence = Gauge(
    "p30_confidence",
    "Attribution confidence",
    ["symbol"],
)

metrics_lkg_used = Counter(
    "p30_lkg_used_total",
    "LKG fallback used",
    ["symbol"],
)

metrics_execution_pnl = Counter(
    "p30_execution_pnl_total",
    "Total P&L from executions",
    ["symbol"],
)

metrics_loop_duration = Histogram(
    "p30_loop_duration_seconds",
    "Duration of attribution loop",
)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ExecutionResult:
    symbol: str
    realized_pnl: float
    timestamp: int
    regime: str
    cluster: str
    signal: str
    time_bucket: str


@dataclass
class AlphaAttribution:
    symbol: str
    alpha_score: float
    performance_factor: float
    confidence: float
    window: int
    ts_utc: int
    source: str  # "live" or "LKG"
    mode: str
    
    # Breakdown by dimension
    regime_contrib: Dict[str, float]
    cluster_contrib: Dict[str, float]
    signal_contrib: Dict[str, float]
    time_contrib: Dict[str, float]
    
    def to_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format."""
        return {
            "alpha_score": str(self.alpha_score),
            "performance_factor": str(self.performance_factor),
            "confidence": str(self.confidence),
            "window": str(self.window),
            "ts_utc": str(self.ts_utc),
            "source": self.source,
            "mode": self.mode,
            # Serialize breakdowns as JSON
            "regime_contrib": json.dumps(self.regime_contrib),
            "cluster_contrib": json.dumps(self.cluster_contrib),
            "signal_contrib": json.dumps(self.signal_contrib),
            "time_contrib": json.dumps(self.time_contrib),
        }


# ============================================================================
# REDIS CLIENT
# ============================================================================

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
        logger.info(f"Redis client initialized: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    
    def get_recent_executions(self, symbol: str, count: int = 50) -> List[ExecutionResult]:
        """Fetch recent execution results from stream."""
        try:
            # Read from quantum:stream:execution.result
            entries = self.client.xrevrange("quantum:stream:execution.result", count=count)
            
            executions = []
            for entry_id, data in entries:
                # Filter by symbol
                if data.get("symbol") != symbol:
                    continue
                
                # Parse execution
                realized_pnl = float(data.get("realized_pnl", 0))
                timestamp = int(data.get("timestamp", 0))
                
                # Get regime (from current state or default)
                regime = data.get("regime", "UNKNOWN")
                
                # Get cluster (from current state or default)
                cluster = data.get("cluster", "UNKNOWN")
                
                # Get signal (from execution metadata)
                signal = data.get("signal", "UNKNOWN")
                
                # Compute time bucket from timestamp
                hour = datetime.fromtimestamp(timestamp, tz=timezone.utc).hour
                time_bucket = self._get_time_bucket(hour)
                
                executions.append(ExecutionResult(
                    symbol=symbol,
                    realized_pnl=realized_pnl,
                    timestamp=timestamp,
                    regime=regime,
                    cluster=cluster,
                    signal=signal,
                    time_bucket=time_bucket,
                ))
            
            return executions
            
        except Exception as e:
            logger.error(f"Error fetching executions for {symbol}: {e}")
            return []
    
    def _get_time_bucket(self, hour: int) -> str:
        """Get time bucket for hour (0-23)."""
        if 0 <= hour < 6:
            return "00-06"
        elif 6 <= hour < 12:
            return "06-12"
        elif 12 <= hour < 18:
            return "12-18"
        else:
            return "18-24"
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols with recent executions."""
        try:
            # Read recent execution stream
            entries = self.client.xrevrange("quantum:stream:execution.result", count=100)
            
            symbols = set()
            for entry_id, data in entries:
                symbol = data.get("symbol")
                if symbol:
                    symbols.add(symbol)
            
            return list(symbols)
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def write_attribution(self, attribution: AlphaAttribution):
        """Write attribution to quantum:alpha:attribution:{symbol}."""
        try:
            key = f"quantum:alpha:attribution:{attribution.symbol}"
            self.client.hset(key, mapping=attribution.to_hash())
            self.client.expire(key, ATTRIBUTION_TTL_SEC)
            logger.debug(f"Wrote attribution for {attribution.symbol}: alpha={attribution.alpha_score:.4f}")
        except Exception as e:
            logger.error(f"Error writing attribution for {attribution.symbol}: {e}")
    
    def stream_attribution(self, attribution: AlphaAttribution):
        """Stream attribution to quantum:stream:alpha.attribution."""
        try:
            event = {
                "symbol": attribution.symbol,
                "alpha_score": str(attribution.alpha_score),
                "performance_factor": str(attribution.performance_factor),
                "confidence": str(attribution.confidence),
                "window": str(attribution.window),
                "ts_utc": str(attribution.ts_utc),
                "source": attribution.source,
                "mode": attribution.mode,
                # Breakdowns
                "regime_contrib": json.dumps(attribution.regime_contrib),
                "cluster_contrib": json.dumps(attribution.cluster_contrib),
                "signal_contrib": json.dumps(attribution.signal_contrib),
                "time_contrib": json.dumps(attribution.time_contrib),
            }
            self.client.xadd("quantum:stream:alpha.attribution", event, maxlen=1000)
        except Exception as e:
            logger.error(f"Error streaming attribution: {e}")


# ============================================================================
# ATTRIBUTION ENGINE
# ============================================================================

class AttributionEngine:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.mode = P30_MODE
        
        # LKG cache: symbol -> (attribution, timestamp)
        self.lkg_cache: Dict[str, Tuple[AlphaAttribution, int]] = {}
        
        # EWMA state: symbol -> performance_factor
        self.ewma_state: Dict[str, float] = {}
        
        logger.info(f"Attribution engine initialized: mode={self.mode}")
    
    def compute_attribution(self, symbol: str) -> Optional[AlphaAttribution]:
        """
        Compute performance attribution for symbol.
        
        Returns AlphaAttribution or None if insufficient data.
        """
        try:
            # Fetch recent executions
            executions = self.redis.get_recent_executions(symbol, count=LOOKBACK_WINDOW)
            
            if not executions:
                logger.debug(f"{symbol}: No recent executions, checking LKG")
                return self._get_lkg_attribution(symbol)
            
            # Compute total P&L
            total_pnl = sum(e.realized_pnl for e in executions)
            
            # Break down by dimensions
            regime_contrib = self._compute_dimension_contrib(executions, "regime")
            cluster_contrib = self._compute_dimension_contrib(executions, "cluster")
            signal_contrib = self._compute_dimension_contrib(executions, "signal")
            time_contrib = self._compute_dimension_contrib(executions, "time_bucket")
            
            # Compute alpha score (normalized P&L)
            alpha_score = self._compute_alpha_score(executions)
            
            # Update EWMA performance factor
            performance_factor = self._update_ewma(symbol, alpha_score)
            
            # Compute confidence (based on sample size)
            confidence = min(1.0, len(executions) / LOOKBACK_WINDOW)
            
            # Create attribution
            attribution = AlphaAttribution(
                symbol=symbol,
                alpha_score=alpha_score,
                performance_factor=performance_factor,
                confidence=confidence,
                window=len(executions),
                ts_utc=int(time.time()),
                source="live",
                mode=self.mode,
                regime_contrib=regime_contrib,
                cluster_contrib=cluster_contrib,
                signal_contrib=signal_contrib,
                time_contrib=time_contrib,
            )
            
            # Update LKG cache
            self.lkg_cache[symbol] = (attribution, int(time.time()))
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error computing attribution for {symbol}: {e}")
            return self._get_lkg_attribution(symbol)
    
    def _compute_dimension_contrib(self, executions: List[ExecutionResult], dimension: str) -> Dict[str, float]:
        """Compute contribution breakdown by dimension."""
        contrib = defaultdict(float)
        
        for execution in executions:
            if dimension == "regime":
                key = execution.regime
            elif dimension == "cluster":
                key = execution.cluster
            elif dimension == "signal":
                key = execution.signal
            elif dimension == "time_bucket":
                key = execution.time_bucket
            else:
                key = "UNKNOWN"
            
            contrib[key] += execution.realized_pnl
        
        # Convert to regular dict
        return dict(contrib)
    
    def _compute_alpha_score(self, executions: List[ExecutionResult]) -> float:
        """
        Compute alpha score from executions.
        
        Alpha score = normalized P&L with sigmoid transformation
        """
        if not executions:
            return 0.0
        
        # Total P&L
        total_pnl = sum(e.realized_pnl for e in executions)
        
        # Normalize by number of trades
        avg_pnl = total_pnl / len(executions)
        
        # Apply sigmoid transformation: 2 / (1 + exp(-x)) - 1
        # Maps [-inf, +inf] to [-1, +1]
        try:
            alpha_score = 2.0 / (1.0 + math.exp(-avg_pnl / 100.0)) - 1.0
        except OverflowError:
            alpha_score = 1.0 if avg_pnl > 0 else -1.0
        
        return alpha_score
    
    def _update_ewma(self, symbol: str, new_value: float) -> float:
        """
        Update EWMA performance factor.
        
        EWMA_t = alpha * value_t + (1 - alpha) * EWMA_t-1
        """
        if symbol not in self.ewma_state:
            # Initialize with first value
            self.ewma_state[symbol] = new_value
            return new_value
        
        # Update EWMA
        ewma = EWMA_ALPHA * new_value + (1 - EWMA_ALPHA) * self.ewma_state[symbol]
        self.ewma_state[symbol] = ewma
        
        return ewma
    
    def _get_lkg_attribution(self, symbol: str) -> Optional[AlphaAttribution]:
        """Get Last Known Good attribution if available and fresh."""
        if symbol not in self.lkg_cache:
            logger.warning(f"{symbol}: No LKG attribution available")
            return None
        
        attribution, timestamp = self.lkg_cache[symbol]
        age = int(time.time()) - timestamp
        
        if age > MAX_LKG_AGE_SEC:
            logger.warning(f"{symbol}: LKG attribution too old ({age}s), discarding")
            del self.lkg_cache[symbol]
            return None
        
        logger.info(f"{symbol}: Using LKG attribution (age={age}s)")
        metrics_lkg_used.labels(symbol=symbol).inc()
        
        # Update source and timestamp
        attribution.source = "LKG"
        attribution.ts_utc = int(time.time())
        
        return attribution
    
    def run_loop(self):
        """Main attribution loop."""
        logger.info("Starting attribution loop...")
        
        while True:
            loop_start = time.time()
            
            try:
                # Get all symbols with recent activity
                symbols = self.redis.get_all_symbols()
                
                if not symbols:
                    logger.debug("No symbols with recent executions")
                    time.sleep(P30_INTERVAL_SEC)
                    continue
                
                logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
                
                for symbol in symbols:
                    # Compute attribution
                    attribution = self.compute_attribution(symbol)
                    
                    if not attribution:
                        logger.warning(f"{symbol}: Could not compute attribution")
                        continue
                    
                    # Update metrics
                    metrics_attributions_computed.labels(symbol=symbol).inc()
                    metrics_alpha_score.labels(symbol=symbol).set(attribution.alpha_score)
                    metrics_performance_factor.labels(symbol=symbol).set(attribution.performance_factor)
                    metrics_confidence.labels(symbol=symbol).set(attribution.confidence)
                    
                    # Log summary
                    logger.info(
                        f"{symbol}: alpha={attribution.alpha_score:.4f} "
                        f"perf={attribution.performance_factor:.4f} "
                        f"conf={attribution.confidence:.3f} "
                        f"window={attribution.window} "
                        f"source={attribution.source}"
                    )
                    
                    if self.mode == "shadow":
                        # Shadow mode: log only
                        logger.info(f"[SHADOW] {symbol}: Would write attribution")
                    else:
                        # Enforce mode: write to Redis
                        self.redis.write_attribution(attribution)
                    
                    # Always stream (both modes)
                    self.redis.stream_attribution(attribution)
                
                # Loop duration
                loop_duration = time.time() - loop_start
                metrics_loop_duration.observe(loop_duration)
                
                logger.info(f"Attribution cycle complete: {len(symbols)} symbols in {loop_duration:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in attribution loop: {e}", exc_info=True)
            
            # Sleep
            time.sleep(P30_INTERVAL_SEC)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=== P3.0 Performance Attribution Brain ===")
    logger.info(f"Mode: {P30_MODE}")
    logger.info(f"Interval: {P30_INTERVAL_SEC}s")
    logger.info(f"Metrics port: {P30_METRICS_PORT}")
    logger.info(f"EWMA alpha: {EWMA_ALPHA}")
    logger.info(f"Lookback window: {LOOKBACK_WINDOW} trades")
    
    # Start Prometheus metrics server
    try:
        start_http_server(P30_METRICS_PORT)
        logger.info(f"Metrics server started on port {P30_METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        sys.exit(1)
    
    # Initialize Redis client
    redis_client = RedisClient()
    
    # Initialize attribution engine
    engine = AttributionEngine(redis_client)
    
    # Run main loop
    try:
        engine.run_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
