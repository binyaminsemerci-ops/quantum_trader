#!/usr/bin/env python3
"""
P2.9 Capital Allocation Brain (Fund-Grade)

Dynamically allocates portfolio budget to per-symbol & per-cluster exposure targets
using regime, cluster performance, drawdown zones, and risk state.

Architecture:
- Port: 8059
- Mode: shadow | enforce (P29_MODE)
- Loop: every 5s
- Fail-safe: missing data → pass-through (no change)

Author: Quantum Trading OS
Date: 2026-01-28
"""

import os
import sys
import time
import math
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# ============================================================================
# CONFIGURATION
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

P29_MODE = os.getenv("P29_MODE", "shadow")  # shadow | enforce
P29_INTERVAL_SEC = int(os.getenv("P29_INTERVAL_SEC", "5"))
P29_METRICS_PORT = int(os.getenv("P29_METRICS_PORT", "8059"))

# Safety caps
MAX_SYMBOL_PCT = float(os.getenv("MAX_SYMBOL_PCT", "0.40"))  # 40% max per symbol
MAX_CLUSTER_PCT = float(os.getenv("MAX_CLUSTER_PCT", "0.60"))  # 60% max per cluster

# Stale data threshold
STALE_DATA_SEC = int(os.getenv("STALE_DATA_SEC", "60"))

# Regime factors
REGIME_FACTORS = {
    "TREND": float(os.getenv("REGIME_FACTOR_TREND", "1.2")),
    "MEAN_REVERSION": float(os.getenv("REGIME_FACTOR_MR", "1.0")),
    "CHOP": float(os.getenv("REGIME_FACTOR_CHOP", "0.6")),
}

# Drawdown factors
DD_LOW_THRESHOLD = float(os.getenv("DD_LOW_THRESHOLD", "0.05"))  # 5%
DD_HIGH_THRESHOLD = float(os.getenv("DD_HIGH_THRESHOLD", "0.15"))  # 15%
DD_FACTOR_LOW = float(os.getenv("DD_FACTOR_LOW", "1.0"))
DD_FACTOR_MID = float(os.getenv("DD_FACTOR_MID", "0.7"))
DD_FACTOR_HIGH = float(os.getenv("DD_FACTOR_HIGH", "0.4"))

# Cluster stress clamp
CLUSTER_STRESS_MIN = float(os.getenv("CLUSTER_STRESS_MIN", "0.3"))
CLUSTER_STRESS_MAX = float(os.getenv("CLUSTER_STRESS_MAX", "1.0"))

# Target TTL
TARGET_TTL_SEC = int(os.getenv("TARGET_TTL_SEC", "300"))

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

metrics_targets_computed = Counter(
    "p29_targets_computed_total",
    "Total allocation targets computed",
    ["symbol"],
)

metrics_shadow_pass = Counter(
    "p29_shadow_pass_total",
    "Total shadow mode passes (no override)",
)

metrics_enforce_overrides = Counter(
    "p29_enforce_overrides_total",
    "Total enforce mode overrides",
    ["symbol"],
)

metrics_stale_fallback = Counter(
    "p29_stale_fallback_total",
    "Total stale data fallbacks",
    ["data_source"],
)

metrics_allocation_confidence = Gauge(
    "p29_allocation_confidence",
    "Allocation confidence score",
    ["symbol"],
)

metrics_target_usd = Gauge(
    "p29_target_usd",
    "Computed allocation target USD",
    ["symbol"],
)

metrics_regime_factor = Gauge(
    "p29_regime_factor",
    "Current regime factor applied",
)

metrics_cluster_factor = Gauge(
    "p29_cluster_factor",
    "Current cluster factor applied",
    ["cluster"],
)

metrics_drawdown_factor = Gauge(
    "p29_drawdown_factor",
    "Current drawdown factor applied",
)

metrics_loop_duration = Histogram(
    "p29_loop_duration_seconds",
    "Duration of allocation loop",
)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PortfolioState:
    equity_usd: float
    drawdown_pct: float
    timestamp: int
    source: str


@dataclass
class BudgetData:
    symbol: str
    budget_usd: float
    equity_usd: float
    mode: str
    timestamp: int


@dataclass
class ClusterStress:
    cluster_id: str
    stress: float
    alpha: float  # performance metric
    timestamp: int


@dataclass
class RegimeState:
    regime: str
    confidence: float
    timestamp: int


@dataclass
class AllocationTarget:
    symbol: str
    target_usd: float
    weight: float
    cluster_id: str
    regime: str
    drawdown_zone: str
    confidence: float
    timestamp: int
    mode: str
    
    def to_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format."""
        return {
            "target_usd": str(self.target_usd),
            "weight": str(self.weight),
            "cluster_id": self.cluster_id,
            "regime": self.regime,
            "drawdown_zone": self.drawdown_zone,
            "confidence": str(self.confidence),
            "timestamp": str(self.timestamp),
            "mode": self.mode,
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
    
    def get_portfolio_state(self) -> Optional[PortfolioState]:
        """Fetch portfolio state from quantum:state:portfolio."""
        try:
            data = self.client.hgetall("quantum:state:portfolio")
            if not data:
                return None
            
            return PortfolioState(
                equity_usd=float(data.get("equity_usd", 0)),
                drawdown_pct=abs(float(data.get("drawdown", 0))),
                timestamp=int(data.get("timestamp", 0)),
                source=data.get("source", "unknown"),
            )
        except Exception as e:
            logger.error(f"Error fetching portfolio state: {e}")
            return None
    
    def get_budget_data(self, symbol: str) -> Optional[BudgetData]:
        """Fetch P2.8 budget for symbol."""
        try:
            key = f"quantum:portfolio:budget:{symbol}"
            data = self.client.hgetall(key)
            if not data:
                return None
            
            return BudgetData(
                symbol=symbol,
                budget_usd=float(data.get("budget_usd", 0)),
                equity_usd=float(data.get("equity_usd", 0)),
                mode=data.get("mode", "shadow"),
                timestamp=int(data.get("timestamp", 0)),
            )
        except Exception as e:
            logger.error(f"Error fetching budget for {symbol}: {e}")
            return None
    
    def get_all_budget_symbols(self) -> List[str]:
        """Get all symbols with budget hashes."""
        try:
            keys = self.client.keys("quantum:portfolio:budget:*")
            symbols = [k.split(":")[-1] for k in keys]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching budget symbols: {e}")
            return []
    
    def get_cluster_stress(self, cluster_id: str) -> Optional[ClusterStress]:
        """Fetch cluster stress from quantum:cluster:stress:{cluster}."""
        try:
            key = f"quantum:cluster:stress:{cluster_id}"
            data = self.client.hgetall(key)
            if not data:
                return None
            
            return ClusterStress(
                cluster_id=cluster_id,
                stress=float(data.get("stress", 0)),
                alpha=float(data.get("alpha", 0)),
                timestamp=int(data.get("timestamp", 0)),
            )
        except Exception as e:
            logger.error(f"Error fetching cluster stress for {cluster_id}: {e}")
            return None
    
    def get_regime_state(self) -> Optional[RegimeState]:
        """Fetch latest regime from quantum:stream:regime.state."""
        try:
            # Read last entry from stream
            entries = self.client.xrevrange("quantum:stream:regime.state", count=1)
            if not entries:
                return None
            
            entry_id, data = entries[0]
            return RegimeState(
                regime=data.get("regime", "CHOP"),
                confidence=float(data.get("confidence", 0)),
                timestamp=int(data.get("timestamp", 0)),
            )
        except Exception as e:
            logger.error(f"Error fetching regime state: {e}")
            return None
    
    def write_allocation_target(self, target: AllocationTarget):
        """Write allocation target to quantum:allocation:target:{symbol}."""
        try:
            key = f"quantum:allocation:target:{target.symbol}"
            self.client.hset(key, mapping=target.to_hash())
            self.client.expire(key, TARGET_TTL_SEC)
            logger.debug(f"Wrote allocation target for {target.symbol}: ${target.target_usd:.2f}")
        except Exception as e:
            logger.error(f"Error writing allocation target for {target.symbol}: {e}")
    
    def stream_allocation_decision(self, target: AllocationTarget, factors: Dict):
        """Stream allocation decision to quantum:stream:allocation.decision."""
        try:
            event = {
                "symbol": target.symbol,
                "target_usd": str(target.target_usd),
                "weight": str(target.weight),
                "cluster_id": target.cluster_id,
                "regime": target.regime,
                "drawdown_zone": target.drawdown_zone,
                "confidence": str(target.confidence),
                "mode": target.mode,
                "timestamp": str(target.timestamp),
                # Factors
                "regime_factor": str(factors.get("regime_factor", 1.0)),
                "cluster_factor": str(factors.get("cluster_factor", 1.0)),
                "drawdown_factor": str(factors.get("drawdown_factor", 1.0)),
                "performance_factor": str(factors.get("performance_factor", 1.0)),
            }
            self.client.xadd("quantum:stream:allocation.decision", event, maxlen=1000)
        except Exception as e:
            logger.error(f"Error streaming allocation decision: {e}")


# ============================================================================
# ALLOCATION ENGINE
# ============================================================================

class AllocationEngine:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.mode = P29_MODE
        logger.info(f"Allocation engine initialized: mode={self.mode}")
    
    def is_data_stale(self, timestamp: int) -> bool:
        """Check if data timestamp is stale."""
        now = int(time.time())
        age = now - timestamp
        return age > STALE_DATA_SEC
    
    def get_regime_factor(self, regime: str) -> float:
        """Get regime multiplier."""
        return REGIME_FACTORS.get(regime, 1.0)
    
    def get_cluster_factor(self, stress: float) -> float:
        """Get cluster factor: clamp(1 - stress, 0.3, 1.0)."""
        factor = 1.0 - stress
        return max(CLUSTER_STRESS_MIN, min(CLUSTER_STRESS_MAX, factor))
    
    def get_drawdown_factor(self, drawdown_pct: float) -> Tuple[float, str]:
        """Get drawdown factor and zone."""
        if drawdown_pct < DD_LOW_THRESHOLD:
            return DD_FACTOR_LOW, "LOW"
        elif drawdown_pct < DD_HIGH_THRESHOLD:
            return DD_FACTOR_MID, "MID"
        else:
            return DD_FACTOR_HIGH, "HIGH"
    
    def get_performance_factor(self, alpha: float) -> float:
        """Get performance factor: sigmoid(alpha)."""
        # Sigmoid: 1 / (1 + exp(-alpha))
        try:
            return 1.0 / (1.0 + math.exp(-alpha))
        except OverflowError:
            return 1.0 if alpha > 0 else 0.0
    
    def compute_allocation(
        self,
        budget: BudgetData,
        portfolio: PortfolioState,
        cluster_stress: Optional[ClusterStress],
        regime: Optional[RegimeState],
    ) -> Optional[Tuple[AllocationTarget, Dict]]:
        """
        Compute allocation target for symbol.
        
        Returns: (AllocationTarget, factors_dict) or None on failure
        """
        now = int(time.time())
        
        # Stale data checks
        if self.is_data_stale(budget.timestamp):
            logger.warning(f"{budget.symbol}: Budget data stale, using base budget")
            metrics_stale_fallback.labels(data_source="budget").inc()
            target_usd = budget.budget_usd
            regime_factor = 1.0
            cluster_factor = 1.0
            drawdown_factor = 1.0
            performance_factor = 1.0
            regime_name = "UNKNOWN"
            drawdown_zone = "UNKNOWN"
            cluster_id = "UNKNOWN"
            confidence = 0.5
        else:
            # Base budget
            base = budget.budget_usd
            
            # Regime factor
            if regime and not self.is_data_stale(regime.timestamp):
                regime_factor = self.get_regime_factor(regime.regime)
                regime_name = regime.regime
                regime_confidence = regime.confidence
            else:
                logger.warning(f"{budget.symbol}: Regime data stale, using default")
                metrics_stale_fallback.labels(data_source="regime").inc()
                regime_factor = 1.0
                regime_name = "UNKNOWN"
                regime_confidence = 0.5
            
            # Cluster factor
            if cluster_stress and not self.is_data_stale(cluster_stress.timestamp):
                cluster_factor = self.get_cluster_factor(cluster_stress.stress)
                performance_factor = self.get_performance_factor(cluster_stress.alpha)
                cluster_id = cluster_stress.cluster_id
            else:
                logger.warning(f"{budget.symbol}: Cluster data stale, using default")
                metrics_stale_fallback.labels(data_source="cluster").inc()
                cluster_factor = 1.0
                performance_factor = 1.0
                cluster_id = "UNKNOWN"
            
            # Drawdown factor
            if not self.is_data_stale(portfolio.timestamp):
                drawdown_factor, drawdown_zone = self.get_drawdown_factor(portfolio.drawdown_pct)
            else:
                logger.warning(f"{budget.symbol}: Portfolio data stale, using default")
                metrics_stale_fallback.labels(data_source="portfolio").inc()
                drawdown_factor = 1.0
                drawdown_zone = "UNKNOWN"
            
            # Compute target
            target_usd = (
                base *
                regime_factor *
                cluster_factor *
                drawdown_factor *
                performance_factor
            )
            
            # Confidence (average of regime confidence and inverse cluster stress)
            if cluster_stress:
                confidence = (regime_confidence + (1.0 - cluster_stress.stress)) / 2.0
            else:
                confidence = regime_confidence
        
        # Safety cap: max per symbol
        max_symbol_usd = portfolio.equity_usd * MAX_SYMBOL_PCT
        if target_usd > max_symbol_usd:
            logger.warning(
                f"{budget.symbol}: Target ${target_usd:.2f} exceeds max ${max_symbol_usd:.2f}, capping"
            )
            target_usd = max_symbol_usd
        
        # Weight
        weight = target_usd / portfolio.equity_usd if portfolio.equity_usd > 0 else 0.0
        
        # Create target
        target = AllocationTarget(
            symbol=budget.symbol,
            target_usd=target_usd,
            weight=weight,
            cluster_id=cluster_id,
            regime=regime_name,
            drawdown_zone=drawdown_zone,
            confidence=confidence,
            timestamp=now,
            mode=self.mode,
        )
        
        factors = {
            "regime_factor": regime_factor,
            "cluster_factor": cluster_factor,
            "drawdown_factor": drawdown_factor,
            "performance_factor": performance_factor,
        }
        
        return target, factors
    
    def apply_cluster_caps(self, targets: List[AllocationTarget], portfolio: PortfolioState):
        """Apply cluster-level caps."""
        # Group by cluster
        cluster_totals = {}
        for target in targets:
            cid = target.cluster_id
            if cid not in cluster_totals:
                cluster_totals[cid] = []
            cluster_totals[cid].append(target)
        
        # Check caps
        for cluster_id, cluster_targets in cluster_totals.items():
            total_usd = sum(t.target_usd for t in cluster_targets)
            max_cluster_usd = portfolio.equity_usd * MAX_CLUSTER_PCT
            
            if total_usd > max_cluster_usd:
                logger.warning(
                    f"Cluster {cluster_id}: Total ${total_usd:.2f} exceeds max ${max_cluster_usd:.2f}, scaling down"
                )
                scale = max_cluster_usd / total_usd
                for target in cluster_targets:
                    target.target_usd *= scale
                    target.weight *= scale
    
    def run_loop(self):
        """Main allocation loop."""
        logger.info("Starting allocation loop...")
        
        while True:
            loop_start = time.time()
            
            try:
                # Fetch portfolio state
                portfolio = self.redis.get_portfolio_state()
                if not portfolio:
                    logger.warning("No portfolio state, skipping cycle")
                    time.sleep(P29_INTERVAL_SEC)
                    continue
                
                if self.is_data_stale(portfolio.timestamp):
                    logger.warning("Portfolio state stale, skipping cycle")
                    metrics_stale_fallback.labels(data_source="portfolio").inc()
                    time.sleep(P29_INTERVAL_SEC)
                    continue
                
                # Fetch regime
                regime = self.redis.get_regime_state()
                if regime:
                    metrics_regime_factor.set(self.get_regime_factor(regime.regime))
                
                # Fetch drawdown factor
                dd_factor, dd_zone = self.get_drawdown_factor(portfolio.drawdown_pct)
                metrics_drawdown_factor.set(dd_factor)
                
                # Get all symbols with budgets
                symbols = self.redis.get_all_budget_symbols()
                if not symbols:
                    logger.warning("No budget symbols found, skipping cycle")
                    time.sleep(P29_INTERVAL_SEC)
                    continue
                
                logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
                
                # Compute allocations
                targets = []
                for symbol in symbols:
                    # Fetch budget
                    budget = self.redis.get_budget_data(symbol)
                    if not budget:
                        logger.warning(f"{symbol}: No budget data, skipping")
                        continue
                    
                    # Fetch cluster stress (TODO: need symbol→cluster mapping)
                    # For now, assume cluster_id from budget or default
                    cluster_stress = None  # self.redis.get_cluster_stress(cluster_id)
                    
                    # Compute allocation
                    result = self.compute_allocation(budget, portfolio, cluster_stress, regime)
                    if not result:
                        continue
                    
                    target, factors = result
                    targets.append(target)
                    
                    # Metrics
                    metrics_targets_computed.labels(symbol=symbol).inc()
                    metrics_allocation_confidence.labels(symbol=symbol).set(target.confidence)
                    metrics_target_usd.labels(symbol=symbol).set(target.target_usd)
                    
                    logger.info(
                        f"{symbol}: target=${target.target_usd:.2f} "
                        f"weight={target.weight:.4f} "
                        f"regime={target.regime} "
                        f"dd_zone={target.drawdown_zone} "
                        f"conf={target.confidence:.3f}"
                    )
                
                # Apply cluster caps
                self.apply_cluster_caps(targets, portfolio)
                
                # Write targets
                for target in targets:
                    if self.mode == "shadow":
                        # Shadow mode: log only, don't write
                        logger.info(f"[SHADOW] {target.symbol}: Would write target=${target.target_usd:.2f}")
                        metrics_shadow_pass.inc()
                    else:
                        # Enforce mode: write to Redis
                        self.redis.write_allocation_target(target)
                        metrics_enforce_overrides.labels(symbol=target.symbol).inc()
                    
                    # Stream decision (both modes)
                    factors = {
                        "regime_factor": self.get_regime_factor(target.regime) if regime else 1.0,
                        "cluster_factor": 1.0,  # TODO: fetch from target metadata
                        "drawdown_factor": dd_factor,
                        "performance_factor": 1.0,
                    }
                    self.redis.stream_allocation_decision(target, factors)
                
                # Loop duration
                loop_duration = time.time() - loop_start
                metrics_loop_duration.observe(loop_duration)
                
                logger.info(f"Allocation cycle complete: {len(targets)} targets in {loop_duration:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in allocation loop: {e}", exc_info=True)
            
            # Sleep
            time.sleep(P29_INTERVAL_SEC)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=== P2.9 Capital Allocation Brain ===")
    logger.info(f"Mode: {P29_MODE}")
    logger.info(f"Interval: {P29_INTERVAL_SEC}s")
    logger.info(f"Metrics port: {P29_METRICS_PORT}")
    
    # Start Prometheus metrics server
    try:
        start_http_server(P29_METRICS_PORT)
        logger.info(f"Metrics server started on port {P29_METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        sys.exit(1)
    
    # Initialize Redis client
    redis_client = RedisClient()
    
    # Initialize allocation engine
    engine = AllocationEngine(redis_client)
    
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
