#!/usr/bin/env python3
"""
P3.1 Capital Efficiency Brain - Performance-Based Capital Allocation

Monitors position performance and computes efficiency scores to guide
capital reallocation decisions. Operates in shadow mode by default.

Efficiency Score Formula:
    efficiency_score = EWMA(
        performance_factor / 
        (1 + drawdown + volatility + holding_time_penalty)
    )

Inputs:
    - quantum:alpha:attribution:{symbol} - Alpha attribution data
    - quantum:allocation:target:{symbol} - Target allocations
    - execution:pnl:stream - Real-time PnL events

Outputs:
    - quantum:capital:efficiency:{symbol} - Efficiency metrics
      * efficiency_score (0..1)
      * capital_pressure (INCREASE|HOLD|DECREASE)
      * reallocation_weight
      * confidence

Endpoints:
    - GET /health - Health check
    - GET /metrics - Prometheus metrics
"""

import os
import sys
import time
import json
import redis
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# Efficiency parameters
EWMA_ALPHA = float(os.getenv('EFFICIENCY_EWMA_ALPHA', 0.1))
EFFICIENCY_THRESHOLD = float(os.getenv('EFFICIENCY_THRESHOLD', 0.3))
MIN_HOLDING_HOURS = float(os.getenv('MIN_HOLDING_HOURS', 24.0))
VOLATILITY_LOOKBACK = int(os.getenv('VOLATILITY_LOOKBACK', 100))

# Mode configuration
SHADOW_MODE = os.getenv('SHADOW_MODE', 'true').lower() == 'true'
ENABLE_REBALANCING = os.getenv('ENABLE_REBALANCING', 'false').lower() == 'true'

# FastAPI app
app = FastAPI(title="Capital Efficiency Brain", version="3.1.0")

# Prometheus metrics
efficiency_scores = Gauge('capital_efficiency_score', 'Current efficiency score', ['symbol'])
capital_pressure = Gauge('capital_pressure_signal', 'Capital pressure signal (-1=DECREASE, 0=HOLD, 1=INCREASE)', ['symbol'])
processing_rate = Counter('efficiency_events_processed', 'Total efficiency calculations')
efficiency_histogram = Histogram('efficiency_score_distribution', 'Distribution of efficiency scores')
shadow_mode_gauge = Gauge('capital_efficiency_shadow_mode', 'Shadow mode enabled (1=shadow, 0=active)')
active_symbols_gauge = Gauge('capital_efficiency_active_symbols', 'Number of symbols being tracked')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for a symbol"""
    symbol: str
    efficiency_score: float
    capital_pressure: str  # INCREASE, HOLD, DECREASE
    reallocation_weight: float
    confidence: float
    performance_factor: float
    drawdown: float
    volatility: float
    holding_time_penalty: float
    timestamp: str

class CapitalEfficiencyEngine:
    """Core capital efficiency calculation engine"""
    
    def __init__(self):
        self.efficiency_scores: Dict[str, float] = {}
        self.pnl_history: Dict[str, List[float]] = defaultdict(list)
        self.position_opened: Dict[str, datetime] = {}
        self.last_update: Dict[str, datetime] = {}
        
        logger.info(f"Capital Efficiency Engine initialized")
        logger.info(f"  EWMA Alpha: {EWMA_ALPHA}")
        logger.info(f"  Efficiency Threshold: {EFFICIENCY_THRESHOLD}")
        logger.info(f"  Min Holding Hours: {MIN_HOLDING_HOURS}")
        logger.info(f"  Shadow Mode: {SHADOW_MODE}")
        logger.info(f"  Rebalancing Enabled: {ENABLE_REBALANCING}")
        
        # Update shadow mode metric
        shadow_mode_gauge.set(1 if SHADOW_MODE else 0)
    
    def calculate_performance_factor(self, symbol: str) -> float:
        """Calculate performance factor from PnL history"""
        pnl_data = self.pnl_history.get(symbol, [])
        if not pnl_data:
            return 0.5  # Neutral for new positions
        
        # Positive PnL = higher performance
        recent_pnl = sum(pnl_data[-20:]) if len(pnl_data) >= 20 else sum(pnl_data)
        # Normalize to 0..1 range (sigmoid-like)
        return max(0.0, min(1.0, 0.5 + recent_pnl / 1000.0))
    
    def calculate_drawdown(self, symbol: str) -> float:
        """Calculate current drawdown percentage"""
        pnl_data = self.pnl_history.get(symbol, [])
        if len(pnl_data) < 2:
            return 0.0
        
        cumulative = [sum(pnl_data[:i+1]) for i in range(len(pnl_data))]
        peak = max(cumulative)
        current = cumulative[-1]
        
        if peak <= 0:
            return 0.5  # Significant penalty for never being profitable
        
        drawdown = (peak - current) / abs(peak)
        return max(0.0, min(1.0, drawdown))
    
    def calculate_volatility(self, symbol: str) -> float:
        """Calculate PnL volatility (normalized standard deviation)"""
        pnl_data = self.pnl_history.get(symbol, [])
        if len(pnl_data) < 5:
            return 0.0
        
        recent = pnl_data[-VOLATILITY_LOOKBACK:]
        mean_pnl = sum(recent) / len(recent)
        variance = sum((x - mean_pnl) ** 2 for x in recent) / len(recent)
        std_dev = variance ** 0.5
        
        # Normalize: higher volatility = higher penalty (0..1)
        return min(1.0, std_dev / 100.0)
    
    def calculate_holding_time_penalty(self, symbol: str) -> float:
        """Penalty for positions held too long without performance"""
        if symbol not in self.position_opened:
            return 0.0
        
        hours_held = (datetime.utcnow() - self.position_opened[symbol]).total_seconds() / 3600.0
        
        # No penalty for early hours, then linear increase
        if hours_held < MIN_HOLDING_HOURS:
            return 0.0
        
        excess_hours = hours_held - MIN_HOLDING_HOURS
        # Max penalty of 0.5 at 7 days excess
        return min(0.5, excess_hours / (7 * 24))
    
    def compute_efficiency_score(self, symbol: str) -> EfficiencyMetrics:
        """Compute comprehensive efficiency score using EWMA"""
        # Get components
        perf = self.calculate_performance_factor(symbol)
        dd = self.calculate_drawdown(symbol)
        vol = self.calculate_volatility(symbol)
        holding_penalty = self.calculate_holding_time_penalty(symbol)
        
        # Efficiency formula
        denominator = 1.0 + dd + vol + holding_penalty
        raw_score = perf / denominator
        
        # EWMA smoothing
        current_score = self.efficiency_scores.get(symbol, 0.5)
        new_score = EWMA_ALPHA * raw_score + (1 - EWMA_ALPHA) * current_score
        self.efficiency_scores[symbol] = new_score
        
        # Determine capital pressure
        if new_score > 0.7:
            pressure = "INCREASE"
        elif new_score < EFFICIENCY_THRESHOLD:
            pressure = "DECREASE"
        else:
            pressure = "HOLD"
        
        # Reallocation weight (how much to adjust capital)
        # High efficiency = increase weight, low = decrease
        realloc_weight = new_score  # 0..1 range
        
        # Confidence based on data availability
        pnl_count = len(self.pnl_history.get(symbol, []))
        confidence = min(1.0, pnl_count / 50.0)  # Full confidence at 50+ data points
        
        return EfficiencyMetrics(
            symbol=symbol,
            efficiency_score=new_score,
            capital_pressure=pressure,
            reallocation_weight=realloc_weight,
            confidence=confidence,
            performance_factor=perf,
            drawdown=dd,
            volatility=vol,
            holding_time_penalty=holding_penalty,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def update_from_pnl(self, symbol: str, pnl: float):
        """Update efficiency metrics from PnL event"""
        self.pnl_history[symbol].append(pnl)
        
        # Limit history size
        if len(self.pnl_history[symbol]) > 500:
            self.pnl_history[symbol] = self.pnl_history[symbol][-500:]
        
        # Track position open time
        if symbol not in self.position_opened:
            self.position_opened[symbol] = datetime.utcnow()
        
        self.last_update[symbol] = datetime.utcnow()
    
    def publish_efficiency_metrics(self, metrics: EfficiencyMetrics):
        """Publish efficiency metrics to Redis"""
        key = f"quantum:capital:efficiency:{metrics.symbol}"
        
        # Store as hash
        data = asdict(metrics)
        redis_client.hset(key, mapping=data)
        redis_client.expire(key, 3600)  # 1 hour TTL
        
        # Update Prometheus metrics
        efficiency_scores.labels(symbol=metrics.symbol).set(metrics.efficiency_score)
        pressure_value = {"INCREASE": 1, "HOLD": 0, "DECREASE": -1}[metrics.capital_pressure]
        capital_pressure.labels(symbol=metrics.symbol).set(pressure_value)
        efficiency_histogram.observe(metrics.efficiency_score)
        processing_rate.inc()
        
        mode = "SHADOW" if SHADOW_MODE else "ACTIVE"
        logger.info(
            f"[{mode}] {metrics.symbol} efficiency={metrics.efficiency_score:.3f} "
            f"pressure={metrics.capital_pressure} confidence={metrics.confidence:.2f} "
            f"(perf={metrics.performance_factor:.2f} dd={metrics.drawdown:.2f} "
            f"vol={metrics.volatility:.2f} hold_pen={metrics.holding_time_penalty:.2f})"
        )

# Global engine instance
engine = CapitalEfficiencyEngine()

def process_pnl_stream():
    """Process execution PnL stream from Redis"""
    logger.info("Starting PnL stream processor...")
    
    stream_key = "execution:pnl:stream"
    last_id = "0"
    
    while True:
        try:
            # Read from stream
            messages = redis_client.xread({stream_key: last_id}, count=10, block=1000)
            
            if not messages:
                # Update active symbols gauge periodically
                active_symbols_gauge.set(len(engine.efficiency_scores))
                continue
            
            for stream, entries in messages:
                for msg_id, data in entries:
                    last_id = msg_id
                    
                    symbol = data.get('symbol')
                    pnl = float(data.get('pnl', 0))
                    
                    if not symbol:
                        continue
                    
                    # Update efficiency tracking
                    engine.update_from_pnl(symbol, pnl)
                    
                    # Compute and publish efficiency score
                    metrics = engine.compute_efficiency_score(symbol)
                    engine.publish_efficiency_metrics(metrics)
                    
                    # In active mode, could trigger rebalancing here
                    if not SHADOW_MODE and ENABLE_REBALANCING:
                        if metrics.capital_pressure == "DECREASE" and metrics.confidence > 0.5:
                            logger.warning(
                                f"[ACTIVE] Would trigger capital reduction for {symbol} "
                                f"(efficiency={metrics.efficiency_score:.3f})"
                            )
        
        except redis.RedisError as e:
            logger.error(f"Redis error in PnL stream processor: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error processing PnL stream: {e}", exc_info=True)
            time.sleep(1)

# FastAPI endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        redis_client.ping()
        redis_ok = True
    except:
        redis_ok = False
    
    return {
        "status": "healthy" if redis_ok else "degraded",
        "service": "capital-efficiency",
        "version": "3.1.0",
        "shadow_mode": SHADOW_MODE,
        "redis": "connected" if redis_ok else "disconnected",
        "active_symbols": len(engine.efficiency_scores),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/efficiency/{symbol}")
async def get_efficiency(symbol: str):
    """Get efficiency metrics for a symbol"""
    key = f"quantum:capital:efficiency:{symbol}"
    data = redis_client.hgetall(key)
    
    if not data:
        return {"error": "Symbol not found"}
    
    return data

def main():
    """Main entry point"""
    import threading
    
    # Start PnL stream processor in background
    processor_thread = threading.Thread(target=process_pnl_stream, daemon=True)
    processor_thread.start()
    
    # Start FastAPI server
    logger.info("Starting Capital Efficiency Brain API server on port 8026...")
    uvicorn.run(app, host="0.0.0.0", port=8026, log_level="info")

if __name__ == "__main__":
    main()
