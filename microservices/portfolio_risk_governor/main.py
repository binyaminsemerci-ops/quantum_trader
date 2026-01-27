"""
P2.8 Portfolio Risk Governor
=============================
Fond-grade budget engine with stress-aware position sizing.

FORMULA:
  base_budget = equity * BASE_RISK_PCT
  stress = α*heat + β*cluster_stress + γ*vol_regime
  budget = clamp(base_budget * (1 - stress), MIN, MAX)
  over_budget = max(0, position_notional - budget)

MODES:
  - SHADOW: Log violations, don't block
  - ENFORCE: Block permits via Governor integration

OUTPUTS:
  - quantum:portfolio:budget:{symbol} (hash)
  - quantum:portfolio:budget:cluster:{cluster_id} (hash)
  - budget.violation events

Redis Inputs:
  - quantum:state:portfolio (equity_usd, drawdown)
  - quantum:cluster:stress
  - quantum:state:market:{symbol}
  - Prometheus /metrics from Heat Gate (p26_heat_value)
"""

import os
import sys
import time
import logging
import asyncio
import json
from typing import Dict, Optional, Any
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import httpx

# ============================================================================
# CONFIG
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PORT = int(os.getenv("PORT", "8049"))

P28_MODE = os.getenv("P28_MODE", "shadow")  # shadow|enforce
BASE_RISK_PCT = float(os.getenv("BASE_RISK_PCT", "0.02"))  # 2% of equity per position
ALPHA_HEAT = float(os.getenv("ALPHA_HEAT", "0.4"))
BETA_CLUSTER = float(os.getenv("BETA_CLUSTER", "0.4"))
GAMMA_VOL = float(os.getenv("GAMMA_VOL", "0.2"))
MIN_BUDGET_K = float(os.getenv("MIN_BUDGET_K", "500"))
MAX_BUDGET_K = float(os.getenv("MAX_BUDGET_K", "10000"))
STALE_SEC = int(os.getenv("STALE_SEC", "30"))
MAX_LKG_AGE_SEC = int(os.getenv("MAX_LKG_AGE_SEC", "900"))  # 15min - max age for last-known-good cache
BUDGET_HASH_TTL_SEC = int(os.getenv("BUDGET_HASH_TTL_SEC", "300"))  # 5min - budget hash expiry
P28_TEST_MODE = os.getenv("P28_TEST_MODE", "0") == "1"  # Enable test injection endpoint

HEAT_GATE_METRICS_URL = os.getenv("HEAT_GATE_METRICS_URL", "http://localhost:8056/metrics")
BUDGET_COMPUTE_INTERVAL_SEC = int(os.getenv("BUDGET_COMPUTE_INTERVAL_SEC", "10"))

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("p2.8-portfolio-risk-governor")

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

metric_enforce_mode = Gauge(
    "p28_enforce_mode",
    "P2.8 mode: 0=shadow, 1=enforce"
)
metric_enforce_mode.set(1.0 if P28_MODE == "enforce" else 0.0)

metric_budget_computed = Counter(
    "p28_budget_computed_total",
    "Total budget computations",
    ["symbol"]
)

metric_budget_blocks = Counter(
    "p28_budget_blocks_total",
    "Total budget violations blocked in enforce mode",
    ["symbol"]
)

metric_budget_allow = Counter(
    "p28_budget_allow_total",
    "Total budget checks passed",
    ["symbol"]
)

metric_stale_input = Counter(
    "p28_stale_input_total",
    "Total stale input failures (fail-open)",
    ["input_type"]
)

metric_redis_write_fail = Counter(
    "p28_redis_write_fail_total",
    "Total Redis write failures"
)

metric_budget_value = Gauge(
    "p28_budget_value_usd",
    "Current budget value in USD",
    ["symbol"]
)

metric_stress_factor = Gauge(
    "p28_stress_factor",
    "Current stress factor (0-1)",
    ["symbol"]
)

metric_lkg_cache_used = Counter(
    "p28_lkg_cache_used_total",
    "Portfolio state served from last-known-good cache"
)

metric_portfolio_too_old = Counter(
    "p28_portfolio_too_old_total",
    "Portfolio state older than MAX_LKG_AGE_SEC (fail-open)"
)

# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(title="P2.8 Portfolio Risk Governor")
redis_client: Optional[aioredis.Redis] = None

# Last-known-good portfolio state cache
lkg_portfolio_state: Optional[Dict[str, Any]] = None
lkg_portfolio_timestamp: int = 0

# ============================================================================
# BUDGET ENGINE
# ============================================================================

def compute_stress_factor(
    portfolio_heat: float,
    cluster_stress: float,
    vol_regime: float
) -> float:
    """
    Compute composite stress factor.
    
    stress = α*heat + β*cluster_stress + γ*vol_regime
    
    Returns value in [0, 1]
    """
    stress = (
        ALPHA_HEAT * portfolio_heat +
        BETA_CLUSTER * cluster_stress +
        GAMMA_VOL * vol_regime
    )
    return max(0.0, min(1.0, stress))


def compute_budget(equity_usd: float, stress: float) -> float:
    """
    Compute position budget in USD.
    
    budget = clamp(base_budget * (1 - stress), MIN, MAX)
    """
    base_budget = equity_usd * BASE_RISK_PCT
    adjusted_budget = base_budget * (1.0 - stress)
    return max(MIN_BUDGET_K, min(MAX_BUDGET_K, adjusted_budget))


async def fetch_portfolio_heat() -> Optional[float]:
    """
    Fetch portfolio heat from Heat Gate metrics endpoint.
    Parse p26_heat_value gauge.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(HEAT_GATE_METRICS_URL)
            if resp.status_code != 200:
                return None
            
            # Parse Prometheus exposition format
            for line in resp.text.split('\n'):
                if line.startswith('p26_heat_value'):
                    # p26_heat_value 0.425
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
            return None
    except Exception as e:
        logger.warning(f"Failed to fetch portfolio heat: {e}")
        return None


async def fetch_portfolio_state() -> Optional[Dict[str, Any]]:
    """
    Fetch quantum:state:portfolio hash with last-known-good (LKG) cache fallback.
    
    Strategy:
    1. Try to fetch fresh portfolio state from Redis
    2. If found and fresh (<30s), update LKG cache and return
    3. If stale but LKG cache is valid (<15min), use cached value
    4. If both too old, fail-open (return None)
    
    Returns:
      {
        "equity_usd": float,
        "drawdown": float,
        "timestamp": int
      }
    """
    global lkg_portfolio_state, lkg_portfolio_timestamp
    
    try:
        # Attempt to fetch fresh state
        data = await redis_client.hgetall("quantum:state:portfolio")
        
        if data:
            decoded = {k.decode(): v.decode() for k, v in data.items()}
            
            state = {
                "equity_usd": float(decoded.get("equity_usd", 0)),
                "drawdown": float(decoded.get("drawdown", 0)),
                "timestamp": int(decoded.get("timestamp", 0))
            }
            
            # If fresh, update LKG cache
            age = time.time() - state["timestamp"]
            if age < STALE_SEC:
                lkg_portfolio_state = state
                lkg_portfolio_timestamp = int(time.time())
                logger.debug(f"Portfolio state fresh (age={age:.0f}s), updated LKG cache")
                return state
            else:
                logger.debug(f"Portfolio state stale (age={age:.0f}s), checking LKG cache")
        
        # Fresh state not available or stale - check LKG cache
        if lkg_portfolio_state is not None:
            cache_age = time.time() - lkg_portfolio_timestamp
            
            if cache_age < MAX_LKG_AGE_SEC:
                logger.info(f"Using last-known-good portfolio state (cache_age={cache_age:.0f}s)")
                metric_lkg_cache_used.inc()
                return lkg_portfolio_state
            else:
                logger.warning(f"LKG cache too old ({cache_age:.0f}s > {MAX_LKG_AGE_SEC}s), fail-open")
                metric_portfolio_too_old.inc()
                return None
        
        # No fresh state and no LKG cache
        logger.warning("No portfolio state available (neither fresh nor cached)")
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch portfolio state: {e}", exc_info=True)
        
        # On error, try LKG cache
        if lkg_portfolio_state is not None:
            cache_age = time.time() - lkg_portfolio_timestamp
            if cache_age < MAX_LKG_AGE_SEC:
                logger.warning(f"Error fetching portfolio, using LKG cache (age={cache_age:.0f}s)")
                metric_lkg_cache_used.inc()
                return lkg_portfolio_state
        
        return None


async def fetch_cluster_stress(cluster_id: str = "default") -> float:
    """
    Fetch quantum:cluster:stress hash.
    Returns stress value [0, 1].
    """
    try:
        key = f"quantum:cluster:stress:{cluster_id}"
        data = await redis_client.hgetall(key)
        if not data:
            return 0.0
        
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        return float(decoded.get("stress", 0.0))
    except Exception as e:
        logger.warning(f"Failed to fetch cluster stress: {e}")
        return 0.0


async def fetch_vol_regime(symbol: str) -> float:
    """
    Fetch volatility regime from quantum:state:market:{symbol}.
    
    Returns normalized value [0, 1]:
      - LOW_VOL: 0.0
      - NORMAL_VOL: 0.33
      - HIGH_VOL: 0.67
      - EXTREME_VOL: 1.0
    """
    try:
        key = f"quantum:state:market:{symbol}"
        data = await redis_client.hgetall(key)
        if not data:
            return 0.33  # Default to NORMAL
        
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        
        # Try vol_regime field first
        vol_regime = decoded.get("vol_regime", "").upper()
        if "EXTREME" in vol_regime:
            return 1.0
        elif "HIGH" in vol_regime:
            return 0.67
        elif "LOW" in vol_regime:
            return 0.0
        
        # Fallback: use sigma
        sigma = float(decoded.get("sigma", 0.01))
        if sigma > 0.03:
            return 1.0
        elif sigma > 0.02:
            return 0.67
        elif sigma < 0.005:
            return 0.0
        else:
            return 0.33
    except Exception as e:
        logger.warning(f"Failed to fetch vol regime for {symbol}: {e}")
        return 0.33


def is_stale(timestamp: Optional[int]) -> bool:
    """Check if timestamp is stale (>STALE_SEC old)."""
    if timestamp is None:
        return True
    now = int(time.time())
    return (now - timestamp) > STALE_SEC


async def compute_and_publish_budget(symbol: str) -> None:
    """
    Core budget computation and publishing logic.
    
    Steps:
      1. Fetch inputs (portfolio state, heat, cluster stress, vol regime)
      2. Compute stress factor
      3. Compute budget
      4. Write to quantum:portfolio:budget:{symbol} hash
      5. Update metrics
    """
    try:
        # 1. Fetch inputs
        portfolio_state = await fetch_portfolio_state()
        portfolio_heat = await fetch_portfolio_heat()
        cluster_stress = await fetch_cluster_stress()
        vol_regime = await fetch_vol_regime(symbol)
        
        # Stale input guard (fail-open)
        if portfolio_state is None:
            logger.warning(f"Missing portfolio state for {symbol}, fail-open")
            metric_stale_input.labels(input_type="portfolio_state").inc()
            return
        
        if is_stale(portfolio_state.get("timestamp")):
            logger.warning(f"Stale portfolio state for {symbol}, fail-open")
            metric_stale_input.labels(input_type="portfolio_state").inc()
            return
        
        if portfolio_heat is None:
            logger.warning(f"Missing portfolio heat for {symbol}, using 0.0")
            portfolio_heat = 0.0
        
        equity_usd = portfolio_state["equity_usd"]
        
        # 2. Compute stress
        stress = compute_stress_factor(portfolio_heat, cluster_stress, vol_regime)
        
        # 3. Compute budget
        budget_usd = compute_budget(equity_usd, stress)
        
        # 4. Write to Redis
        hash_key = f"quantum:portfolio:budget:{symbol}"
        hash_data = {
            "symbol": symbol,
            "budget_usd": str(budget_usd),
            "stress_factor": str(stress),
            "equity_usd": str(equity_usd),
            "portfolio_heat": str(portfolio_heat),
            "cluster_stress": str(cluster_stress),
            "vol_regime": str(vol_regime),
            "mode": P28_MODE,
            "timestamp": str(int(time.time())),
            "base_risk_pct": str(BASE_RISK_PCT)
        }
        
        await redis_client.hset(hash_key, mapping=hash_data)
        await redis_client.expire(hash_key, BUDGET_HASH_TTL_SEC)  # TTL 5min (300s)
        
        # 5. Update metrics
        metric_budget_computed.labels(symbol=symbol).inc()
        metric_budget_value.labels(symbol=symbol).set(budget_usd)
        metric_stress_factor.labels(symbol=symbol).set(stress)
        
        logger.info(
            f"Budget computed for {symbol}: "
            f"equity={equity_usd:.0f} stress={stress:.3f} budget={budget_usd:.0f} USD"
        )
        
    except Exception as e:
        logger.error(f"Failed to compute budget for {symbol}: {e}", exc_info=True)
        metric_redis_write_fail.inc()


async def check_budget_violation(symbol: str, position_notional: float) -> bool:
    """
    Check if position_notional exceeds budget.
    
    Returns:
      True if VIOLATION (over budget)
      False if OK
    """
    try:
        hash_key = f"quantum:portfolio:budget:{symbol}"
        data = await redis_client.hgetall(hash_key)
        
        if not data:
            # No budget data = fail-open (allow)
            logger.warning(f"No budget data for {symbol}, fail-open")
            return False
        
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        
        # Check stale
        timestamp = int(decoded.get("timestamp", 0))
        if is_stale(timestamp):
            logger.warning(f"Stale budget data for {symbol}, fail-open")
            metric_stale_input.labels(input_type="budget_hash").inc()
            return False
        
        budget_usd = float(decoded.get("budget_usd", MAX_BUDGET_K))
        over_budget = max(0.0, position_notional - budget_usd)
        
        if over_budget > 0:
            logger.warning(
                f"Budget violation for {symbol}: "
                f"notional={position_notional:.0f} budget={budget_usd:.0f} over={over_budget:.0f}"
            )
            
            # Publish violation event
            event = {
                "event_type": "budget.violation",
                "symbol": symbol,
                "position_notional": position_notional,
                "budget_usd": budget_usd,
                "over_budget": over_budget,
                "mode": P28_MODE,
                "timestamp": int(time.time())
            }
            await redis_client.xadd(
                "quantum:stream:budget.violation",
                {"json": json.dumps(event)},
                maxlen=1000
            )
            
            metric_budget_blocks.labels(symbol=symbol).inc()
            return True
        else:
            metric_budget_allow.labels(symbol=symbol).inc()
            return False
            
    except Exception as e:
        logger.error(f"Failed to check budget violation for {symbol}: {e}")
        # Fail-open on errors
        return False


# ============================================================================
# BACKGROUND LOOP
# ============================================================================

async def budget_compute_loop():
    """
    Background loop that continuously computes budgets for active symbols.
    Runs every BUDGET_COMPUTE_INTERVAL_SEC.
    """
    logger.info("Starting budget compute loop")
    
    while True:
        try:
            await asyncio.sleep(BUDGET_COMPUTE_INTERVAL_SEC)
            
            # Get active symbols from quantum:position:snapshot hash keys
            keys = await redis_client.keys("quantum:position:snapshot:*")
            symbols = set()
            for key in keys:
                key_str = key.decode()
                parts = key_str.split(":")
                if len(parts) >= 4:
                    symbols.add(parts[3])
            
            if not symbols:
                logger.debug("No active symbols, skipping budget compute")
                continue
            
            logger.info(f"Computing budgets for {len(symbols)} symbols: {symbols}")
            
            # Compute budgets for all active symbols
            tasks = [compute_and_publish_budget(symbol) for symbol in symbols]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in budget compute loop: {e}", exc_info=True)
            await asyncio.sleep(5)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=False)
    logger.info(f"P2.8 Portfolio Risk Governor started (mode={P28_MODE}, port={PORT})")
    
    # Start background loop
    asyncio.create_task(budget_compute_loop())


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    logger.info("P2.8 Portfolio Risk Governor shutdown")


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        await redis_client.ping()
        return {
            "status": "healthy",
            "service": "p2.8-portfolio-risk-governor",
            "mode": P28_MODE,
            "redis": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {e}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/test/inject_portfolio_state")
async def inject_portfolio_state(equity_usd: float, drawdown: float = 0.0):
    """
    TEST-ONLY: Inject portfolio state for testing.
    Only active when P28_TEST_MODE=1.
    """
    if not P28_TEST_MODE:
        raise HTTPException(status_code=403, detail="Test mode not enabled (P28_TEST_MODE=0)")
    
    try:
        key = "quantum:state:portfolio"
        data = {
            "equity_usd": str(equity_usd),
            "drawdown": str(drawdown),
            "timestamp": str(int(time.time()))
        }
        
        # Write to Redis with 2min TTL
        for field, value in data.items():
            await redis_client.hset(key, field, value)
        await redis_client.expire(key, 120)
        
        logger.info(f"TEST: Injected portfolio state equity={equity_usd} drawdown={drawdown}")
        
        return {
            "success": True,
            "injected": data,
            "message": "Portfolio state injected (TTL 120s)"
        }
    except Exception as e:
        logger.error(f"Failed to inject portfolio state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/budget/check")
async def check_budget(symbol: str, position_notional: float):
    """
    Check budget violation for a symbol.
    
    Used by Governor for permit checks.
    
    Returns:
      {
        "violation": bool,
        "budget_usd": float,
        "over_budget": float,
        "mode": str
      }
    """
    try:
        hash_key = f"quantum:portfolio:budget:{symbol}"
        data = await redis_client.hgetall(hash_key)
        
        if not data:
            return {
                "violation": False,
                "budget_usd": None,
                "over_budget": 0.0,
                "mode": P28_MODE,
                "reason": "no_budget_data_fail_open"
            }
        
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        budget_usd = float(decoded.get("budget_usd", MAX_BUDGET_K))
        over_budget = max(0.0, position_notional - budget_usd)
        violation = over_budget > 0
        
        if violation and P28_MODE == "enforce":
            metric_budget_blocks.labels(symbol=symbol).inc()
        else:
            metric_budget_allow.labels(symbol=symbol).inc()
        
        return {
            "violation": violation,
            "budget_usd": budget_usd,
            "over_budget": over_budget,
            "mode": P28_MODE,
            "enforce": P28_MODE == "enforce"
        }
        
    except Exception as e:
        logger.error(f"Budget check failed for {symbol}: {e}")
        return {
            "violation": False,
            "budget_usd": None,
            "over_budget": 0.0,
            "mode": P28_MODE,
            "reason": f"error_fail_open: {e}"
        }


@app.get("/budget/{symbol}")
async def get_budget(symbol: str):
    """Get current budget for a symbol."""
    try:
        hash_key = f"quantum:portfolio:budget:{symbol}"
        data = await redis_client.hgetall(hash_key)
        
        if not data:
            return {"error": "no_budget_data"}
        
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        return decoded
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
