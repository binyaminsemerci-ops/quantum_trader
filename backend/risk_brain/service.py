"""Risk Brain Service - Position sizing and risk assessment.

Phase 2.2: RL Integration - Applies policy adjustments from quantum:ai_policy_adjustment
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging
import redis
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Risk Brain Service", version="1.1.0")

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# RL adjustment configuration
MAX_LEVERAGE = 10.0  # Global leverage cap
RL_ADJUSTMENT_STALENESS_SEC = 300  # 5 minutes
RL_DELTA_MIN = -0.5
RL_DELTA_MAX = 0.5
RL_LEVERAGE_MULTIPLIER_MIN = 0.5  # clamp(1 + delta) lower bound
RL_LEVERAGE_MULTIPLIER_MAX = 1.5  # clamp(1 + delta) upper bound


class RiskRequest(BaseModel):
    """Request model for risk evaluation."""
    signal: Dict[str, Any]
    operating_mode: str  # EXPANSION/PRESERVATION/EMERGENCY


def get_rl_policy_adjustment(symbol: str) -> Optional[Dict[str, Any]]:
    """Read RL policy adjustment from Redis.
    
    Returns:
        dict with {delta, reward, symbol, timestamp} if valid and recent
        None if missing, stale, or invalid
    """
    try:
        data = redis_client.hgetall("quantum:ai_policy_adjustment")
        
        if not data:
            logger.debug("[RL] No policy adjustment found in Redis")
            return None
        
        # Parse delta
        try:
            delta = float(data.get("delta", "0"))
        except (ValueError, TypeError):
            logger.warning(f"[RL] Invalid delta value: {data.get('delta')}")
            return None
        
        # Validate delta range
        if not (RL_DELTA_MIN <= delta <= RL_DELTA_MAX):
            logger.warning(f"[RL] Delta {delta:.3f} out of bounds [{RL_DELTA_MIN}, {RL_DELTA_MAX}]")
            return None
        
        # Parse timestamp (support ISO format and epoch)
        timestamp_str = data.get("timestamp", "")
        try:
            # Try ISO format first
            if "T" in timestamp_str or "-" in timestamp_str:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                # Try epoch
                ts = datetime.fromtimestamp(float(timestamp_str), tz=timezone.utc)
        except Exception as e:
            logger.warning(f"[RL] Invalid timestamp: {timestamp_str} - {e}")
            return None
        
        # Check staleness
        age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
        if age_sec > RL_ADJUSTMENT_STALENESS_SEC:
            logger.debug(f"[RL] Adjustment stale: {age_sec:.0f}s old (limit: {RL_ADJUSTMENT_STALENESS_SEC}s)")
            return None
        
        # Symbol match (optional - allow GLOBAL or missing symbol field)
        adj_symbol = data.get("symbol", "GLOBAL")
        if adj_symbol not in [symbol, "GLOBAL", "?", ""]:
            logger.debug(f"[RL] Symbol mismatch: adjustment={adj_symbol}, request={symbol}")
            return None
        
        # Parse reward (optional)
        reward = None
        try:
            reward = float(data.get("reward", "0"))
        except (ValueError, TypeError):
            pass
        
        return {
            "delta": delta,
            "reward": reward,
            "symbol": adj_symbol,
            "timestamp": ts.isoformat(),
            "age_sec": age_sec
        }
    
    except Exception as e:
        logger.error(f"[RL] Error reading policy adjustment: {e}")
        return None


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check Redis connectivity
    redis_ok = False
    try:
        redis_client.ping()
        redis_ok = True
    except Exception:
        pass
    
    return {
        "status": "healthy" if redis_ok else "degraded",
        "service": "risk_brain",
        "timestamp": datetime.utcnow().isoformat(),
    Now integrates RL policy adjustments from quantum:ai_policy_adjustment.
    
    Args:
        request: Signal and operating mode
    
    Returns:
        Position size, leverage (RL-adjusted if available), risk score, max loss
    """
    symbol = request.signal.get("symbol", "UNKNOWN")
    mode = request.operating_mode
    confidence = request.signal.get("confidence", 0.5)
    
    # Base leverage from operating mode (existing stub logic)
    if mode == "EMERGENCY":
        position_size = 0.0
        base_leverage = 0.0
        risk_score = 100.0
    elif mode == "PRESERVATION":
        position_size = 50.0
        base_leverage = 1.0
        risk_score = 40.0
    else:  # EXPANSION
        position_size = 100.0
        base_leverage = 2.0
        risk_score = 30.0
    
    # Apply RL adjustment if available
    leverage = base_leverage
    rl_applied = False
    
    if base_leverage > 0:  # Only adjust non-zero leverage
        rl_adjustment = get_rl_policy_adjustment(symbol)
        
        if rl_adjustment:
            delta = rl_adjustment["delta"]
            multiplier = 1.0 + delta
            
            # Clamp multiplier
            multiplier = max(RL_LEVERAGE_MULTIPLIER_MIN, min(multiplier, RL_LEVERAGE_MULTIPLIER_MAX))
            
            # Apply to base leverage
            adjusted_leverage = base_leverage * multiplier
            
            # Final clamp to global limits
            leverage = max(1.0, min(adjusted_leverage, MAX_LEVERAGE))
            
            rl_applied = True
            
            logger.info(
                f"[RL-APPLIED] {symbol}: delta={delta:+.3f}, "
                f"base={base_leverage:.1f}x -> adjusted={leverage:.2f}x "
                f"(age={rl_adjustment['age_sec']:.0f}s, reward={rl_adjustment.get('reward', 0):.3f})"
            )
        else:
            logger.debug(f"[RL-SKIPPED] {symbol}: Using base leverage={base_leverage:.1f}x")
    
    max_loss = position_size * 0.02  # 2% max loss
    Starting Risk Brain Service (RL-enabled)
    return {
        "position_size": position_size,
        "leverage": leverage,
        "risk_score": risk_score,
        "max_loss": max_loss,
        "timestamp": datetime.utcnow().isoformat(),
        "rl_applied": rl_applied,  # Telemetry flag
    
    max_loss = position_size * 0.02  # 2% max loss
    
    return {
        "position_size": position_size,
        "leverage": leverage,
        "risk_score": risk_score,
        "max_loss": max_loss,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("⚠️ Starting Risk Brain Service on port 8012...")
    uvicorn.run(app, host="0.0.0.0", port=8012)
