#!/usr/bin/env python3
"""
Portfolio State Publisher (PSP)
Keeps quantum:state:portfolio fresh with authoritative portfolio data

Reads position snapshots and publishes consolidated portfolio state every 5s
to prevent P2.8 and other services from experiencing stale data gaps.
"""

import os
import sys
import time
import logging
from typing import Dict, Optional
import redis

# Config from environment
PSP_INTERVAL_SEC = int(os.getenv("PSP_INTERVAL_SEC", "5"))
PSP_PORTFOLIO_KEY = os.getenv("PSP_PORTFOLIO_KEY", "quantum:state:portfolio")
PSP_STATE_TTL_SEC = int(os.getenv("PSP_STATE_TTL_SEC", "120"))
PSP_STREAM_KEY = os.getenv("PSP_STREAM_KEY", "quantum:stream:portfolio.state")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Fail-safe config
MAX_LKG_AGE_SEC = 900  # 15 minutes
EQUITY_FALLBACK = 100000.0  # Conservative fallback if no equity source

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portfolio-state-publisher")

# Last-known-good cache
lkg_equity: Optional[float] = None
lkg_equity_timestamp: Optional[int] = None


def get_redis_connection() -> redis.Redis:
    """Create Redis connection with retry"""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5
    )


def get_position_snapshots(r: redis.Redis) -> Dict[str, Dict]:
    """Get all active position snapshots"""
    positions = {}
    
    try:
        # Get all position snapshot keys
        keys = r.keys("quantum:position:snapshot:*")
        
        for key in keys:
            try:
                data = r.hgetall(key)
                if not data:
                    continue
                
                symbol = key.decode().split(":")[-1]
                
                # Parse position data
                position_amt = float(data.get(b"position_amt", 0) or 0)
                mark_price = float(data.get(b"mark_price", 0) or 0)
                leverage = float(data.get(b"leverage", 1) or 1)
                unrealized_pnl = float(data.get(b"unrealized_pnl", 0) or 0)
                ts_epoch = int(data.get(b"ts_epoch", 0) or 0)
                
                # Calculate notional
                position_notional = abs(position_amt * mark_price)
                
                positions[symbol] = {
                    "position_amt": position_amt,
                    "mark_price": mark_price,
                    "leverage": leverage,
                    "notional_usd": position_notional,
                    "unrealized_pnl": unrealized_pnl,
                    "timestamp": ts_epoch
                }
                
            except Exception as e:
                logger.warning(f"Failed to parse position {key}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to get position snapshots: {e}")
    
    return positions


def get_equity(r: redis.Redis, positions: Dict) -> Optional[float]:
    """Get equity from balance_tracker (authoritative source)"""
    global lkg_equity, lkg_equity_timestamp
    
    now = int(time.time())
    
    # PRIMARY SOURCE: balance_tracker (quantum:account:balance)
    try:
        balance_data = r.hgetall(b"quantum:account:balance")
        if balance_data and b"balance" in balance_data:
            equity = float(balance_data[b"balance"])
            balance_ts = int(balance_data.get(b"timestamp", 0) or 0)
            
            # Verify freshness (balance_tracker updates every 30s)
            age = now - balance_ts
            if age < 120:  # Accept if < 2 minutes old
                lkg_equity = equity
                lkg_equity_timestamp = now
                logger.debug(f"Using balance_tracker equity: ${equity:.2f} (age={age}s)")
                return equity
            else:
                logger.warning(f"balance_tracker data stale (age={age}s), using LKG")
    except Exception as e:
        logger.error(f"Failed to read quantum:account:balance: {e}")
    
    # FALLBACK: Last-known-good if recent
    if lkg_equity is not None and lkg_equity_timestamp is not None:
        age = now - lkg_equity_timestamp
        if age < MAX_LKG_AGE_SEC:
            logger.warning(f"Using LKG equity (age={age}s): ${lkg_equity:.2f}")
            return lkg_equity
    
    # FAIL-SAFE: No valid equity source - return None to skip cycle
    logger.error("No valid equity source available (balance_tracker + LKG both unavailable)")
    return None


def update_risk_guard_equity(r: redis.Redis, equity: float) -> None:
    """Update RiskGuard equity key with peak tracking (canonical source)"""
    try:
        equity_key = "quantum:equity:current"
        
        # Get current peak
        existing_data = r.hgetall(equity_key)
        current_peak = float(existing_data.get(b"peak", equity) or equity) if existing_data else equity
        new_peak = max(current_peak, equity)
        
        # Update canonical RiskGuard key
        r.hset(equity_key, mapping={
            "equity": str(equity),
            "peak": str(new_peak),
            "last_update_ts": str(time.time())
        })
        
        drawdown_pct = ((new_peak - equity) / new_peak * 100) if new_peak > 0 else 0.0
        logger.info(f"✅ RiskGuard equity updated: ${equity:.2f} (peak=${new_peak:.2f}, dd={drawdown_pct:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to update RiskGuard equity: {e}")


def publish_portfolio_state(r: redis.Redis):
    """Main loop: publish portfolio state"""
    
    try:
        # Get current timestamp
        now = int(time.time())
        
        # Get positions
        positions = get_position_snapshots(r)
        
        # Calculate aggregates
        positions_count = len(positions)
        positions_notional_usd = sum(p["notional_usd"] for p in positions.values())
        total_unrealized_pnl = sum(p["unrealized_pnl"] for p in positions.values())
        
        # Get equity from balance_tracker
        equity_usd = get_equity(r, positions)
        
        if equity_usd is None:
            logger.error("Could not determine equity, skipping this cycle")
            return
        
        # CRITICAL: Update RiskGuard canonical equity key
        update_risk_guard_equity(r, equity_usd)
        
        # Calculate balance (equity - unrealized PnL)
        balance_usd = equity_usd - total_unrealized_pnl
        
        # Prepare state for secondary consumers (P2.8, dashboards)
        state = {
            "ts_utc": str(now),
            "timestamp": str(now),  # P2.8 compatibility
            "equity_usd": f"{equity_usd:.2f}",
            "balance_usd": f"{balance_usd:.2f}",
            "unrealized_pnl_usd": f"{total_unrealized_pnl:.2f}",
            "positions_count": str(positions_count),
            "positions_notional_usd": f"{positions_notional_usd:.2f}",
            "source": "portfolio-state-publisher",
            "drawdown": "0.0"  # TODO: Calculate from high-water mark
        }
        
        # Write to secondary state key (for dashboards, P2.8)
        r.hset(PSP_PORTFOLIO_KEY, mapping=state)
        r.expire(PSP_PORTFOLIO_KEY, PSP_STATE_TTL_SEC)
        
        # Publish to stream (for debugging/monitoring)
        try:
            r.xadd(PSP_STREAM_KEY, state, maxlen=1000)
        except Exception as e:
            logger.warning(f"Failed to publish to stream: {e}")
        
        logger.info(
            f"Published state: equity=${equity_usd:.2f} "
            f"positions={positions_count} notional=${positions_notional_usd:.2f}"
        )
        
    except redis.RedisError as e:
        logger.error(f"Redis error: {e}")
    except Exception as e:
        logger.error(f"Failed to publish state: {e}", exc_info=True)


def main():
    """Main loop"""
    logger.info("Portfolio State Publisher starting...")
    logger.info(f"Config: interval={PSP_INTERVAL_SEC}s ttl={PSP_STATE_TTL_SEC}s")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    
    while True:
        try:
            r = get_redis_connection()
            
            # Test connection
            r.ping()
            logger.info("Redis connection established")
            
            # Main loop
            while True:
                try:
                    publish_portfolio_state(r)
                except redis.ConnectionError as e:
                    logger.error(f"Redis connection lost: {e}")
                    break  # Reconnect
                except Exception as e:
                    logger.error(f"Error in publish cycle: {e}")
                
                time.sleep(PSP_INTERVAL_SEC)
                
        except redis.ConnectionError as e:
            logger.error(f"Could not connect to Redis: {e}")
            logger.info("Retrying in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
