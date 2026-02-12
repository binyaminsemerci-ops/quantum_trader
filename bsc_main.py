#!/usr/bin/env python3
"""
Baseline Safety Controller (BSC)
Emergency Exit-Only Controller under Authority Freeze

Authority: CONTROLLER (RESTRICTED)
Scope: Exit only, fixed thresholds, direct Binance
Fail Mode: FAIL OPEN (no action on error)
Lifespan: Max 30 days (then automatic demotion)
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import redis

# === HARD BOUNDARIES (IMMUTABLE) ===
MAX_LOSS_PCT = -3.0          # Close if unrealized loss <= -3%
MAX_DURATION_HOURS = 72      # Close if position age >= 72 hours
MAX_MARGIN_RATIO = 0.85      # Close if margin ratio >= 85%
POLL_INTERVAL_SEC = 60       # Check positions every 60 seconds
MAX_RETRY_COUNT = 3          # Max 3 retries on Binance API failure

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BSC")

# === REDIS CONNECTION ===
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# === BINANCE CLIENT ===
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    logger.error("❌ BINANCE_API_KEY or BINANCE_API_SECRET not set")
    sys.exit(1)

binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=BINANCE_TESTNET)

if BINANCE_TESTNET:
    logger.warning("⚠️  TESTNET MODE - Using Binance Testnet")

# === TELEMETRY ===
def log_bsc_event(event_type: str, symbol: str, reason: str, details: Dict):
    """Log BSC action to Redis (audit trail only, not control)"""
    try:
        event = {
            "event": event_type,
            "symbol": symbol,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details
        }
        redis_client.xadd("quantum:stream:bsc.events", event)
        logger.info(f"📊 Event logged: {event_type} {symbol} ({reason})")
    except Exception as e:
        logger.warning(f"⚠️  Failed to log event: {e} (continuing)")

# === POSITION FETCHER (DIRECT BINANCE POLL) ===
def get_open_positions() -> List[Dict]:
    """Fetch open positions directly from Binance (no Redis streams)"""
    try:
        positions = binance_client.futures_position_information()
        open_positions = []
        
        for pos in positions:
            position_amt = float(pos["positionAmt"])
            if position_amt != 0:  # Only active positions
                open_positions.append({
                    "symbol": pos["symbol"],
                    "position_amt": position_amt,
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                    "margin_ratio": float(pos.get("marginRatio", 0)),
                    "update_time": int(pos["updateTime"])
                })
        
        return open_positions
    
    except BinanceAPIException as e:
        logger.error(f"❌ Binance API error: {e} (FAIL OPEN - no action)")
        return []  # FAIL OPEN
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching positions: {e} (FAIL OPEN)")
        return []  # FAIL OPEN

# === DECISION LOGIC (NORMATIVE - DO NOT MODIFY) ===
def should_close_position(pos: Dict, position_age_hours: float) -> Optional[str]:
    """
    Returns close reason if ANY threshold breached, else None
    
    Rules (OR logic):
    1. unrealized_pnl_pct <= -3.0%
    2. position_age_hours >= 72h
    3. margin_ratio >= 0.85
    """
    # Calculate unrealized PnL %
    entry_price = pos["entry_price"]
    position_amt = pos["position_amt"]
    unrealized_pnl = pos["unrealized_pnl"]
    
    if entry_price != 0 and position_amt != 0:
        position_value = abs(position_amt * entry_price)
        unrealized_pnl_pct = (unrealized_pnl / position_value) * 100
    else:
        unrealized_pnl_pct = 0.0
    
    # RULE 1: Max loss
    if unrealized_pnl_pct <= MAX_LOSS_PCT:
        return f"MAX_LOSS_BREACH (pnl={unrealized_pnl_pct:.2f}%)"
    
    # RULE 2: Max duration
    if position_age_hours >= MAX_DURATION_HOURS:
        return f"MAX_DURATION_BREACH (age={position_age_hours:.1f}h)"
    
    # RULE 3: Liquidation risk
    margin_ratio = pos["margin_ratio"]
    if margin_ratio >= MAX_MARGIN_RATIO:
        return f"LIQUIDATION_RISK (margin={margin_ratio:.3f})"
    
    return None  # No breach

# === EXECUTION (DIRECT BINANCE - NO PIPELINE) ===
def force_close_position(pos: Dict, reason: str) -> bool:
    """
    Execute MARKET close order directly to Binance
    
    CRITICAL: No apply_layer, no harvest_brain, no execution_service
    Direct API call with reduceOnly=True
    """
    symbol = pos["symbol"]
    position_amt = pos["position_amt"]
    
    # Determine side (LONG positions → SELL, SHORT positions → BUY)
    if position_amt > 0:
        side = "SELL"  # Close LONG
    else:
        side = "BUY"   # Close SHORT
    
    quantity = abs(position_amt)
    
    logger.info(f"🔥 FORCE CLOSE: {symbol} {side} {quantity} (reason: {reason})")
    
    for attempt in range(1, MAX_RETRY_COUNT + 1):
        try:
            order = binance_client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=True  # CRITICAL: Only close, never increase
            )
            
            logger.info(f"✅ CLOSE EXECUTED: {symbol} orderId={order['orderId']}")
            
            # Log to audit trail
            log_bsc_event(
                event_type="BSC_FORCE_CLOSE",
                symbol=symbol,
                reason=reason,
                details={
                    "side": side,
                    "quantity": quantity,
                    "order_id": order["orderId"],
                    "unrealized_pnl": pos["unrealized_pnl"],
                    "entry_price": pos["entry_price"]
                }
            )
            
            return True
        
        except BinanceAPIException as e:
            logger.error(f"❌ Binance API error (attempt {attempt}/{MAX_RETRY_COUNT}): {e}")
            if attempt < MAX_RETRY_COUNT:
                time.sleep(2)  # Brief retry delay
            else:
                logger.error(f"❌ CLOSE FAILED after {MAX_RETRY_COUNT} attempts (FAIL OPEN)")
                log_bsc_event(
                    event_type="BSC_CLOSE_FAILED",
                    symbol=symbol,
                    reason=f"API_ERROR:{e.code}",
                    details={"error_msg": str(e)}
                )
                return False  # FAIL OPEN
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e} (FAIL OPEN)")
            return False  # FAIL OPEN
    
    return False

# === POSITION AGE TRACKER ===
position_first_seen = {}  # symbol → first_seen_timestamp

def get_position_age_hours(symbol: str, update_time: int) -> float:
    """Track how long position has been open (approx via first poll)"""
    if symbol not in position_first_seen:
        position_first_seen[symbol] = update_time
    
    age_ms = update_time - position_first_seen[symbol]
    age_hours = age_ms / (1000 * 3600)
    return age_hours

# === MAIN LOOP ===
def run_bsc():
    """
    BSC main loop:
    1. Poll Binance for open positions
    2. Check each against fixed thresholds
    3. Force close if ANY threshold breached
    4. FAIL OPEN on any error
    """
    logger.info("=" * 60)
    logger.info("🛑 BASELINE SAFETY CONTROLLER (BSC) STARTING")
    logger.info("=" * 60)
    logger.info(f"Authority: CONTROLLER (RESTRICTED - Exit Only)")
    logger.info(f"Thresholds: MAX_LOSS={MAX_LOSS_PCT}%, MAX_DURATION={MAX_DURATION_HOURS}h, MAX_MARGIN={MAX_MARGIN_RATIO}")
    logger.info(f"Poll Interval: {POLL_INTERVAL_SEC}s")
    logger.info(f"Fail Mode: FAIL OPEN (no action on error)")
    logger.info("=" * 60)
    
    # Log activation event
    try:
        redis_client.xadd("quantum:stream:bsc.events", {
            "event": "BSC_ACTIVATED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "max_loss_pct": MAX_LOSS_PCT,
            "max_duration_hours": MAX_DURATION_HOURS,
            "max_margin_ratio": MAX_MARGIN_RATIO
        })
    except Exception as e:
        logger.warning(f"⚠️  Failed to log activation: {e}")
    
    cycle_count = 0
    
    while True:
        cycle_count += 1
        logger.info(f"🔍 BSC Check Cycle #{cycle_count}")
        
        try:
            # Fetch open positions directly from Binance
            positions = get_open_positions()
            
            if not positions:
                logger.info(f"✅ No open positions (or fetch failed → FAIL OPEN)")
            else:
                logger.info(f"📊 Found {len(positions)} open position(s)")
                
                for pos in positions:
                    symbol = pos["symbol"]
                    position_amt = pos["position_amt"]
                    unrealized_pnl = pos["unrealized_pnl"]
                    margin_ratio = pos["margin_ratio"]
                    
                    # Calculate position age
                    position_age_hours = get_position_age_hours(symbol, pos["update_time"])
                    
                    logger.info(f"  {symbol}: amt={position_amt} pnl={unrealized_pnl:.2f} margin={margin_ratio:.3f} age={position_age_hours:.1f}h")
                    
                    # Check if close triggered
                    close_reason = should_close_position(pos, position_age_hours)
                    
                    if close_reason:
                        logger.warning(f"⚠️  THRESHOLD BREACHED: {symbol} - {close_reason}")
                        force_close_position(pos, close_reason)
                    else:
                        logger.info(f"  {symbol}: Within safety limits ✓")
            
            # Log health check
            try:
                redis_client.xadd("quantum:stream:bsc.events", {
                    "event": "BSC_HEALTH_CHECK",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "positions_checked": len(positions),
                    "cycle": cycle_count
                })
            except Exception as e:
                logger.warning(f"⚠️  Failed to log health check: {e}")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in main loop: {e} (FAIL OPEN - continuing)")
        
        # Sleep until next cycle
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    try:
        run_bsc()
    except KeyboardInterrupt:
        logger.info("🛑 BSC stopped by user")
        try:
            redis_client.xadd("quantum:stream:bsc.events", {
                "event": "BSC_STOPPED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "USER_INTERRUPT"
            })
        except:
            pass
    except Exception as e:
        logger.error(f"❌ BSC FATAL ERROR: {e}")
        sys.exit(1)
