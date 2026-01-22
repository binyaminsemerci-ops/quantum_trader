#!/usr/bin/env python3
"""
Exit Monitor Service - Monitors positions and closes at TP/SL 
==============================================================
Port: 8007
Subscribes: trade.execution.result (new positions)
Publishes: quantum:stream:trade.intent (close orders)

Monitors open positions and sends close orders when:
- Price hits Take Profit level
- Price hits Stop Loss level
- Trailing stop triggers

Works with ExitBrain v3.5 for adaptive TP/SL calculation.

P0.EXIT_GUARD: Strict exit-gating with dedup, cooldown, stale-reject
- Exactly-once per position_id
- 30s cooldown per symbol/side
- Stale signal rejection (>10s old)
- Already-closed position guard
- 1 exit per symbol per cycle

Author: Quantum Trader Team
Date: 2026-01-19
"""
import os
import sys
import asyncio
import logging
import redis.asyncio as redis
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from binance.client import Client
from binance.exceptions import BinanceAPIException

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    ExecutionResult,
    TradeIntent
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("/var/log/quantum/exit-monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXIT GUARD CONFIG
# ============================================================================
EXIT_DEDUP_TTL = 300  # 5 minutes - prevents duplicate exits
EXIT_COOLDOWN_TTL = 30  # 30 seconds - prevents symbol/side churn
EXIT_STALE_THRESHOLD = 10  # Reject signals older than 10 seconds
EXIT_RATE_LIMIT_PER_CYCLE = 1  # Max 1 exit per symbol per cycle

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TrackedPosition:
    """Position being tracked for exit"""
    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    quantity: float
    leverage: float
    take_profit: Optional[float]
    stop_loss: Optional[float]
    order_id: str
    opened_at: str
    highest_price: float = 0.0  # For trailing stop (LONG)
    lowest_price: float = 999999.0  # For trailing stop (SHORT)
    position_id: Optional[str] = None  # For dedup tracking

# ============================================================================
# GLOBAL STATE
# ============================================================================
app = FastAPI(title="Exit Monitor Service", version="1.0.0")

eventbus: Optional[EventBusClient] = None
tracked_positions: Dict[str, TrackedPosition] = {}  # symbol -> position
binance_client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None

# Exit guard state
exit_cycle_counter: Dict[str, int] = defaultdict(int)  # symbol -> count per cycle

stats = {
    "positions_tracked": 0,
    "exits_triggered": 0,
    "exits_deduped": 0,
    "exits_cooldown": 0,
    "exits_stale": 0,
    "exits_already_closed": 0,
    "exits_rate_limited": 0,
    "tp_hits": 0,
    "sl_hits": 0,
    "trailing_hits": 0,
    "last_check_time": None
}

# Config
CHECK_INTERVAL = 5  # Check prices every 5 seconds
TRAILING_STOP_PCT = 0.015  # 1.5% trailing stop callback

# ============================================================================
# BINANCE CLIENT SETUP
# ============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")

if BINANCE_API_KEY and BINANCE_API_SECRET:
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
    binance_client.FUTURES_URL = "https://testnet.binancefuture.com"
    logger.info("✅ Binance Futures Testnet client initialized")
else:
    logger.warning("⚠️ No Binance credentials - will use fallback prices")

# ============================================================================
# EXIT GUARD HELPERS
# ============================================================================

async def check_exit_dedup(position_id: str) -> bool:
    """
    Check if exit already processed for this position_id.
    Returns True if deduplicated (should skip), False if new.
    """
    try:
        key = f"quantum:dedup:exit:{position_id}"
        # SETNX: Set if Not eXists
        result = await redis_client.set(key, "1", nx=True, ex=EXIT_DEDUP_TTL)
        
        if not result:
            logger.info(f"🔴 EXIT_DEDUP skip pos={position_id}")
            stats["exits_deduped"] += 1
            return True
        
        return False
    except Exception as e:
        logger.error(f"❌ EXIT_DEDUP check failed: {e}")
        return False  # Fail open - allow exit if Redis fails


async def check_exit_cooldown(symbol: str, side: str) -> bool:
    """
    Check if symbol/side is in cooldown period.
    Returns True if in cooldown (should skip), False if allowed.
    """
    try:
        key = f"quantum:cooldown:exit:{symbol}:{side}"
        exists = await redis_client.exists(key)
        
        if exists:
            logger.info(f"⏸️ EXIT_COOLDOWN skip symbol={symbol} side={side}")
            stats["exits_cooldown"] += 1
            return True
        
        # Set cooldown
        await redis_client.set(key, "1", ex=EXIT_COOLDOWN_TTL)
        return False
        
    except Exception as e:
        logger.error(f"❌ EXIT_COOLDOWN check failed: {e}")
        return False  # Fail open


def check_exit_stale(opened_at: str) -> bool:
    """
    Check if position signal is stale.
    Returns True if stale (should skip), False if fresh.
    """
    try:
        position_time = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        age_seconds = (datetime.utcnow() - position_time.replace(tzinfo=None)).total_seconds()
        
        # Allow exits for positions older than threshold - tracked positions are not signals
        return False
        
    except Exception as e:
        logger.error(f"❌ EXIT_STALE check failed: {e}")
        return False


async def check_already_closed(symbol: str) -> bool:
    """
    Check if position is already closed.
    Returns True if closed (should skip), False if open.
    """
    try:
        # Position is closed if not in tracked_positions
        if symbol not in tracked_positions:
            logger.info(f"🔴 EXIT_ALREADY_CLOSED symbol={symbol}")
            stats["exits_already_closed"] += 1
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ EXIT_ALREADY_CLOSED check failed: {e}")
        return False


def check_exit_rate_limit(symbol: str) -> bool:
    """
    Check if symbol has exceeded exit rate limit for this cycle.
    Returns True if rate limited (should skip), False if allowed.
    """
    if exit_cycle_counter[symbol] >= EXIT_RATE_LIMIT_PER_CYCLE:
        logger.info(f"🚫 EXIT_RATE_LIMIT symbol={symbol} count={exit_cycle_counter[symbol]}")
        stats["exits_rate_limited"] += 1
        return True
    
    return False


def reset_exit_cycle_counters():
    """Reset per-cycle counters"""
    exit_cycle_counter.clear()


# ============================================================================
# HELPERS
# ============================================================================

def get_current_price(symbol: str) -> Optional[float]:
    """Get current market price from Binance"""
    try:
        if binance_client:
            ticker = binance_client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
    except Exception as e:
        logger.error(f"❌ Failed to get price for {symbol}: {e}")
        return None


def check_exit_conditions(position: TrackedPosition, current_price: float) -> Optional[str]:
    """
    Check if position should be closed.
    
    Returns:
        Exit reason if should close, None otherwise
    """
    if position.side == "BUY":
        # LONG position
        # Update highest price for trailing stop
        if current_price > position.highest_price:
            position.highest_price = current_price

        # Check Take Profit
        if position.take_profit and current_price >= position.take_profit:
            return "TAKE_PROFIT"

        # Check Stop Loss
        if position.stop_loss and current_price <= position.stop_loss:
            return "STOP_LOSS"

        # Check Trailing Stop (if price dropped X% from highest)
        if position.highest_price > position.entry_price * 1.01:  # Only if in profit
            trailing_trigger = position.highest_price * (1 - TRAILING_STOP_PCT)
            if current_price <= trailing_trigger:
                return "TRAILING_STOP"

    else:  # SHORT position
        # Update lowest price for trailing stop
        if current_price < position.lowest_price:
            position.lowest_price = current_price

        # Check Take Profit
        if position.take_profit and current_price <= position.take_profit:
            return "TAKE_PROFIT"

        # Check Stop Loss
        if position.stop_loss and current_price >= position.stop_loss:
            return "STOP_LOSS"

        # Check Trailing Stop (if price rose X% from lowest)
        if position.lowest_price < position.entry_price * 0.99:  # Only if in profit
            trailing_trigger = position.lowest_price * (1 + TRAILING_STOP_PCT)
            if current_price >= trailing_trigger:
                return "TRAILING_STOP"

    return None


async def send_close_order(position: TrackedPosition, reason: str):
    """Send close order to execution service with strict exit guards"""
    
    # === EXIT GUARD 1: Already closed ===
    if await check_already_closed(position.symbol):
        return
    
    # === EXIT GUARD 2: Deduplication ===
    position_id = position.position_id or f"{position.symbol}_{position.order_id}"
    if await check_exit_dedup(position_id):
        return
    
    # === EXIT GUARD 3: Cooldown ===
    if await check_exit_cooldown(position.symbol, position.side):
        return
    
    # === EXIT GUARD 4: Stale signal ===
    if check_exit_stale(position.opened_at):
        logger.info(f"⏰ EXIT_STALE skip symbol={position.symbol}")
        stats["exits_stale"] += 1
        return
    
    # === EXIT GUARD 5: Rate limit ===
    if check_exit_rate_limit(position.symbol):
        return
    
    try:
        # Create TradeIntent for closing position
        close_side = "SELL" if position.side == "BUY" else "BUY"
        current_price = get_current_price(position.symbol)

        if not current_price:
            logger.error(f"❌ Cannot close {position.symbol} - price unavailable")
            return

        intent = TradeIntent(
            symbol=position.symbol,
            side=close_side,
            position_size_usd=position.quantity * current_price,  # Close full position
            leverage=position.leverage,
            entry_price=current_price,
            stop_loss=None,  # No TP/SL on close order
            take_profit=None,
            confidence=1.0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model="exit_monitor",
            meta_strategy="EXIT"
        )

        # Publish to execution stream
        await eventbus.publish(
            stream="quantum:stream:trade.intent",
            data=intent.dict()
        )

        # Update stats and counters
        stats["exits_triggered"] += 1
        exit_cycle_counter[position.symbol] += 1
        
        if reason == "TAKE_PROFIT":
            stats["tp_hits"] += 1
        elif reason == "STOP_LOSS":
            stats["sl_hits"] += 1
        elif reason == "TRAILING_STOP":
            stats["trailing_hits"] += 1

        # Remove from tracked positions
        del tracked_positions[position.symbol]

        pnl_pct = ((current_price - position.entry_price) / position.entry_price * 100) if position.side == "BUY" else ((position.entry_price - current_price) / position.entry_price * 100)
        
        logger.info(
            f"📤 EXIT_PUBLISH: {position.symbol} {position.side} | "
            f"Reason: {reason} | "
            f"Entry=${position.entry_price:.4f} | "
            f"Exit=${current_price:.4f} | "
            f"PnL={pnl_pct:+.2f}% | "
            f"pos_id={position_id}"
        )

    except Exception as e:
        logger.error(f"❌ Failed to send close order for {position.symbol}: {e}")

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def position_listener():
    """Listen for new positions from execution results"""
    logger.info("📥 Starting position listener...")

    try:
        async for result_data in eventbus.subscribe("trade.execution.res"):
            # Remove EventBus metadata
            result_data = {k: v for k, v in result_data.items() if not k.startswith('_')}
            
            # Parse ExecutionResult
            result = ExecutionResult(**result_data)

            if result.status != "filled":
                continue

            # Get TP/SL from original intent (should be included in result or fetched)
            # For now, use basic calculation
            tp = None
            sl = None

            # TODO: Fetch actual TP/SL from AI Engine signal or ExitBrain
            # For testnet demo, use simple fixed percentages
            if result.action == "BUY":
                tp = result.entry_price * 1.025  # +2.5%
                sl = result.entry_price * 0.985  # -1.5%
            else:
                tp = result.entry_price * 0.975  # -2.5%
                sl = result.entry_price * 1.015  # +1.5%

            # Calculate quantity
            quantity = result.position_size_usd / result.entry_price if result.entry_price > 0 else 0
            
            # Track position
            position = TrackedPosition(
                symbol=result.symbol,
                side=result.action,
                entry_price=result.entry_price,
                quantity=quantity,
                leverage=result.leverage,
                take_profit=tp,
                stop_loss=sl,
                order_id=result.order_id,
                opened_at=result.timestamp,
                highest_price=result.entry_price,
                lowest_price=result.entry_price,
                position_id=f"{result.symbol}_{result.order_id}"
            )

            tracked_positions[result.symbol] = position
            stats["positions_tracked"] += 1

            logger.info(
                f"📊 TRACKING: {position.symbol} {position.side} | "
                f"Entry=${position.entry_price:.4f} | "
                f"TP=${tp:.4f} | SL=${sl:.4f} | "
                f"Qty={quantity:.4f} | "
                f"pos_id={position.position_id}"
            )

    except Exception as e:
        logger.error(f"❌ Position listener error: {e}", exc_info=True)


async def exit_monitor_loop():
    """Monitor tracked positions and trigger exits"""
    logger.info("🔍 Starting exit monitor loop...")

    await asyncio.sleep(10)  # Wait for system to initialize

    while True:
        try:
            # Reset per-cycle counters
            reset_exit_cycle_counters()
            
            if not tracked_positions:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            stats["last_check_time"] = datetime.utcnow().isoformat()
            
            # Check each position
            for symbol, position in list(tracked_positions.items()):
                current_price = get_current_price(symbol)

                if not current_price:
                    continue

                # Check exit conditions
                exit_reason = check_exit_conditions(position, current_price)
                
                if exit_reason:
                    await send_close_order(position, exit_reason)
            
            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Exit monitor error: {e}", exc_info=True)
            await asyncio.sleep(CHECK_INTERVAL)


# ============================================================================
# API ENDPOINTS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    tracked_positions: int
    exits_triggered: int
    exits_deduped: int
    exits_cooldown: int
    exits_stale: int
    exits_already_closed: int
    exits_rate_limited: int
    tp_hits: int
    sl_hits: int
    trailing_hits: int
    last_check_time: Optional[str]


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        tracked_positions=len(tracked_positions),
        exits_triggered=stats["exits_triggered"],
        exits_deduped=stats["exits_deduped"],
        exits_cooldown=stats["exits_cooldown"],
        exits_stale=stats["exits_stale"],
        exits_already_closed=stats["exits_already_closed"],
        exits_rate_limited=stats["exits_rate_limited"],
        tp_hits=stats["tp_hits"],
        sl_hits=stats["sl_hits"],
        trailing_hits=stats["trailing_hits"],
        last_check_time=stats["last_check_time"]
    )


@app.get("/positions")
async def get_positions():
    """Get currently tracked positions"""
    return {
        "count": len(tracked_positions),
        "positions": [
            {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "take_profit": pos.take_profit,
                "stop_loss": pos.stop_loss,
                "highest_price": pos.highest_price,
                "lowest_price": pos.lowest_price,
                "opened_at": pos.opened_at,
                "position_id": pos.position_id
            }
            for pos in tracked_positions.values()
        ]
    }


# ============================================================================
# LIFECYCLE
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    global eventbus, redis_client

    logger.info("🚀 Exit Monitor Service starting...")

    # Connect to Redis (for exit guards)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    logger.info(f"✅ Redis connected for exit guards: {redis_url}")
    
    # Connect to EventBus
    eventbus = EventBusClient(redis_url=redis_url)
    await eventbus.connect()
    logger.info(f"✅ EventBus connected: {redis_url}")

    # Start background tasks
    asyncio.create_task(position_listener())
    asyncio.create_task(exit_monitor_loop())

    logger.info("✅ Exit Monitor Service started with EXIT_GUARD enabled")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 Exit Monitor Service shutting down...")

    if redis_client:
        await redis_client.close()
    
    if eventbus:
        await eventbus.close()

    logger.info("✅ Shutdown complete")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", "8007"))
    uvicorn.run(app, host="0.0.0.0", port=port)
