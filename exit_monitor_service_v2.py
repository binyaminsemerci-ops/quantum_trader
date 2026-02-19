#!/usr/bin/env python3
"""
Exit Monitor Service - UPGRADED WITH NEW EXIT MATH
==================================================
Port: 8007
Subscribes: trade.execution.result (new positions)
Publishes: quantum:stream:trade.intent (close orders)

MAJOR UPGRADE: Now uses common.exit_math.py for dynamic, risk-based exits!
- Replaces hardcoded percentages with mathematical formulas
- Risk-normalized stop losses based on account equity
- ATR-adaptive trailing stops
- Leverage-aware exit calculations

Author: Quantum Trader Team
Date: 2026-02-18 (Upgraded)
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

# ============================================================================
# NEW: Import advanced exit math
# ============================================================================
from common.exit_math import (
    Position as ExitPosition, Account, Market, RiskSettings,
    compute_dynamic_stop, evaluate_exit, get_exit_metrics,
    compute_R, should_activate_trailing
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
# EXIT GUARD CONFIG (unchanged)
# ============================================================================
EXIT_DEDUP_TTL = 300
EXIT_COOLDOWN_TTL = 30
EXIT_STALE_THRESHOLD = 10
EXIT_RATE_LIMIT_PER_CYCLE = 1

# ============================================================================
# DATA MODELS (kept for compatibility)
# ============================================================================

@dataclass
class TrackedPosition:
    """Position being tracked for exit (enhanced for new exit math)"""
    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    quantity: float
    leverage: float
    take_profit: Optional[float]  # Still track for backwards compat
    stop_loss: Optional[float]    # Still track for backwards compat
    order_id: str
    opened_at: str
    highest_price: float = 0.0
    lowest_price: float = 999999.0
    position_id: Optional[str] = None

# ============================================================================
# GLOBAL STATE
# ============================================================================
app = FastAPI(title="Exit Monitor Service - NEW EXIT MATH", version="2.0.0")

eventbus: Optional[EventBusClient] = None
tracked_positions: Dict[str, TrackedPosition] = {}
binance_client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None

# Exit guard state
exit_cycle_counter: Dict[str, int] = defaultdict(int)

# Enhanced stats
stats = {
    "positions_tracked": 0,
    "exits_triggered": 0,
    "exits_deduped": 0,
    "exits_cooldown": 0,
    "exits_stale": 0,
    "exits_already_closed": 0,
    "exits_rate_limited": 0,
    
    # NEW: Exit reason breakdown
    "liq_protection_exits": 0,
    "risk_stop_exits": 0,
    "trailing_stop_exits": 0,
    "time_exits": 0,
    
    "last_check_time": None,
    "last_exit_evaluation": None
}

# Config
CHECK_INTERVAL = 5  # Check prices every 5 seconds

# NEW: Exit math settings
exit_settings = RiskSettings(
    RISK_FRACTION=0.005,  # 0.5% of account equity per trade
    STOP_ATR_MULT=1.2,
    TRAILING_ATR_MULT=1.5,
    TRAILING_ACTIVATION_R=1.0,
    MAX_HOLD_TIME=14400,  # 4 hours max hold
    LIQ_BUFFER_PCT=0.05   # 5% liquidation buffer
)

# ============================================================================
# BINANCE CLIENT SETUP
# ============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

if BINANCE_API_KEY and BINANCE_API_SECRET:
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
    binance_client.FUTURES_URL = "https://testnet.binancefuture.com"
    logger.info("✅ Binance Futures Testnet client initialized")
else:
    logger.warning("⚠️ No Binance credentials - will use fallback prices")

# ============================================================================
# NEW: Advanced market data functions
# ============================================================================

def get_account_equity() -> float:
    """Get account total equity for risk calculations"""
    try:
        account_info = binance_client.futures_account()
        equity = float(account_info['totalWalletBalance'])
        return equity
    except Exception as e:
        logger.error(f"❌ Failed to get account equity: {e}")
        return 5000.0  # Fallback

def calculate_atr(symbol: str) -> float:
    """Calculate ATR estimate for exit math"""
    try:
        # Get 24h ticker for ATR estimate
        ticker = binance_client.get_24hr_ticker(symbol=symbol)
        high_24h = float(ticker['highPrice'])
        low_24h = float(ticker['lowPrice'])
        current_price = float(ticker['lastPrice'])
        
        # ATR estimate: 20% of daily range
        daily_range = high_24h - low_24h
        atr_estimate = daily_range * 0.2
        
        # Minimum ATR (0.1% of price)
        min_atr = current_price * 0.001
        return max(atr_estimate, min_atr)
        
    except Exception as e:
        logger.error(f"❌ ATR calculation failed for {symbol}: {e}")
        # Fallback: 2% of current price
        try:
            price = get_current_price(symbol)
            return price * 0.02 if price > 0 else 0.01
        except:
            return 0.01

def get_time_in_trade(opened_at: str) -> float:
    """Calculate seconds since position opened"""
    try:
        position_time = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        age_seconds = (datetime.utcnow() - position_time.replace(tzinfo=None)).total_seconds()
        return age_seconds
    except:
        return 0.0

# ============================================================================
# EXIT GUARD HELPERS (unchanged)
# ============================================================================

async def check_exit_dedup(position_id: str) -> bool:
    """Check if exit already processed for this position_id"""
    try:
        key = f"quantum:dedup:exit:{position_id}"
        result = await redis_client.set(key, "1", nx=True, ex=EXIT_DEDUP_TTL)
        
        if not result:
            logger.info(f"🔴 EXIT_DEDUP skip pos={position_id}")
            stats["exits_deduped"] += 1
            return True
        
        return False
    except Exception as e:
        logger.error(f"❌ EXIT_DEDUP check failed: {e}")
        return False

async def check_exit_cooldown(symbol: str, side: str) -> bool:
    """Check if symbol/side is in cooldown period"""
    try:
        key = f"quantum:cooldown:exit:{symbol}:{side}"
        exists = await redis_client.exists(key)
        
        if exists:
            logger.info(f"⏸️ EXIT_COOLDOWN skip symbol={symbol} side={side}")
            stats["exits_cooldown"] += 1
            return True
        
        await redis_client.set(key, "1", ex=EXIT_COOLDOWN_TTL)
        return False
        
    except Exception as e:
        logger.error(f"❌ EXIT_COOLDOWN check failed: {e}")
        return False

async def check_already_closed(symbol: str) -> bool:
    """Check if position is already closed"""
    try:
        if symbol not in tracked_positions:
            logger.info(f"🔴 EXIT_ALREADY_CLOSED symbol={symbol}")
            stats["exits_already_closed"] += 1
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ EXIT_ALREADY_CLOSED check failed: {e}")
        return False

def check_exit_rate_limit(symbol: str) -> bool:
    """Check exit rate limit"""
    if exit_cycle_counter[symbol] >= EXIT_RATE_LIMIT_PER_CYCLE:
        logger.info(f"🚫 EXIT_RATE_LIMIT symbol={symbol} count={exit_cycle_counter[symbol]}")
        stats["exits_rate_limited"] += 1
        return True
    
    return False

def reset_exit_cycle_counters():
    """Reset per-cycle counters"""
    exit_cycle_counter.clear()

# ============================================================================
# ENHANCED HELPERS
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

# ============================================================================
# NEW: Advanced exit evaluation using exit_math.py
# ============================================================================

def evaluate_position_exit(position: TrackedPosition) -> Optional[str]:
    """
    NEW: Use advanced exit math for exit decisions
    Replaces old hardcoded percentage logic
    """
    try:
        # Get market data
        current_price = get_current_price(position.symbol)
        if not current_price:
            return None
            
        atr = calculate_atr(position.symbol)
        account_equity = get_account_equity()
        time_in_trade = get_time_in_trade(position.opened_at)
        
        # Create data structures for exit math
        exit_position = ExitPosition(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            size=position.quantity,
            leverage=position.leverage,
            highest_price=position.highest_price,
            lowest_price=position.lowest_price,
            time_in_trade=time_in_trade,
            distance_to_liq=0.10  # Estimate 10% to liquidation
        )
        
        account = Account(equity=account_equity)
        market = Market(current_price=current_price, atr=atr)
        
        # Update highest/lowest price tracking
        if position.side == "BUY":
            position.highest_price = max(position.highest_price, current_price)
        else:
            position.lowest_price = min(position.lowest_price, current_price)
        
        # Use advanced exit evaluation
        exit_reason = evaluate_exit(exit_position, account, market, exit_settings)
        
        # Log evaluation details
        if exit_reason:
            dynamic_stop = compute_dynamic_stop(exit_position, account, market, exit_settings)
            current_r = compute_R(exit_position, current_price, abs(exit_position.entry_price - dynamic_stop))
            
            logger.info(
                f"🧮 EXIT_MATH: {position.symbol} {position.side} | "
                f"Price=${current_price:.4f} | "
                f"DynamicStop=${dynamic_stop:.4f} | "
                f"R={current_r:+.2f} | "
                f"ATR=${atr:.6f} | "
                f"Equity=${account_equity:.2f} | "
                f"Reason={exit_reason}"
            )
        
        # Update stats timestamp
        stats["last_exit_evaluation"] = datetime.utcnow().isoformat()
        
        return exit_reason
        
    except Exception as e:
        logger.error(f"❌ Exit evaluation failed for {position.symbol}: {e}")
        return None

async def send_close_order(position: TrackedPosition, reason: str):
    """Send close order with all exit guards (enhanced with new stats)"""
    
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
    
    # === EXIT GUARD 4: Rate limit ===
    if check_exit_rate_limit(position.symbol):
        return
    
    try:
        close_side = "SELL" if position.side == "BUY" else "BUY"
        current_price = get_current_price(position.symbol)

        if not current_price:
            logger.error(f"❌ Cannot close {position.symbol} - price unavailable")
            return

        # Create TradeIntent for closing position
        intent = TradeIntent(
            symbol=position.symbol,
            side=close_side,
            position_size_usd=position.quantity * current_price,
            leverage=position.leverage,
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            confidence=1.0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model="exit_monitor_v2",
            meta_strategy="EXIT_MATH"
        )

        # Publish to execution stream
        await eventbus.publish(
            stream="quantum:stream:trade.intent",
            data=intent.dict()
        )

        # Update enhanced stats
        stats["exits_triggered"] += 1
        exit_cycle_counter[position.symbol] += 1
        
        # Track exit reason stats
        if reason == "liq_protection":
            stats["liq_protection_exits"] += 1
        elif reason == "risk_stop":
            stats["risk_stop_exits"] += 1
        elif reason == "trailing_stop":
            stats["trailing_stop_exits"] += 1
        elif reason == "time_exit":
            stats["time_exits"] += 1

        # Calculate PnL
        if position.side == "BUY":
            pnl_pct = ((current_price - position.entry_price) / position.entry_price * 100)
        else:
            pnl_pct = ((position.entry_price - current_price) / position.entry_price * 100)
        
        # Remove from tracked positions
        del tracked_positions[position.symbol]

        logger.info(
            f"📤 EXIT_PUBLISH_V2: {position.symbol} {position.side} | "
            f"Reason: {reason} | "
            f"Entry=${position.entry_price:.4f} | "
            f"Exit=${current_price:.4f} | "
            f"PnL={pnl_pct:+.2f}% | "
            f"pos_id={position_id}"
        )

    except Exception as e:
        logger.error(f"❌ Failed to send close order for {position.symbol}: {e}")

# ============================================================================
# BACKGROUND TASKS (enhanced)
# ============================================================================

async def position_listener():
    """Listen for new positions (now logs using exit math for preview)"""
    logger.info("📥 Starting position listener with EXIT MATH preview...")

    try:
        async for result_data in eventbus.subscribe("trade.execution.res"):
            result_data = {k: v for k, v in result_data.items() if not k.startswith('_')}
            result = ExecutionResult(**result_data)

            if result.status != "filled":
                continue

            # Calculate quantity
            quantity = result.position_size_usd / result.entry_price if result.entry_price > 0 else 0
            
            # Track position
            position = TrackedPosition(
                symbol=result.symbol,
                side=result.action,
                entry_price=result.entry_price,
                quantity=quantity,
                leverage=result.leverage,
                take_profit=None,  # Will be calculated dynamically
                stop_loss=None,    # Will be calculated dynamically
                order_id=result.order_id,
                opened_at=result.timestamp,
                highest_price=result.entry_price,
                lowest_price=result.entry_price,
                position_id=f"{result.symbol}_{result.order_id}"
            )

            tracked_positions[result.symbol] = position
            stats["positions_tracked"] += 1

            # NEW: Log exit math preview
            try:
                atr = calculate_atr(result.symbol)
                account_equity = get_account_equity()
                
                exit_position = ExitPosition(
                    symbol=result.symbol,
                    side=result.action,
                    entry_price=result.entry_price,
                    size=quantity,
                    leverage=result.leverage,
                    highest_price=result.entry_price,
                    lowest_price=result.entry_price,
                    time_in_trade=0,
                    distance_to_liq=0.10
                )
                
                account = Account(equity=account_equity)
                market = Market(current_price=result.entry_price, atr=atr)
                
                dynamic_stop = compute_dynamic_stop(exit_position, account, market, exit_settings)
                stop_distance_pct = abs(result.entry_price - dynamic_stop) / result.entry_price * 100
                
                logger.info(
                    f"📊 TRACKING_V2: {position.symbol} {position.side} | "
                    f"Entry=${position.entry_price:.4f} | "
                    f"DynamicStop=${dynamic_stop:.4f} ({stop_distance_pct:.2f}%) | "
                    f"ATR=${atr:.6f} | "
                    f"Qty={quantity:.4f} | "
                    f"pos_id={position.position_id}"
                )
                
            except Exception as e:
                logger.warning(f"Exit math preview failed: {e}")
                logger.info(f"📊 TRACKING: {position.symbol} {position.side} fallback log")

    except Exception as e:
        logger.error(f"❌ Position listener error: {e}", exc_info=True)

async def exit_monitor_loop():
    """Monitor tracked positions using NEW EXIT MATH"""
    logger.info("🔍 Starting exit monitor loop with ADVANCED EXIT MATH...")

    await asyncio.sleep(10)

    while True:
        try:
            reset_exit_cycle_counters()
            
            if not tracked_positions:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            stats["last_check_time"] = datetime.utcnow().isoformat()
            
            # Check each position with NEW EXIT MATH
            for symbol, position in list(tracked_positions.items()):
                exit_reason = evaluate_position_exit(position)
                
                if exit_reason:
                    await send_close_order(position, exit_reason)
            
            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Exit monitor error: {e}", exc_info=True)
            await asyncio.sleep(CHECK_INTERVAL)

# ============================================================================
# API ENDPOINTS (enhanced)
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
    
    # NEW: Exit reason breakdown
    liq_protection_exits: int
    risk_stop_exits: int
    trailing_stop_exits: int
    time_exits: int
    
    last_check_time: Optional[str]
    last_exit_evaluation: Optional[str]
    exit_math_version: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Enhanced health check with exit math stats"""
    return HealthResponse(
        status="healthy_v2",
        tracked_positions=len(tracked_positions),
        exits_triggered=stats["exits_triggered"],
        exits_deduped=stats["exits_deduped"],
        exits_cooldown=stats["exits_cooldown"],
        exits_stale=stats["exits_stale"],
        exits_already_closed=stats["exits_already_closed"],
        exits_rate_limited=stats["exits_rate_limited"],
        liq_protection_exits=stats["liq_protection_exits"],
        risk_stop_exits=stats["risk_stop_exits"],
        trailing_stop_exits=stats["trailing_stop_exits"],
        time_exits=stats["time_exits"],
        last_check_time=stats["last_check_time"],
        last_exit_evaluation=stats["last_exit_evaluation"],
        exit_math_version="v2.0_dynamic_risk_based"
    )

@app.get("/positions")
async def get_positions():
    """Get currently tracked positions with exit math preview"""
    positions_with_math = []
    
    for pos in tracked_positions.values():
        try:
            # Add exit math preview
            current_price = get_current_price(pos.symbol)
            if current_price:
                atr = calculate_atr(pos.symbol)
                account_equity = get_account_equity()
                time_in_trade = get_time_in_trade(pos.opened_at)
                
                exit_position = ExitPosition(
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    size=pos.quantity,
                    leverage=pos.leverage,
                    highest_price=pos.highest_price,
                    lowest_price=pos.lowest_price,
                    time_in_trade=time_in_trade,
                    distance_to_liq=0.10
                )
                
                account = Account(equity=account_equity)
                market = Market(current_price=current_price, atr=atr)
                
                dynamic_stop = compute_dynamic_stop(exit_position, account, market, exit_settings)
                current_r = compute_R(exit_position, current_price, abs(pos.entry_price - dynamic_stop))
                exit_reason = evaluate_exit(exit_position, account, market, exit_settings)
                
                positions_with_math.append({
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "quantity": pos.quantity,
                    "dynamic_stop": dynamic_stop,
                    "current_r": current_r,
                    "exit_reason": exit_reason,
                    "atr": atr,
                    "time_in_trade": time_in_trade,
                    "opened_at": pos.opened_at,
                    "position_id": pos.position_id
                })
            else:
                # Fallback without exit math
                positions_with_math.append({
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "opened_at": pos.opened_at,
                    "position_id": pos.position_id,
                    "note": "Market data unavailable"
                })
                
        except Exception as e:
            logger.error(f"Position math failed for {pos.symbol}: {e}")
            
    return {
        "count": len(tracked_positions),
        "positions": positions_with_math,
        "exit_math_version": "v2.0"
    }

# ============================================================================
# LIFECYCLE (unchanged)
# ============================================================================

@app.on_event("startup")
async def startup():
    global eventbus, redis_client
    
    try:
        # Initialize eventbus
        eventbus = EventBusClient()
        await eventbus.connect()
        logger.info("✅ EventBus connected")
        
        # Initialize Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis connected")
        
        # Start background tasks
        asyncio.create_task(position_listener())
        asyncio.create_task(exit_monitor_loop())
        
        logger.info("🚀 Exit Monitor Service V2 with EXIT MATH started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    if eventbus:
        await eventbus.disconnect()
    if redis_client:
        await redis_client.close()

if __name__ == "__main__":
    uvicorn.run(
        "exit_monitor_service_v2:app",
        host="0.0.0.0",
        port=8007,
        log_level="info",
        access_log=True
    )