#!/usr/bin/env python3
"""
Exit Monitor Service - Monitors positions and closes at TP/SL
==============================================================
Port: 8007
Subscribes: quantum:stream:apply.result (FIXED: was trade.execution.result)
Publishes: quantum:stream:trade.intent (close orders)

Monitors open positions and sends close orders when:
- Price hits Take Profit level
- Price hits Stop Loss level
- Trailing stop triggers

Works with ExitBrain v3.5 for adaptive TP/SL calculation.

Author: Quantum Trader Team
Date: 2026-01-16
Updated: 2026-02-18 (FIX: Subscribe to apply.result instead of trade.execution.res)
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

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


# ============================================================================
# GLOBAL STATE
# ============================================================================

app = FastAPI(title="Exit Monitor Service", version="1.0.0")

eventbus: Optional[EventBusClient] = None
tracked_positions: Dict[str, TrackedPosition] = {}  # symbol -> position
binance_client: Optional[Client] = None

stats = {
    "positions_tracked": 0,
    "exits_triggered": 0,
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
    """Send close order to execution service"""
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
        
        # Update stats
        stats["exits_triggered"] += 1
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
            f"🎯 EXIT TRIGGERED: {position.symbol} {position.side} | "
            f"Reason: {reason} | "
            f"Entry=${position.entry_price:.4f} | "
            f"Exit=${current_price:.4f} | "
            f"PnL={pnl_pct:+.2f}%"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to send close order for {position.symbol}: {e}")


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def position_listener():
    """Listen for new positions from execution results (using direct Redis XREAD)"""
    logger.info("📥 Starting position listener (monitoring quantum:stream:apply.result)...")
    
    # Get Redis client from EventBus
    redis_client = eventbus.redis
    if not redis_client:
        logger.error("❌ Redis client not available from EventBus")
        return
    
    stream_name = "quantum:stream:apply.result"
    last_id = "$"  # Start from new messages only
    
    logger.info(f"✅ Position listener using direct Redis XREAD on {stream_name}")
    
    loop_count = 0
    try:
        while True:
            loop_count += 1
            if loop_count % 10 == 0:  # Log every 10 iterations (10 seconds)
                logger.info(f"📊 Position listener loop active (iteration {loop_count})")
            
            try:
                # Direct XREAD (bypassing EventBus to handle flat field format)
                messages = await redis_client.xread(
                    {stream_name: last_id},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                if not messages:
                    # No new messages, continue
                    continue
                
                # Process messages: messages = [(stream_name, [(message_id, fields_dict), ...])]
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        # Decode bytes to strings
                        result_data = {}
                        for k, v in fields.items():
                            key = k.decode() if isinstance(k, bytes) else k
                            value = v.decode() if isinstance(v, bytes) else v
                            result_data[key] = value
                        
                        # Update last_id for next iteration
                        last_id = message_id
                        
                        # Check if this is a successful execution
                        executed = result_data.get('executed')
                        
                        if executed != 'true' and executed != True:
                            # Skip non-executed events (most events)
                            continue
                        
                        # ALWAYS log when we find executed=true
                        logger.info(f"✅ EXECUTED=TRUE event for {result_data.get('symbol')} | fields={list(result_data.keys())}")
                        
                        # Parse details JSON if present
                        details_str = result_data.get('details', '{}')
                        
                        try:
                            import json
                            details = json.loads(details_str) if isinstance(details_str, str) else details_str
                            logger.info(f"🔍 details parsed: keys={list(details.keys())}")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to parse details JSON: {e} | raw={details_str[:200]}")
                            details = {}
                        
                        # Extract position data
                        symbol = result_data.get('symbol') or details.get('symbol')
                        side = details.get('side')  # "BUY" or "SELL"
                        filled_qty = details.get('filled_qty') or details.get('qty')
                        order_id = details.get('order_id', 'unknown')
                        order_status = details.get('order_status')
                        
                        logger.info(f"🔍 Extracted: symbol={symbol}, side={side}, qty={filled_qty}, status={order_status}")
                        
                        # Only track FILLED orders that open positions (not closes)
                        if not symbol or not side or not filled_qty:
                            logger.info(f"⏭️  SKIP: missing required fields (symbol={symbol}, side={side}, qty={filled_qty})")
                            continue
                        
                        if order_status != 'FILLED':
                            logger.info(f"⏭️  SKIP: order_status={order_status} (not FILLED, expected FILLED)")
                            continue
                        
                        # Check if this is a position close by looking at permit data
                        permit = details.get('permit', {})
                        if isinstance(permit, str):
                            try:
                                import json
                                permit = json.loads(permit)
                            except:
                                permit = {}
                        
                        # Skip if this is a close order (check permit for close actions)
                        reduce_only = details.get('reduceOnly') or details.get('reduce_only')
                        if reduce_only:
                            logger.info(f"⏭️  SKIP: reduceOnly={reduce_only} (close order, not position open)")
                            continue
                        
                        # Get entry price from Binance (more reliable than order price)
                        current_price = get_current_price(symbol)
                        if not current_price:
                            logger.warning(f"Cannot track {symbol} - price unavailable")
                            continue
                        
                        entry_price = current_price  # Use current market price as entry
                        
                        # Get TP/SL from original intent (should be included in result or fetched)
                        # For now, use basic calculation
                        tp = None
                        sl = None
                        
                        # TODO: Fetch actual TP/SL from AI Engine signal or ExitBrain
                        # For testnet protection, use simple fixed percentages
                        if side == "BUY":
                            tp = entry_price * 1.025  # +2.5%
                            sl = entry_price * 0.985  # -1.5%
                        else:  # SELL (SHORT)
                            tp = entry_price * 0.975  # -2.5%
                            sl = entry_price * 1.015  # +1.5%
                        
                        # Calculate quantity (use filled_qty from order)
                        quantity = float(filled_qty) if filled_qty else 0
                        
                        # Track position
                        position = TrackedPosition(
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            quantity=quantity,
                            leverage=float(details.get('leverage', 1.0)),
                            take_profit=tp,
                            stop_loss=sl,
                            order_id=str(order_id),
                            opened_at=result_data.get('timestamp', ''),
                            highest_price=entry_price,
                            lowest_price=entry_price
                        )
                        
                        tracked_positions[symbol] = position
                        stats["positions_tracked"] += 1
                        
                        logger.info(
                            f"📌 TRACKING NEW POSITION: {symbol} {side} | "
                            f"Entry=${entry_price:.4f} | "
                            f"TP=${tp:.4f} | SL=${sl:.4f} | "
                            f"Qty={quantity:.4f} | "
                            f"Order={order_id}"
                        )
            
            except asyncio.CancelledError:
                logger.info("🛑 Position listener cancelled")
                break
            except Exception as e:
                logger.error(f"❌ XREAD error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Retry delay
    
    except Exception as e:
        logger.error(f"❌ Position listener fatal error: {e}", exc_info=True)


async def exit_monitor_loop():
    """Monitor tracked positions and trigger exits"""
    logger.info("🔍 Starting exit monitor loop...")
    
    await asyncio.sleep(10)  # Wait for system to initialize
    
    while True:
        try:
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
                "opened_at": pos.opened_at
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
    global eventbus
    
    logger.info("🚀 Exit Monitor Service starting...")
    
    # Connect to EventBus
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    eventbus = EventBusClient(redis_url=redis_url)
    await eventbus.connect()
    logger.info(f"✅ EventBus connected: {redis_url}")
    
    # Start background tasks
    asyncio.create_task(position_listener())
    asyncio.create_task(exit_monitor_loop())
    
    logger.info("✅ Exit Monitor Service started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 Exit Monitor Service shutting down...")
    
    if eventbus:
        await eventbus.close()
    
    logger.info("✅ Shutdown complete")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", "8007"))
    uvicorn.run(app, host="0.0.0.0", port=port)
