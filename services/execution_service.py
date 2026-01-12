#!/usr/bin/env python3
"""
Execution Service - Paper-mode trade execution
===============================================
Port: 8002
Subscribes: trade.signal.safe
Publishes: trade.execution.res

Simulates order execution with:
- Virtual order book
- Slippage simulation (0.05% avg)
- Fee calculation (0.04% taker fee)
- Order ID generation

Author: Quantum Trader Team
Date: 2026-01-12
"""
import os
import sys
import asyncio
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    RiskApprovedSignal,
    ExecutionResult
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("/var/log/quantum/execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Execution Service", version="1.0.0")

# Global state
eventbus: Optional[EventBusClient] = None
stats = {
    "orders_received": 0,
    "orders_filled": 0,
    "orders_rejected": 0,
    "total_volume_usd": 0.0,
    "total_fees_usd": 0.0,
    "avg_slippage_pct": 0.0,
    "last_order_time": None
}

# Mock market prices (updated periodically)
market_prices = {
    "BTCUSDT": 95000.0,
    "ETHUSDT": 3400.0,
    "BNBUSDT": 680.0,
    "SOLUSDT": 185.0,
    "XRPUSDT": 2.45
}


# ============================================================================
# MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    orders_received: int
    orders_filled: int
    orders_rejected: int
    fill_rate: float


class StatsResponse(BaseModel):
    total_volume_usd: float
    total_fees_usd: float
    avg_slippage_pct: float
    orders_filled: int
    orders_rejected: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    fill_rate = (
        stats["orders_filled"] / stats["orders_received"]
        if stats["orders_received"] > 0
        else 0.0
    )
    
    return HealthResponse(
        status="healthy" if eventbus else "degraded",
        uptime_seconds=(datetime.utcnow() - start_time).total_seconds(),
        orders_received=stats["orders_received"],
        orders_filled=stats["orders_filled"],
        orders_rejected=stats["orders_rejected"],
        fill_rate=fill_rate
    )


@app.get("/stats")
async def get_stats():
    """Get execution stats"""
    return StatsResponse(
        total_volume_usd=stats["total_volume_usd"],
        total_fees_usd=stats["total_fees_usd"],
        avg_slippage_pct=stats["avg_slippage_pct"],
        orders_filled=stats["orders_filled"],
        orders_rejected=stats["orders_rejected"]
    )


@app.get("/prices")
async def get_prices():
    """Get current market prices"""
    return market_prices


@app.post("/prices/{symbol}")
async def update_price(symbol: str, price: float):
    """Update market price (for testing)"""
    market_prices[symbol] = price
    return {"symbol": symbol, "price": price}


@app.post("/reset")
async def reset_stats():
    """Reset statistics"""
    stats["orders_received"] = 0
    stats["orders_filled"] = 0
    stats["orders_rejected"] = 0
    stats["total_volume_usd"] = 0.0
    stats["total_fees_usd"] = 0.0
    stats["avg_slippage_pct"] = 0.0
    
    return {"status": "reset", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# CORE LOGIC
# ============================================================================

def get_market_price(symbol: str) -> Optional[float]:
    """Get current market price for symbol"""
    return market_prices.get(symbol)


def simulate_slippage(base_price: float, action: str, volatility: float = 0.0005) -> tuple[float, float]:
    """
    Simulate order slippage
    
    Args:
        base_price: Market price
        action: BUY or SELL
        volatility: Base volatility (0.05%)
    
    Returns:
        (execution_price, slippage_pct)
    """
    import random
    
    # Random slippage between 0-0.1%
    slippage_pct = random.uniform(0, 0.001)
    
    if action == "BUY":
        # Buying - price slips UP
        execution_price = base_price * (1 + slippage_pct)
    else:  # SELL
        # Selling - price slips DOWN
        execution_price = base_price * (1 - slippage_pct)
    
    return execution_price, slippage_pct


def calculate_fee(volume_usd: float, fee_rate: float = 0.0004) -> float:
    """
    Calculate trading fee
    
    Args:
        volume_usd: Order volume in USD
        fee_rate: Fee rate (0.04% default - Binance taker fee)
    
    Returns:
        Fee in USD
    """
    return volume_usd * fee_rate


async def execute_order(signal: RiskApprovedSignal):
    """
    Execute trade order (paper mode)
    
    1. Get market price
    2. Simulate slippage
    3. Calculate fee
    4. Generate order ID
    5. Publish execution result
    """
    global stats
    
    try:
        stats["orders_received"] += 1
        stats["last_order_time"] = datetime.utcnow().isoformat()
        
        logger.info(
            f"📥 Order received: {signal.symbol} {signal.action} "
            f"${signal.position_size_usd:.2f}"
        )
        
        # Get market price
        market_price = get_market_price(signal.symbol)
        if not market_price:
            logger.error(f"❌ No market price for {signal.symbol}")
            stats["orders_rejected"] += 1
            return
        
        # Simulate slippage
        execution_price, slippage_pct = simulate_slippage(
            market_price,
            signal.action
        )
        
        # Calculate fee
        fee_usd = calculate_fee(signal.position_size_usd)
        
        # Generate order ID
        order_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
        
        # Determine leverage (use 1x for now)
        leverage = 1.0
        
        # Create execution result
        result = ExecutionResult(
            symbol=signal.symbol,
            action=signal.action,
            entry_price=execution_price,
            position_size_usd=signal.position_size_usd,
            leverage=leverage,
            timestamp=datetime.utcnow().isoformat() + "Z",
            order_id=order_id,
            status="filled",
            slippage_pct=slippage_pct,
            fee_usd=fee_usd
        )
        
        # Publish result
        await eventbus.publish_execution(result)
        
        # Update stats
        stats["orders_filled"] += 1
        stats["total_volume_usd"] += signal.position_size_usd
        stats["total_fees_usd"] += fee_usd
        
        # Update avg slippage (running average)
        n = stats["orders_filled"]
        stats["avg_slippage_pct"] = (
            (stats["avg_slippage_pct"] * (n - 1) + slippage_pct) / n
        )
        
        logger.info(
            f"✅ FILLED: {signal.symbol} {signal.action} | "
            f"Price=${execution_price:.2f} | "
            f"Size=${signal.position_size_usd:.2f} | "
            f"Slippage={slippage_pct:.4%} | "
            f"Fee=${fee_usd:.2f} | "
            f"Order={order_id}"
        )
    
    except Exception as e:
        logger.error(f"❌ Execution error: {e}", exc_info=True)
        stats["orders_rejected"] += 1


async def order_consumer():
    """Background task: consume approved orders from trade.signal.safe"""
    logger.info("🚀 Starting order consumer...")
    
    try:
        async for signal_data in eventbus.subscribe("trade.signal.safe"):
            # Remove EventBus metadata
            signal_data = {k: v for k, v in signal_data.items() if not k.startswith('_')}
            # Parse signal
            signal = RiskApprovedSignal(**signal_data)
            await execute_order(signal)
    except asyncio.CancelledError:
        logger.info("🛑 Order consumer cancelled")
    except Exception as e:
        logger.error(f"❌ Order consumer error: {e}", exc_info=True)


async def price_updater():
    """Background task: update market prices every 5 seconds"""
    import random
    
    logger.info("🚀 Starting price updater...")
    
    try:
        while True:
            # Simulate price movements
            for symbol in market_prices:
                # Random walk: ±0.1% per 5 seconds
                change_pct = random.uniform(-0.001, 0.001)
                market_prices[symbol] *= (1 + change_pct)
            
            logger.debug(f"📊 Updated prices: {market_prices}")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("🛑 Price updater cancelled")


# ============================================================================
# LIFECYCLE
# ============================================================================

start_time = datetime.utcnow()


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global eventbus
    
    logger.info("🚀 Execution Service starting...")
    
    # Initialize EventBus
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    eventbus = EventBusClient(redis_url=redis_url)
    await eventbus.connect()
    logger.info(f"✅ EventBus connected: {redis_url}")
    
    # Start order consumer
    asyncio.create_task(order_consumer())
    logger.info("✅ Order consumer started")
    
    # Start price updater
    asyncio.create_task(price_updater())
    logger.info("✅ Price updater started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 Execution Service shutting down...")
    
    if eventbus:
        await eventbus.disconnect()
    
    logger.info("✅ Shutdown complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run service"""
    port = int(os.getenv("SERVICE_PORT", "8002"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
