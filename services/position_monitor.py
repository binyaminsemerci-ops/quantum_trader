#!/usr/bin/env python3
"""
Position Monitor - Track open positions and PnL
================================================
Port: 8004
Subscribes: trade.execution.res
Publishes: trade.position.update (every 30s)

Tracks:
- Open positions
- Unrealized PnL
- Realized PnL
- Total exposure
- Win rate

Author: Quantum Trader Team
Date: 2026-01-12
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    ExecutionResult,
    PositionUpdate
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("/var/log/quantum/position-monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Position:
    """Open position"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: str
    size_usd: float
    leverage: float
    order_id: str
    realized_pnl: float = 0.0
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.side == "LONG":
            return (current_price - self.entry_price) / self.entry_price * self.size_usd
        else:  # SHORT
            return (self.entry_price - current_price) / self.entry_price * self.size_usd


@dataclass
class Portfolio:
    """Portfolio state"""
    initial_balance: float = 10000.0
    current_balance: float = 10000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Dict[str, Any]] = field(default_factory=list)
    total_realized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    
    def get_total_exposure(self) -> float:
        """Get total USD exposure across all positions"""
        return sum(pos.size_usd for pos in self.positions.values())
    
    def get_total_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Get total unrealized PnL"""
        total = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                total += pos.calculate_unrealized_pnl(prices[symbol])
        return total
    
    def get_win_rate(self, last_n: int = 20) -> float:
        """Get win rate from recent closed positions"""
        if not self.closed_positions:
            return 0.0
        
        recent = self.closed_positions[-last_n:]
        wins = sum(1 for p in recent if p.get("pnl", 0) > 0)
        return wins / len(recent) if recent else 0.0


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Position Monitor", version="1.0.0")

# Global state
eventbus: Optional[EventBusClient] = None
portfolio = Portfolio()
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
    open_positions: int
    total_exposure_usd: float
    unrealized_pnl: float


class PortfolioResponse(BaseModel):
    initial_balance: float
    current_balance: float
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_pnl: float
    total_return_pct: float
    open_positions: int
    total_exposure_usd: float
    exposure_pct: float
    win_rate: float
    total_fees_paid: float


class PositionResponse(BaseModel):
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size_usd: float
    leverage: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: str
    order_id: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    unrealized_pnl = portfolio.get_total_unrealized_pnl(market_prices)
    
    return HealthResponse(
        status="healthy" if eventbus else "degraded",
        uptime_seconds=(datetime.utcnow() - start_time).total_seconds(),
        open_positions=len(portfolio.positions),
        total_exposure_usd=portfolio.get_total_exposure(),
        unrealized_pnl=unrealized_pnl
    )


@app.get("/portfolio")
async def get_portfolio():
    """Get portfolio summary"""
    unrealized_pnl = portfolio.get_total_unrealized_pnl(market_prices)
    total_pnl = portfolio.total_realized_pnl + unrealized_pnl
    total_return_pct = (total_pnl / portfolio.initial_balance) * 100
    exposure = portfolio.get_total_exposure()
    exposure_pct = (exposure / portfolio.current_balance) * 100
    
    return PortfolioResponse(
        initial_balance=portfolio.initial_balance,
        current_balance=portfolio.current_balance,
        total_realized_pnl=portfolio.total_realized_pnl,
        total_unrealized_pnl=unrealized_pnl,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        open_positions=len(portfolio.positions),
        total_exposure_usd=exposure,
        exposure_pct=exposure_pct,
        win_rate=portfolio.get_win_rate(),
        total_fees_paid=portfolio.total_fees_paid
    )


@app.get("/positions")
async def get_positions():
    """Get all open positions"""
    positions = []
    
    for symbol, pos in portfolio.positions.items():
        current_price = market_prices.get(symbol, pos.entry_price)
        unrealized_pnl = pos.calculate_unrealized_pnl(current_price)
        unrealized_pnl_pct = (unrealized_pnl / pos.size_usd) * 100
        
        positions.append(PositionResponse(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            current_price=current_price,
            size_usd=pos.size_usd,
            leverage=pos.leverage,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            entry_time=pos.entry_time,
            order_id=pos.order_id
        ))
    
    return positions


@app.get("/prices")
async def get_prices():
    """Get current market prices"""
    return market_prices


@app.post("/prices/{symbol}")
async def update_price(symbol: str, price: float):
    """Update market price (for testing)"""
    market_prices[symbol] = price
    return {"symbol": symbol, "price": price}


@app.post("/close/{symbol}")
async def close_position(symbol: str):
    """Manually close position (for testing)"""
    if symbol not in portfolio.positions:
        return {"error": "Position not found"}
    
    pos = portfolio.positions[symbol]
    current_price = market_prices.get(symbol, pos.entry_price)
    pnl = pos.calculate_unrealized_pnl(current_price)
    
    # Close position
    del portfolio.positions[symbol]
    portfolio.total_realized_pnl += pnl
    portfolio.current_balance += pnl
    
    # Record closed position
    portfolio.closed_positions.append({
        "symbol": symbol,
        "side": pos.side,
        "entry_price": pos.entry_price,
        "exit_price": current_price,
        "size_usd": pos.size_usd,
        "pnl": pnl,
        "pnl_pct": (pnl / pos.size_usd) * 100,
        "closed_time": datetime.utcnow().isoformat()
    })
    
    logger.info(f"🔴 Closed {symbol}: PnL=${pnl:.2f}")
    
    return {
        "symbol": symbol,
        "pnl": pnl,
        "pnl_pct": (pnl / pos.size_usd) * 100
    }


# ============================================================================
# CORE LOGIC
# ============================================================================

async def process_execution(result_data: Dict[str, Any]):
    """
    Process execution result
    
    1. Create/update position
    2. Deduct fees
    3. Update balance
    """
    try:
        # Remove EventBus metadata
        result_data = {k: v for k, v in result_data.items() if not k.startswith('_')}
        result = ExecutionResult(**result_data)
        
        logger.info(
            f"📥 Execution result: {result.symbol} {result.action} "
            f"@ ${result.entry_price:.2f}"
        )
        
        # Deduct fees
        portfolio.total_fees_paid += result.fee_usd
        portfolio.current_balance -= result.fee_usd
        
        # Determine position side
        side = "LONG" if result.action == "BUY" else "SHORT"
        
        # Check if position already exists
        if result.symbol in portfolio.positions:
            # Update existing position (average entry price)
            existing = portfolio.positions[result.symbol]
            total_size = existing.size_usd + result.position_size_usd
            avg_price = (
                (existing.entry_price * existing.size_usd + 
                 result.entry_price * result.position_size_usd) / total_size
            )
            existing.entry_price = avg_price
            existing.size_usd = total_size
            
            logger.info(
                f"📊 Updated position: {result.symbol} | "
                f"Avg Price=${avg_price:.2f} | "
                f"Size=${total_size:.2f}"
            )
        else:
            # Create new position
            position = Position(
                symbol=result.symbol,
                side=side,
                entry_price=result.entry_price,
                entry_time=result.timestamp,
                size_usd=result.position_size_usd,
                leverage=result.leverage,
                order_id=result.order_id
            )
            portfolio.positions[result.symbol] = position
            
            logger.info(
                f"✅ Opened position: {result.symbol} {side} | "
                f"Entry=${result.entry_price:.2f} | "
                f"Size=${result.position_size_usd:.2f}"
            )
    
    except Exception as e:
        logger.error(f"❌ Error processing execution: {e}", exc_info=True)


async def execution_consumer():
    """Background task: consume execution results"""
    logger.info("🚀 Starting execution consumer...")
    
    try:
        async for result_data in eventbus.subscribe("trade.execution.res"):
            await process_execution(result_data)
    except asyncio.CancelledError:
        logger.info("🛑 Execution consumer cancelled")
    except Exception as e:
        logger.error(f"❌ Execution consumer error: {e}", exc_info=True)


async def position_publisher():
    """Background task: publish position updates every 30s"""
    logger.info("🚀 Starting position publisher...")
    
    try:
        while True:
            await asyncio.sleep(30)
            
            # Publish update for each open position
            for symbol, pos in portfolio.positions.items():
                current_price = market_prices.get(symbol, pos.entry_price)
                unrealized_pnl = pos.calculate_unrealized_pnl(current_price)
                
                update = PositionUpdate(
                    symbol=symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    current_price=current_price,
                    size_usd=pos.size_usd,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
                
                await eventbus.publish_position(update)
                
                logger.debug(
                    f"📊 Position update: {symbol} | "
                    f"PnL=${unrealized_pnl:.2f}"
                )
    
    except asyncio.CancelledError:
        logger.info("🛑 Position publisher cancelled")


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
    
    logger.info("🚀 Position Monitor starting...")
    
    # Initialize EventBus
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    eventbus = EventBusClient(redis_url=redis_url)
    await eventbus.connect()
    logger.info(f"✅ EventBus connected: {redis_url}")
    
    # Start execution consumer
    asyncio.create_task(execution_consumer())
    logger.info("✅ Execution consumer started")
    
    # Start position publisher
    asyncio.create_task(position_publisher())
    logger.info("✅ Position publisher started")
    
    # Start price updater
    asyncio.create_task(price_updater())
    logger.info("✅ Price updater started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 Position Monitor shutting down...")
    
    if eventbus:
        await eventbus.disconnect()
    
    logger.info("✅ Shutdown complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run service"""
    port = int(os.getenv("SERVICE_PORT", "8005"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
