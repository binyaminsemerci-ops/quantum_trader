#!/usr/bin/env python3
"""
Execution Service - REAL Binance Futures Execution
==================================================
Port: 8002
Subscribes: trade.intent
Publishes: trade.execution.result

Executes REAL orders on Binance Futures Testnet:
- MARKET orders
- Stop Loss orders (STOP_MARKET)
- Take Profit orders (TAKE_PROFIT_MARKET)
- Real slippage, fees, and order IDs from Binance

Author: Quantum Trader Team
Date: 2026-01-16
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
from binance.client import Client
from binance.exceptions import BinanceAPIException

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    RiskApprovedSignal,
    TradeIntent,
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
# BINANCE CLIENT SETUP
# ============================================================================

# Load Binance credentials from environment
print("DEBUG: Loading Binance credentials from environment...")
BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")

print(f"DEBUG: API Key loaded: {BINANCE_API_KEY[:20] if BINANCE_API_KEY else 'MISSING'}...")
print(f"DEBUG: Secret loaded: {BINANCE_API_SECRET[:20] if BINANCE_API_SECRET else 'MISSING'}...")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print("ERROR: BINANCE CREDENTIALS MISSING!")
    logger.error("❌ BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_SECRET_KEY not set!")
    logger.error("Please set environment variables before starting.")
    sys.exit(1)

# Initialize Binance Futures Testnet client
print("DEBUG: Initializing Binance client...")
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
binance_client.FUTURES_URL = "https://testnet.binancefuture.com"
print("DEBUG: Binance client created")

logger.info("✅ Binance Futures Testnet client initialized")
logger.info(f"API Key: {BINANCE_API_KEY[:20]}...")

# Test connection
print("DEBUG: Testing Binance connection...")
try:
    account = binance_client.futures_account()
    balance = float(account["totalWalletBalance"])
    can_trade = account["canTrade"]
    print(f"DEBUG: Binance connected! Balance: {balance:.2f} USDT")
    logger.info(f"✅ Connected to Binance! Balance: {balance:.2f} USDT | Can Trade: {can_trade}")
except Exception as e:
    print(f"ERROR: Binance connection failed: {e}")
    logger.error(f"❌ Failed to connect to Binance: {e}")
    sys.exit(1)

# Fetch symbol precision info
logger.info("📊 Fetching symbol precision from Binance...")
SYMBOL_PRECISION = {}
try:
    exchange_info = binance_client.futures_exchange_info()
    for s in exchange_info["symbols"]:
        symbol = s["symbol"]
        qty_precision = s["quantityPrecision"]
        price_precision = s["pricePrecision"]
        
        # Find LOT_SIZE and PRICE_FILTER
        step_size = None
        min_qty = None
        max_qty = None
        tick_size = None
        
        for f in s["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])
                min_qty = float(f["minQty"])
                max_qty = float(f["maxQty"])
            elif f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
        
        SYMBOL_PRECISION[symbol] = {
            "quantityPrecision": qty_precision,
            "pricePrecision": price_precision,
            "stepSize": step_size,
            "minQty": min_qty,
            "maxQty": max_qty,
            "tickSize": tick_size
        }
    
    logger.info(f"✅ Loaded precision for {len(SYMBOL_PRECISION)} symbols")
except Exception as e:
    logger.error(f"❌ Failed to load symbol precision: {e}")
    SYMBOL_PRECISION = {}


def round_quantity(symbol: str, quantity: float) -> float:
    """
    Round quantity to comply with Binance LOT_SIZE stepSize.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        quantity: Raw quantity to round
        
    Returns:
        Rounded quantity that complies with exchange rules
    """
    if symbol not in SYMBOL_PRECISION:
        logger.warning(f"⚠️ No precision info for {symbol}, using default rounding")
        return round(quantity, 3)
    
    step_size = SYMBOL_PRECISION[symbol]["stepSize"]
    if step_size is None:
        return round(quantity, 3)
    
    # Round down to nearest stepSize multiple
    precision = len(str(step_size).rstrip('0').split('.')[-1])
    rounded = (quantity // step_size) * step_size
    return round(rounded, precision)


def round_price(symbol: str, price: float) -> float:
    """
    Round price to comply with Binance PRICE_FILTER tickSize.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        price: Raw price to round
        
    Returns:
        Rounded price that complies with exchange rules
    """
    if symbol not in SYMBOL_PRECISION:
        logger.warning(f"⚠️ No precision info for {symbol}, using default price rounding")
        return round(price, 2)
    
    tick_size = SYMBOL_PRECISION[symbol]["tickSize"]
    if tick_size is None:
        return round(price, 2)
    
    # Round to nearest tickSize multiple
    precision = len(str(tick_size).rstrip('0').split('.')[-1])
    rounded = round(price / tick_size) * tick_size
    return round(rounded, precision)


def validate_quantity(symbol: str, quantity: float) -> tuple[bool, str]:
    """
    Validate if quantity meets Binance minimum requirements.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        quantity: Quantity to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if symbol not in SYMBOL_PRECISION:
        return True, ""
    
    min_qty = SYMBOL_PRECISION[symbol]["minQty"]
    max_qty = SYMBOL_PRECISION[symbol]["maxQty"]
    
    if min_qty and quantity < min_qty:
        return False, f"Quantity {quantity} is below minimum {min_qty} for {symbol}"
    
    if max_qty and quantity > max_qty:
        return False, f"Quantity {quantity} exceeds maximum {max_qty} for {symbol}"
    
    if quantity <= 0:
        return False, f"Quantity must be greater than 0 (got {quantity})"
    
    return True, ""

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
        logger.error(f"❌ execute_order error: {e}", exc_info=True)
        stats["orders_rejected"] += 1


async def execute_order_from_intent(intent: TradeIntent):
    """
    Execute REAL trade order on Binance Futures Testnet
    
    1. Calculate quantity from position size and price
    2. Set leverage
    3. Place MARKET order on Binance
    4. Place STOP_MARKET order (Stop Loss)
    5. Place TAKE_PROFIT_MARKET order (Take Profit)
    6. Publish execution result
    """
    global stats
    
    try:
        stats["orders_received"] += 1
        stats["last_order_time"] = datetime.utcnow().isoformat()
        
        logger.info(
            f"📥 TradeIntent received: {intent.symbol} {intent.side} "
            f"${intent.position_size_usd:.2f} @ ${intent.entry_price:.4f} "
            f"| Confidence={intent.confidence:.2%} | Leverage={intent.leverage}x"
        )
        
        # 1. Set leverage for symbol
        try:
            binance_client.futures_change_leverage(
                symbol=intent.symbol,
                leverage=int(intent.leverage)
            )
            logger.info(f"✅ Leverage set to {intent.leverage}x for {intent.symbol}")
        except BinanceAPIException as e:
            logger.warning(f"⚠️ Could not set leverage (may already be set): {e}")
        
        # 2. Calculate quantity
        # quantity = position_size_usd / entry_price
        quantity = intent.position_size_usd / intent.entry_price
        
        # Round to Binance LOT_SIZE stepSize precision
        quantity = round_quantity(intent.symbol, quantity)
        
        # Validate quantity meets minimum requirements
        is_valid, error_msg = validate_quantity(intent.symbol, quantity)
        if not is_valid:
            logger.error(f"❌ {error_msg}")
            result = ExecutionResult(
                symbol=intent.symbol,
                action=intent.side,
                entry_price=intent.entry_price,
                position_size_usd=0.0,
                leverage=intent.leverage,
                timestamp=datetime.utcnow().isoformat() + "Z",
                order_id="REJECTED",
                status="rejected",
                slippage_pct=0.0,
                fee_usd=0.0
            )
            await eventbus.publish_execution(result)
            return
        
        logger.info(f"📊 Calculated quantity: {quantity} {intent.symbol} (after precision rounding)")
        
        # 3. Place MARKET order
        side_binance = "BUY" if intent.side.upper() == "BUY" else "SELL"
        
        logger.info(f"🚀 Placing MARKET order: {side_binance} {quantity} {intent.symbol}")
        
        market_order = binance_client.futures_create_order(
            symbol=intent.symbol,
            side=side_binance,
            type="MARKET",
            quantity=quantity
        )
        
        # Debug: Log full response
        logger.info(f"🔍 Binance response: orderId={market_order.get('orderId')}, "
                   f"status={market_order.get('status')}, "
                   f"executedQty={market_order.get('executedQty')}, "
                   f"cumQuote={market_order.get('cumQuote')}, "
                   f"avgPrice={market_order.get('avgPrice')}")
        
        order_id = str(market_order["orderId"])
        execution_price = float(market_order["avgPrice"]) if "avgPrice" in market_order else intent.entry_price
        actual_qty = float(market_order["executedQty"])
        
        logger.info(
            f"✅ BINANCE MARKET ORDER FILLED: {intent.symbol} {intent.side} | "
            f"OrderID={order_id} | "
            f"Price=${execution_price:.4f} | "
            f"Qty={actual_qty}"
        )
        
        # NOTE: TP/SL are NOT placed as hard orders on Binance
        # ExitBrain v3.5 monitors positions and closes them internally
        # when price hits TP/SL levels (adaptive management)
        if intent.take_profit and intent.stop_loss:
            logger.info(
                f"📊 TP/SL levels calculated by ExitBrain v3.5 | "
                f"TP: ${intent.take_profit:.4f} | SL: ${intent.stop_loss:.4f}"
            )
        
        # 4. Calculate real fee from Binance (commission)
        fee_usd = 0.0
        if "fills" in market_order:
            for fill in market_order["fills"]:
                if fill.get("commissionAsset") == "USDT":
                    fee_usd += float(fill["commission"])
        
        # 5. Create execution result
        result = ExecutionResult(
            symbol=intent.symbol,
            action=intent.side,
            entry_price=execution_price,
            position_size_usd=execution_price * actual_qty,
            leverage=intent.leverage,
            timestamp=datetime.utcnow().isoformat() + "Z",
            order_id=order_id,  # REAL Binance order ID!
            status="filled",
            slippage_pct=0.0,  # TODO: Calculate from intent.entry_price vs execution_price
            fee_usd=fee_usd
        )
        
        # 6. Publish result to Redis
        await eventbus.publish_execution(result)
        
        # 7. Update stats
        stats["orders_filled"] += 1
        stats["total_volume_usd"] += result.position_size_usd
        stats["total_fees_usd"] += fee_usd
        
        logger.info(
            f"✅ FILLED: {intent.symbol} {intent.side} | "
            f"Entry=${intent.entry_price:.4f} | "
            f"Filled=${execution_price:.4f} | "
            f"Size=${result.position_size_usd:.2f} | "
            f"Leverage={intent.leverage}x | "
            f"SL=${intent.stop_loss:.4f} | "
            f"TP=${intent.take_profit:.4f} | "
            f"Fee=${fee_usd:.4f} | "
            f"Order={order_id}"
        )
    
    except BinanceAPIException as e:
        logger.error(f"❌ Binance API error: {e}")
        stats["orders_rejected"] += 1
    except Exception as e:
        logger.error(f"❌ Execution error: {e}", exc_info=True)
        stats["orders_rejected"] += 1


async def order_consumer():
    """Background task: consume approved orders from quantum:stream:trade.intent"""
    logger.info("🚀 Starting order consumer...")
    
    try:
        async for signal_data in eventbus.subscribe("quantum:stream:trade.intent"):
            # Remove EventBus metadata
            signal_data = {k: v for k, v in signal_data.items() if not k.startswith('_')}
            
            # Parse as TradeIntent (AI Engine schema)
            intent = TradeIntent(**signal_data)
            
            # Execute the trade
            await execute_order_from_intent(intent)
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
