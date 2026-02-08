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
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from decimal import Decimal
from dateutil import parser as date_parser

# P0.D.5: Backlog hardening configuration
INTENT_MAX_AGE_SEC = int(os.getenv("INTENT_MAX_AGE_SEC", "600"))  # 10 minutes default
XREADGROUP_COUNT = int(os.getenv("XREADGROUP_COUNT", "10"))  # Start conservative
EXEC_CONCURRENCY = int(os.getenv("EXEC_CONCURRENCY", "1"))  # Bounded concurrency

# P0 FIX: Import Redis for margin guard, rate limiting, and idempotency
import redis

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
        logging.FileHandler("/var/log/quantum/execution.log")
        # P2 FIX: Removed StreamHandler - systemd already logs stdout to file
        # This prevents duplicate log entries
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
        
        # P0.D.4d: Log before publishing
        logger.info(f"[P0.D.4d] Publishing execution result (market order) {result.symbol} status={result.status}")
        
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
    Execute REAL trade order on Binance Futures Testnet with P0 GUARDRAILS:
    
    P0.1: Margin Guard - Check available margin before placing order
    P0.2: Idempotency - Prevent duplicate order processing
    P0.3: Per-Symbol Lock - Prevent concurrent orders on same symbol+side
    P0.4: Rate Limiting - Max 1 order/30s per symbol, 5/min globally
    P0.5: Fail-Closed - Reject on any safety check failure
    """
    global stats
    
    # Initialize Redis client
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
    
    try:
        stats["orders_received"] += 1
        stats["last_order_time"] = datetime.utcnow().isoformat()
        
        # Generate stable trace_id for idempotency
        trace_id = f"{intent.symbol}_{intent.timestamp}"
        
        # ===================================================================
        # P0.2: IDEMPOTENCY CHECK
        # ===================================================================
        dedup_key = f"quantum:exec:processed:{trace_id}"
        
        # SETNX returns 1 if key was set (first time), 0 if already exists
        if not redis_client.setnx(dedup_key, "1"):
            logger.warning(
                f"🔁 IDEMPOTENCY_SKIP: {intent.symbol} {intent.side} "
                f"trace_id={trace_id} - already processed"
            )
            return
        
        # Set 1h TTL on dedup key
        redis_client.expire(dedup_key, 3600)
        
        # ===================================================================
        # P0.3: PER-SYMBOL INFLIGHT LOCK
        # ===================================================================
        lock_key = f"quantum:exec:lock:{intent.symbol}:{intent.side}"
        lock_acquired = redis_client.setnx(lock_key, "1")
        
        if not lock_acquired:
            logger.warning(
                f"🔒 LOCK_SKIP: {intent.symbol} {intent.side} - "
                f"concurrent order already processing"
            )
            return
        
        # Set 30s TTL on lock (auto-release if process dies)
        redis_client.expire(lock_key, 30)
        
        try:
            # =============================================================
            # P0.4: RATE LIMITING
            # =============================================================
            
            # Check per-symbol rate limit (max 1 per 30s)
            symbol_rate_key = f"quantum:exec:rate:symbol:{intent.symbol}"
            if redis_client.exists(symbol_rate_key):
                logger.warning(
                    f"⏱️ RATE_LIMIT_SKIP: {intent.symbol} - "
                    f"max 1 order per 30s per symbol"
                )
                return
            
            # Check global rate limit (max 5 per minute)
            global_rate_key = "quantum:exec:rate:global"
            current_count = redis_client.get(global_rate_key)
            
            if current_count and int(current_count) >= 5:
                logger.warning(
                    f"⏱️ GLOBAL_RATE_LIMIT: Max 5 orders per minute reached"
                )
                # Set cooldown
                redis_client.setex("quantum:safety:cooldown_global", 300, "1")
                return
            
            # =============================================================
            # P0.1: MARGIN GUARD (CRITICAL - FAIL-CLOSED)
            # =============================================================
            
            try:
                account = binance_client.futures_account()
                available_margin = float(account["availableBalance"])
                
                # P0.4C: Skip margin check for reduce_only closes
                # Close orders are risk-REDUCING and must NEVER be blocked by margin checks
                if getattr(intent, 'reduce_only', False):
                    logger.info(f"💰 MARGIN CHECK SKIPPED: {intent.symbol} {intent.action} (reduce_only=True)")
                # P0.4D: Skip margin check for exploration trades (SimpleCLM data collection)
                # Exploration trades are $50 exactly with 5x leverage - minimal risk for testnet
                elif intent.position_size_usd == 50.0 and intent.leverage == 5:
                    logger.info(f"💰 MARGIN CHECK SKIPPED: {intent.symbol} {intent.action} (exploration_trade=$50@5x for CLM data collection)")
                else:
                    # Calculate required margin
                    notional_value = intent.position_size_usd or 1000.0  # Default if not set
                    leverage_val = intent.leverage or 10.0  # Default if not set
                    required_margin = (notional_value / leverage_val) * 1.25  # 25% buffer
                    
                    logger.info(
                        f"💰 MARGIN CHECK: Available=${available_margin:.2f}, "
                        f"Required=${required_margin:.2f} (notional=${notional_value:.2f}, "
                        f"leverage={leverage_val}x, buffer=25%)"
                    )
                
                if not getattr(intent, 'reduce_only', False) and intent.position_size_usd != 50.0 and available_margin < required_margin:
                    logger.error(
                        f"❌ INSUFFICIENT MARGIN: {intent.symbol} {intent.action} "
                        f"Need=${required_margin:.2f}, Have=${available_margin:.2f}, "
                        f"Deficit=${required_margin - available_margin:.2f}"
                    )
                    
                    # Log terminal state
                    logger.info(
                        f"🚫 TERMINAL STATE: REJECTED_MARGIN | {intent.symbol} {intent.action} | "
                        f"Reason: Insufficient margin | trace_id={trace_id}"
                    )
                    
                    # Set cooldowns to prevent spam
                    redis_client.setex("quantum:safety:cooldown_global", 300, "1")
                    redis_client.setex(f"quantum:safety:cooldown_symbol:{intent.symbol}", 300, "1")
                    
                    # Check if this is repeated failure - trigger safe mode
                    margin_fail_key = "quantum:exec:margin_fails"
                    fails = redis_client.incr(margin_fail_key)
                    redis_client.expire(margin_fail_key, 300)  # 5 min window
                    
                    if fails >= 5:
                        logger.error(
                            f"🚨 SAFE MODE TRIGGERED: {fails} margin failures in 5 minutes"
                        )
                        redis_client.setex("quantum:safety:safe_mode", 900, "1")  # 15 min
                    
                    stats["orders_rejected"] += 1
                    return
                
            except Exception as margin_err:
                logger.error(f"❌ Margin check failed: {margin_err}", exc_info=True)
                stats["orders_rejected"] += 1
                return
            
            # =============================================================
            # BRIDGE-PATCH: RISK GOVERNOR (Fail-Closed)
            # =============================================================
            
            try:
                from services.risk_governor import get_risk_governor
                governor = get_risk_governor()
                
                # BRIDGE-PATCH: Extract sizing with explicit AI vs LEGACY source tracking
                # Safe parsing: treat None/empty as missing, fall back to legacy
                ai_size = intent.ai_size_usd if (hasattr(intent, 'ai_size_usd') and intent.ai_size_usd is not None) else None
                ai_lev = intent.ai_leverage if (hasattr(intent, 'ai_leverage') and intent.ai_leverage is not None) else None
                ai_harvest = intent.ai_harvest_policy if (hasattr(intent, 'ai_harvest_policy') and intent.ai_harvest_policy) else None
                
                # Determine sizing source
                size_source = "AI" if ai_size else "LEGACY"
                lev_source = "AI" if ai_lev else "LEGACY"
                harvest_source = "AI" if ai_harvest else "-"
                
                # Use AI fields if present, else fall back to legacy
                requested_size = ai_size if ai_size else (intent.position_size_usd or 100.0)
                requested_lev = ai_lev if ai_lev else (intent.leverage or 10.0)
                
                # Run governance policy
                approved, reason, gov_metadata = governor.evaluate(
                    symbol=intent.symbol,
                    action=intent.action,
                    confidence=intent.confidence,
                    position_size_usd=requested_size,
                    leverage=requested_lev,
                    risk_budget_usd=getattr(intent, 'risk_budget_usd', None)
                )
                
                if not approved:
                    logger.warning(
                        f"[GOVERNOR] ❌ Order rejected: {intent.symbol} {intent.action} | {reason}"
                    )
                    stats["orders_rejected"] += 1
                    return
                
                # Use clamped values from governor
                final_size = gov_metadata.get('clamped_size_usd', requested_size or 1000.0)
                final_lev = gov_metadata.get('clamped_leverage', requested_lev or 10.0)
                
                # BRIDGE-PATCH: Explicit source logging for audit trail
                harvest_str = harvest_source if harvest_source == "-" else f"{harvest_source}:{ai_harvest.get('mode', 'unknown') if ai_harvest else '-'}"
                logger.info(
                    f"[EXEC_INTENT] symbol={intent.symbol} action={intent.action} conf={intent.confidence:.2f} "
                    f"leverage={final_lev:.1f}x (src={lev_source}) size_usd={final_size:.0f} (src={size_source}) "
                    f"harvest={harvest_str} governor=PASS"
                )
                
            except Exception as gov_err:
                logger.error(f"[GOVERNOR] Failed to evaluate: {gov_err}", exc_info=True)
                # Fail-closed: reject if governor error
                stats["orders_rejected"] += 1
                return
                
            except Exception as margin_err:
                # FAIL-CLOSED: If we can't check margin, reject
                logger.error(
                    f"❌ MARGIN CHECK FAILED: {margin_err} - FAILING CLOSED"
                )
                logger.info(
                    f"🚫 TERMINAL STATE: REJECTED_SAFETY | {intent.symbol} {intent.side} | "
                    f"Reason: Cannot verify margin | trace_id={trace_id}"
                )
                stats["orders_rejected"] += 1
                return
            
            # =============================================================
            # PASSED ALL SAFETY CHECKS - UPDATE RATE LIMITERS
            # =============================================================
            
            # Set per-symbol rate limiter (30s TTL)
            redis_client.setex(symbol_rate_key, 30, "1")
            
            # Increment global rate counter (60s TTL)
            if not redis_client.exists(global_rate_key):
                redis_client.setex(global_rate_key, 60, "1")
            else:
                redis_client.incr(global_rate_key)
            
            logger.info(
                f"📥 TradeIntent APPROVED: {intent.symbol} {intent.side} "
                f"${intent.position_size_usd:.2f} @ ${intent.entry_price:.4f} "
                f"| Confidence={intent.confidence:.2%} | Leverage={intent.leverage}x | trace_id={trace_id}"
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
                
                # P0 FIX: Log terminal state for rejection (Phase 2)
                logger.info(
                    f"🚫 TERMINAL STATE: REJECTED | {intent.symbol} {intent.side} | "
                    f"Reason: {error_msg} | trace_id={trace_id}"
                )
                
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
                # P0.D.4d: Log before publishing
                logger.info(f"[P0.D.4d] Publishing execution result (zero quantity) {result.symbol} status={result.status}")
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
            
            # P0.D.4d: Log before publishing
            logger.info(f"[P0.D.4d] Publishing execution result (main path) {intent.symbol} status={result.status}")
            
            # 6. Publish result to Redis
            await eventbus.publish_execution(result)
            
            # P0.4C: Log CLOSE_EXECUTED for reduceOnly exits
            if getattr(intent, 'reduce_only', False):
                logger.info(
                    f"✅ CLOSE_EXECUTED: {intent.symbol} {intent.side} reduceOnly=True | "
                    f"OrderID={order_id} | Price=${execution_price:.4f} | Qty={quantity} | "
                    f"source={getattr(intent, 'source', 'unknown')} | "
                    f"reason={getattr(intent, 'reason', 'unknown')} | "
                    f"trace_id={trace_id}"
                )
            
            # P0 FIX: Log terminal state for watchdog monitoring (Phase 2)
            logger.info(
                f"✅ TERMINAL STATE: FILLED | {intent.symbol} {intent.side} | "
                f"OrderID={order_id} | trace_id={trace_id}"
            )
            
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
            
        finally:
            # G2: ALWAYS release inflight lock (only runs if lock_acquired=True)
            try:
                redis_client.delete(lock_key)
            except Exception as e:
                logger.warning(f"⚠️ INFLIGHT_CLEANUP_FAILED: {e}")
    
    except BinanceAPIException as e:
        logger.error(f"❌ Binance API error: {e}")
        
        # P0 FIX: Log terminal state for error (Phase 2)
        logger.info(
            f"🚫 TERMINAL STATE: FAILED | {intent.symbol} {intent.side} | "
            f"Reason: Binance API error | trace_id={trace_id}"
        )
        
        # P0.3: Release lock on error
        redis_client.delete(lock_key)
        stats["orders_rejected"] += 1
        
    except Exception as e:
        logger.error(f"❌ Execution error: {e}", exc_info=True)
        
        # P0 FIX: Log terminal state for error (Phase 2)
        logger.info(
            f"🚫 TERMINAL STATE: FAILED | {intent.symbol} {intent.side} | "
            f"Reason: Exception - {str(e)[:100]} | trace_id={trace_id}"
        )
        
        # P0.3: Release lock on error
        redis_client.delete(lock_key)
        stats["orders_rejected"] += 1


async def order_consumer():
    """Background task: consume approved orders from quantum:stream:trade.intent"""
    logger.info("🚀 Starting order consumer...")
    
    import socket
    import os
    
    # P0 FIX: Use consumer groups for reliability (Phase 2)
    group_name = "quantum:group:execution:trade.intent"
    consumer_name = f"execution-{socket.gethostname()}-{os.getpid()}"
    
    logger.info(f"📥 Consumer group: {group_name}")
    logger.info(f"📥 Consumer name: {consumer_name}")
    logger.info(f"📊 P0.D.5 Config: TTL={INTENT_MAX_AGE_SEC}s, COUNT={XREADGROUP_COUNT}, CONCURRENCY={EXEC_CONCURRENCY}")
    
    # P0.D.5: Concurrency semaphore (bounded parallelism)
    exec_semaphore = asyncio.Semaphore(EXEC_CONCURRENCY)
    stale_count = 0
    last_stale_log = 0
    
    try:
        async for signal_data in eventbus.subscribe_with_group(
            "quantum:stream:trade.intent",
            group_name=group_name,
            consumer_name=consumer_name,
            start_id=">",  # Only new messages
            create_group=True
        ):
            # Process trade intent message
            msg_id = signal_data.get('_message_id', 'unknown')
            symbol = signal_data.get('symbol', 'N/A')
            
            # Optional diagnostic logging (controlled by PIPELINE_DIAG env var)
            if os.getenv('PIPELINE_DIAG') == 'true':
                logger.info(f"[DIAG] Processing message {msg_id} symbol={symbol}")
            
            # P0.D.5: TTL check - drop stale intents
            intent_timestamp = signal_data.get('timestamp')
            if intent_timestamp:
                try:
                    intent_time = date_parser.isoparse(intent_timestamp)
                    if intent_time.tzinfo is None:
                        intent_time = intent_time.replace(tzinfo=timezone.utc)
                    age_seconds = (datetime.now(timezone.utc) - intent_time).total_seconds()
                    
                    if age_seconds > INTENT_MAX_AGE_SEC:
                        stale_count += 1
                        now = time.time()
                        # Rate-limited logging (max 1 per 30s)
                        if now - last_stale_log > 30:
                            logger.warning(f"⏰ STALE_INTENT_DROP: {symbol} age={int(age_seconds)}s (>{INTENT_MAX_AGE_SEC}s) | Total stale: {stale_count}")
                            last_stale_log = now
                        # ACK without execution (intent too old)
                        if '_stream_name' in signal_data and '_group_name' in signal_data:
                            try:
                                await eventbus.redis.xack(
                                    signal_data['_stream_name'],
                                    signal_data['_group_name'],
                                    msg_id
                                )
                            except Exception as ack_err:
                                logger.error(f"❌ Failed to ACK stale message {msg_id}: {ack_err}")
                        continue
                except Exception as parse_err:
                    logger.warning(f"⚠️ Failed to parse timestamp for TTL check: {parse_err}")
            
            # Remove EventBus metadata
            signal_data = {k: v for k, v in signal_data.items() if not k.startswith('_')}
            
            # Schema normalization: convert 'side' to 'action' if needed
            if 'side' in signal_data and 'action' not in signal_data:
                signal_data['action'] = signal_data['side']
                del signal_data['side']
            
            # Filter signal_data to only include TradeIntent fields
            # BRIDGE-PATCH: Added ai_size_usd, ai_leverage, ai_harvest_policy for AI-driven sizing
            # P0.4C: Added reason, reduce_only for exit flow audit trail
            allowed_fields = {
                'symbol', 'action', 'confidence', 'position_size_usd', 'leverage', 
                'timestamp', 'source', 'stop_loss_pct', 'take_profit_pct', 
                'entry_price', 'stop_loss', 'take_profit', 'quantity',
                'ai_size_usd', 'ai_leverage', 'ai_harvest_policy',  # BRIDGE-PATCH v1.1
                'reason', 'reduce_only'  # P0.4C exit flow
            }
            filtered_data = {k: v for k, v in signal_data.items() if k in allowed_fields}
            
            # Add default 'source' if missing
            if 'source' not in filtered_data:
                filtered_data['source'] = 'trading-bot'
            
            # Parse as TradeIntent (AI Engine schema)
            try:
                intent = TradeIntent(**filtered_data)
                
                # P0.4C: Schema guard - warn if reduce_only without full context
                if getattr(intent, 'reduce_only', False):
                    if not getattr(intent, 'source', None) or not getattr(intent, 'reason', None):
                        logger.warning(
                            f"⚠️ SCHEMA_GUARD: {symbol} has reduce_only=True but missing source or reason | "
                            f"source={getattr(intent, 'source', None)} reason={getattr(intent, 'reason', None)}"
                        )
            except Exception as parse_err:
                logger.error(f"❌ Failed to parse TradeIntent for {symbol}: {parse_err}")
                if os.getenv('PIPELINE_DIAG') == 'true':
                    logger.error(f"[DIAG] Filtered data: {filtered_data}")
                continue
            
            # P0.D.5: Execute with concurrency control
            if EXEC_CONCURRENCY > 1:
                async with exec_semaphore:
                    await execute_order_from_intent(intent)
            else:
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
