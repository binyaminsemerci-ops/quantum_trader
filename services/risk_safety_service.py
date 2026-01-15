#!/usr/bin/env python3
"""
Risk Safety Service - Validates trade signals using Governer logic
===================================================================
Port: 8003
Subscribes: trade.signal.v5
Publishes: trade.signal.safe

Applies:
- Confidence threshold (>= 0.65)
- Position size limits (<= 10%)
- Daily trade limits (<= 20)
- Circuit breakers (drawdown, exposure)

Author: Quantum Trader Team
Date: 2026-01-12
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    TradeSignal,
    RiskApprovedSignal
)
from ai_engine.agents.governer_agent import GovernerAgent, RiskConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("/var/log/quantum/risk-safety.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Risk Safety Service", version="1.0.0")

# Global state
governer: Optional[GovernerAgent] = None
eventbus: Optional[EventBusClient] = None
stats = {
    "signals_received": 0,
    "signals_approved": 0,
    "signals_rejected": 0,
    "rejection_reasons": {},
    "last_signal_time": None
}


# ============================================================================
# MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    signals_received: int
    signals_approved: int
    signals_rejected: int
    approval_rate: float


class StatsResponse(BaseModel):
    governer_balance: float
    governer_peak: float
    governer_drawdown: float
    total_trades: int
    recent_win_rate: float
    rejection_reasons: Dict[str, int]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    approval_rate = (
        stats["signals_approved"] / stats["signals_received"]
        if stats["signals_received"] > 0
        else 0.0
    )
    
    return HealthResponse(
        status="healthy" if eventbus and governer else "degraded",
        uptime_seconds=(datetime.utcnow() - start_time).total_seconds(),
        signals_received=stats["signals_received"],
        signals_approved=stats["signals_approved"],
        signals_rejected=stats["signals_rejected"],
        approval_rate=approval_rate
    )


@app.get("/stats")
async def get_stats():
    """Get governer stats"""
    if not governer:
        raise HTTPException(status_code=503, detail="Governer not initialized")
    
    governer_stats = governer.get_stats()
    
    return StatsResponse(
        governer_balance=governer_stats["balance"],
        governer_peak=governer_stats["peak_balance"],
        governer_drawdown=governer_stats["drawdown_pct"],
        total_trades=governer_stats["total_trades"],
        recent_win_rate=governer_stats["recent_win_rate"],
        rejection_reasons=stats["rejection_reasons"]
    )


@app.post("/reset")
async def reset_stats():
    """Reset statistics"""
    stats["signals_received"] = 0
    stats["signals_approved"] = 0
    stats["signals_rejected"] = 0
    stats["rejection_reasons"] = {}
    
    return {"status": "reset", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# CORE LOGIC
# ============================================================================

async def process_signal(signal_data: Dict[str, Any]):
    """
    Process incoming trade signal
    
    1. Parse signal
    2. Apply governer logic
    3. Publish approved signals
    4. Update stats
    """
    global stats
    
    try:
        # Remove EventBus metadata
        signal_data = {k: v for k, v in signal_data.items() if not k.startswith('_')}
        
        # Parse signal
        signal = TradeSignal(**signal_data)
        stats["signals_received"] += 1
        stats["last_signal_time"] = datetime.utcnow().isoformat()
        
        logger.info(
            f"📥 Signal received: {signal.symbol} {signal.action} "
            f"@ {signal.confidence:.3f} from {signal.source}"
        )
        
        # Skip HOLD signals
        if signal.action == "HOLD":
            stats["signals_rejected"] += 1
            stats["rejection_reasons"]["HOLD_SIGNAL"] = (
                stats["rejection_reasons"].get("HOLD_SIGNAL", 0) + 1
            )
            logger.info(f"⏭️  Skipped HOLD signal for {signal.symbol}")
            return
        
        # Apply governer logic
        allocation = governer.allocate_position(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            balance=None,  # Uses governer's internal balance
            meta_override=signal.meta_override
        )
        
        if not allocation.approved:
            # Rejected
            stats["signals_rejected"] += 1
            stats["rejection_reasons"][allocation.reason] = (
                stats["rejection_reasons"].get(allocation.reason, 0) + 1
            )
            logger.warning(
                f"❌ REJECTED: {signal.symbol} {signal.action} | "
                f"Reason: {allocation.reason}"
            )
            return
        
        # Approved - publish to safe topic
        approved_signal = RiskApprovedSignal(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            position_size_usd=allocation.position_size_usd,
            position_size_pct=allocation.position_size_pct,
            risk_amount_usd=allocation.risk_amount_usd,
            kelly_optimal=allocation.kelly_optimal,
            timestamp=datetime.utcnow().isoformat() + "Z",
            source=signal.source
        )
        
        await eventbus.publish_approved(approved_signal)
        
        stats["signals_approved"] += 1
        logger.info(
            f"✅ APPROVED: {signal.symbol} {signal.action} | "
            f"Size=${allocation.position_size_usd:.2f} "
            f"({allocation.position_size_pct:.1%}) | "
            f"Risk=${allocation.risk_amount_usd:.2f}"
        )
    
    except Exception as e:
        logger.error(f"❌ Error processing signal: {e}", exc_info=True)


async def signal_consumer():
    """Background task: consume signals from trade.signal.v5"""
    logger.info("🚀 Starting signal consumer...")
    
    try:
        async for signal_data in eventbus.subscribe("trade.signal.v5"):
            await process_signal(signal_data)
    except asyncio.CancelledError:
        logger.info("🛑 Signal consumer cancelled")
    except Exception as e:
        logger.error(f"❌ Signal consumer error: {e}", exc_info=True)


# ============================================================================
# LIFECYCLE
# ============================================================================

start_time = datetime.utcnow()


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global governer, eventbus
    
    logger.info("🚀 Risk Safety Service starting...")
    
    # Initialize governer
    risk_config = RiskConfig(
        max_position_size_pct=0.10,
        max_total_exposure_pct=0.50,
        max_drawdown_pct=0.15,
        min_confidence_threshold=0.65,
        kelly_fraction=0.25,
        cooldown_after_loss_minutes=60,
        max_daily_trades=20,
        emergency_stop=False
    )
    
    governer = GovernerAgent(
        config=risk_config,
        state_file="/app/data/governer_state.json"
    )
    logger.info("✅ Governer initialized")
    
    # Initialize EventBus
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    eventbus = EventBusClient(redis_url=redis_url)
    await eventbus.connect()
    logger.info(f"✅ EventBus connected: {redis_url}")
    
    # Start signal consumer
    asyncio.create_task(signal_consumer())
    logger.info("✅ Signal consumer started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 Risk Safety Service shutting down...")
    
    if eventbus:
        await eventbus.disconnect()
    
    logger.info("✅ Shutdown complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run service"""
    port = int(os.getenv("SERVICE_PORT", "8003"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
