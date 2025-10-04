"""
TRADING BOT API ENDPOINTS
========================

REST endpoints to control the autonomous trading bot
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import asyncio
import logging

from .autonomous_trader import get_trading_bot

# NOTE: Do not set prefix here because main.py mounts with prefix="/trading-bot".
router = APIRouter(
    tags=["Trading Bot"],
    responses={
        400: {"description": "Invalid request"},
        500: {"description": "Trading bot error"}
    }
)

logger = logging.getLogger(__name__)

# Background task reference
_bot_task: Optional[asyncio.Task] = None


@router.get("/status")
async def get_bot_status() -> Dict[str, Any]:
    """Get current trading bot status"""
    bot = get_trading_bot()
    return bot.get_status()


@router.post("/start")
async def start_bot(dry_run: bool = True) -> Dict[str, str]:
    """Start the autonomous trading bot"""
    global _bot_task

    try:
        bot = get_trading_bot()

        if bot.running:
            return {"status": "already_running", "message": "Trading bot is already running"}

        # Start bot in background
        bot.dry_run = dry_run
        _bot_task = asyncio.create_task(bot.start())

        mode = "DRY RUN" if dry_run else "LIVE TRADING"
        logger.info(f"🚀 Trading bot started in {mode} mode")

        return {
            "status": "started",
            "message": f"Trading bot started in {mode} mode",
            "dry_run": str(dry_run)
        }

    except Exception as e:
        logger.error(f"Failed to start trading bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_bot() -> Dict[str, str]:
    """Stop the autonomous trading bot"""
    global _bot_task

    try:
        bot = get_trading_bot()

        if not bot.running:
            return {"status": "already_stopped", "message": "Trading bot is not running"}

        # Stop bot
        bot.stop()

        # Cancel background task
        if _bot_task and not _bot_task.done():
            _bot_task.cancel()
            try:
                await _bot_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Trading bot stopped")

        return {"status": "stopped", "message": "Trading bot stopped successfully"}

    except Exception as e:
        logger.error(f"Failed to stop trading bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions() -> Dict[str, Any]:
    """Get current trading positions across all enabled markets."""
    bot = get_trading_bot()
    positions_summary: Dict[str, Any] = {}
    total_positions = 0
    try:
        # New multi-market structure
        for market, pos in bot.positions.items():  # type: ignore[attr-defined]
            market_entries = {}
            for symbol, p in pos.items():
                market_entries[symbol] = {
                    "side": p.get("side"),
                    "qty": p.get("qty"),
                    "entry_price": p.get("entry_price"),
                    "confidence": p.get("confidence"),
                    "stop_loss": p.get("stop_loss"),
                    "take_profit": p.get("take_profit"),
                    "entry_time": p["entry_time"].isoformat() if p.get("entry_time") is not None else None,
                }
            positions_summary[market] = market_entries
            total_positions += len(pos)
    except Exception:
        # Fallback to legacy flat mapping
        positions_summary = bot.positions  # type: ignore[assignment]
        total_positions = len(bot.positions)  # type: ignore[arg-type]

    return {
        "positions": positions_summary,
        "total_positions": total_positions,
        "markets": list(positions_summary.keys()),
        "balance": bot.balance,
        "market_balances": getattr(bot, "market_balances", {}),
        "running": bot.running,
    }


@router.post("/settings")
async def update_settings(
    risk_per_trade: Optional[float] = None,
    min_confidence: Optional[float] = None,
    balance: Optional[float] = None
) -> Dict[str, str]:
    """Update bot settings"""
    try:
        bot = get_trading_bot()

        if bot.running:
            raise HTTPException(status_code=400, detail="Cannot update settings while bot is running")

        if risk_per_trade is not None:
            if not 0.001 <= risk_per_trade <= 0.1:  # 0.1% to 10%
                raise HTTPException(status_code=400, detail="Risk per trade must be between 0.1% and 10%")
            bot.risk_per_trade = risk_per_trade

        if min_confidence is not None:
            if not 0.1 <= min_confidence <= 1.0:
                raise HTTPException(status_code=400, detail="Min confidence must be between 0.1 and 1.0")
            bot.min_confidence = min_confidence

        if balance is not None:
            if balance <= 0:
                raise HTTPException(status_code=400, detail="Balance must be positive")
            bot.balance = balance

        return {"status": "updated", "message": "Bot settings updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
