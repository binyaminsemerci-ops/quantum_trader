"""
Trades Router
Live trading activity, position management, order flow
"""
import sys
sys.path.append("/app")

from fastapi import APIRouter, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import asyncio
import math
from .auth_router import verify_token
from db.connection import SessionLocal
from db.models.trade_journal import TradeJournal

try:
    from utils.realtime import push
except ImportError:
    async def push(channel, data):
        print(f"Mock push to {channel}: {data}")
        return True

router = APIRouter(prefix="/trades", tags=["Trades"])

async def compute_tp_sl(price: float, direction: str, conf: float):
    """Calculate dynamic TP/SL based on confidence"""
    base = 0.003 + conf * 0.004  # proportional to confidence
    tp = price * (1 + base) if direction == "BUY" else price * (1 - base)
    sl = price * (1 - base/2) if direction == "BUY" else price * (1 + base/2)
    trail = abs(tp - price) / 2
    return tp, sl, trail

@router.post("/execute")
async def execute_trade(trade: dict):
    """Execute trade with dynamic TP/SL and record to journal"""
    price = trade["price"]
    direction = trade["direction"]
    conf = trade["confidence"]
    tp, sl, trail = await compute_tp_sl(price, direction, conf)
    result = {
        "symbol": trade["symbol"],
        "direction": direction,
        "price": price,
        "tp": round(tp,5),
        "sl": round(sl,5),
        "trailing": round(trail,5),
        "confidence": conf,
        "model": trade.get("model", "unknown"),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Record to trade journal
    db = SessionLocal()
    try:
        journal = TradeJournal(
            symbol=trade["symbol"],
            direction=direction,
            entry_price=price,
            tp=round(tp,5),
            sl=round(sl,5),
            trailing_stop=round(trail,5),
            confidence=conf,
            model=trade.get("model", "unknown"),
            features={"expected_move": trade.get("expected_move", None)},
            policy_state={"max_leverage": 5, "risk_cap": 0.4},
            exit_reason="open"
        )
        db.add(journal)
        db.commit()
        db.refresh(journal)
        result["journal_id"] = journal.id
    except Exception as e:
        print(f"Journal recording error: {e}")
        db.rollback()
    finally:
        db.close()
    
    await push("positions", result)
    return result

class TradeResponse(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: Optional[float]
    status: str
    opened_at: str

@router.get("/active", response_model=List[TradeResponse])
def get_active_trades():
    """Get all active positions"""
    # Mock data - will connect to database
    return [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.5,
            "price": 42500.00,
            "pnl": 850.50,
            "status": "open",
            "opened_at": datetime.utcnow().isoformat()
        },
        {
            "id": 2,
            "symbol": "ETHUSDT",
            "side": "SHORT",
            "quantity": 5.0,
            "price": 2250.00,
            "pnl": -125.30,
            "status": "open",
            "opened_at": datetime.utcnow().isoformat()
        }
    ]

@router.get("/history")
def get_trade_history(
    limit: int = 50,
    symbol: Optional[str] = None
):
    """Get trade history with optional filters"""
    return {
        "trades": [],
        "total": 0,
        "filters": {"symbol": symbol, "limit": limit}
    }

@router.get("/{trade_id}")
def get_trade_details(trade_id: int):
    """Get detailed information about a specific trade"""
    return {
        "id": trade_id,
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 42500.00,
        "current_price": 43200.00,
        "pnl": 850.50,
        "pnl_percentage": 2.0,
        "quantity": 0.5,
        "stop_loss": 41500.00,
        "take_profit": 44000.00,
        "opened_at": datetime.utcnow().isoformat(),
        "strategy": "momentum_v3"
    }

@router.get("/live/feed")
def get_live_feed():
    """Get real-time trade feed"""
    return {
        "last_update": datetime.utcnow().isoformat(),
        "recent_trades": [],
        "execution_speed_ms": 45
    }
