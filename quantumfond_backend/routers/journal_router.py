import sys
sys.path.append("/app")

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from db.connection import get_db
from db.models.trade_journal import TradeJournal
from utils.realtime import push

router = APIRouter(prefix="/journal", tags=["Journal"])

@router.post("/record")
async def record_trade(trade: dict, db: Session = Depends(get_db)):
    """
    Record a trade to the journal with full context.
    Called automatically on trade execution.
    """
    try:
        t = TradeJournal(**trade)
        db.add(t)
        db.commit()
        db.refresh(t)
        
        # Push to realtime stream
        await push("journal", {
            "id": t.id,
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "confidence": t.confidence,
            "timestamp": t.timestamp.isoformat()
        })
        
        return {"saved": True, "id": t.id, "timestamp": t.timestamp}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to record trade: {str(e)}")

@router.get("/history")
def get_journal(
    limit: int = 50,
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve trade history with optional filters.
    Returns recent trades ordered by timestamp descending.
    """
    query = db.query(TradeJournal)
    
    if symbol:
        query = query.filter(TradeJournal.symbol == symbol)
    if direction:
        query = query.filter(TradeJournal.direction == direction)
    
    rows = query.order_by(TradeJournal.timestamp.desc()).limit(limit).all()
    
    return [{
        "id": r.id,
        "symbol": r.symbol,
        "direction": r.direction,
        "entry_price": r.entry_price,
        "exit_price": r.exit_price,
        "pnl": r.pnl,
        "tp": r.tp,
        "sl": r.sl,
        "trailing": r.trailing,
        "confidence": r.confidence,
        "model": r.model,
        "features": r.features,
        "policy_state": r.policy_state,
        "exit_reason": r.exit_reason,
        "timestamp": r.timestamp.isoformat()
    } for r in rows]

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Get journal statistics: total trades, win rate, avg PnL.
    """
    total = db.query(TradeJournal).count()
    closed = db.query(TradeJournal).filter(TradeJournal.pnl != None).all()
    
    if not closed:
        return {"total": total, "closed": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
    
    wins = [t for t in closed if t.pnl > 0]
    total_pnl = sum(t.pnl for t in closed)
    avg_pnl = total_pnl / len(closed)
    win_rate = len(wins) / len(closed) if closed else 0
    
    return {
        "total": total,
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "win_rate": round(win_rate * 100, 2),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2)
    }

@router.post("/tag/{trade_id}")
def tag_trade(trade_id: int, tag: str, db: Session = Depends(get_db)):
    """
    Add a tag to a trade for categorization and analysis.
    Tags are stored in policy_state JSON field.
    """
    t = db.query(TradeJournal).filter(TradeJournal.id == trade_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    ps = t.policy_state or {}
    tags = ps.get("tags", [])
    if tag not in tags:
        tags.append(tag)
    ps["tags"] = tags
    t.policy_state = ps
    
    db.commit()
    return {"tagged": True, "trade_id": trade_id, "tags": tags}

@router.post("/close/{trade_id}")
def close_trade(trade_id: int, exit_price: float, exit_reason: str, db: Session = Depends(get_db)):
    """
    Close a trade and calculate PnL.
    """
    t = db.query(TradeJournal).filter(TradeJournal.id == trade_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    # Calculate PnL
    if t.direction == "BUY":
        pnl = exit_price - t.entry_price
    else:  # SELL
        pnl = t.entry_price - exit_price
    
    t.exit_price = exit_price
    t.pnl = pnl
    t.exit_reason = exit_reason
    
    db.commit()
    return {"closed": True, "trade_id": trade_id, "pnl": pnl}
