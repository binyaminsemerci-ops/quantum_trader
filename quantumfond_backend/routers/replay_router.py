import sys
sys.path.append("/app")

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from db.connection import get_db
from db.models.trade_journal import TradeJournal

router = APIRouter(prefix="/replay", tags=["Replay"])

@router.get("/{trade_id}")
def replay_trade(trade_id: int, db: Session = Depends(get_db)):
    """
    Replay a trade with full context and timeline.
    Provides explainability for trade decisions.
    """
    t = db.query(TradeJournal).filter(TradeJournal.id == trade_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    # Generate timeline with key events
    timeline = [
        {
            "time": t.timestamp.isoformat(),
            "price": t.entry_price,
            "event": "entry",
            "detail": f"{t.direction} {t.symbol} @ {t.entry_price}"
        },
        {
            "time": t.timestamp.isoformat(),
            "price": t.tp,
            "event": "tp_target",
            "detail": f"Take Profit set at {t.tp} (+{abs(t.tp - t.entry_price):.2f})"
        },
        {
            "time": t.timestamp.isoformat(),
            "price": t.sl,
            "event": "sl_threshold",
            "detail": f"Stop Loss set at {t.sl} (-{abs(t.entry_price - t.sl):.2f})"
        },
        {
            "time": t.timestamp.isoformat(),
            "price": t.entry_price,
            "event": "trailing_active",
            "detail": f"Trailing Stop: {t.trailing_stop} points"
        }
    ]
    
    # Add exit event if trade is closed
    if t.exit_price:
        timeline.append({
            "time": t.timestamp.isoformat(),  # Would be close_time if we track it
            "price": t.exit_price,
            "event": "exit",
            "detail": f"Closed: {t.exit_reason} | PnL: {t.pnl:+.2f}"
        })
    
    replay = {
        "trade_id": t.id,
        "symbol": t.symbol,
        "direction": t.direction,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "pnl": t.pnl,
        "tp": t.tp,
        "sl": t.sl,
        "trailing_stop": t.trailing_stop,
        "confidence": t.confidence,
        "model": t.model,
        "timeline": timeline,
        "features": t.features,
        "policy_state": t.policy_state,
        "exit_reason": t.exit_reason,
        "timestamp": t.timestamp.isoformat()
    }
    
    return replay

@router.get("/batch/{trade_ids}")
def replay_batch(trade_ids: str, db: Session = Depends(get_db)):
    """
    Replay multiple trades for comparison.
    trade_ids should be comma-separated, e.g., "1,2,3"
    """
    ids = [int(id.strip()) for id in trade_ids.split(",")]
    trades = db.query(TradeJournal).filter(TradeJournal.id.in_(ids)).all()
    
    return [replay_trade(t.id, db) for t in trades]

@router.get("/analyze/{symbol}")
def analyze_symbol(symbol: str, db: Session = Depends(get_db)):
    """
    Analyze all trades for a specific symbol.
    Returns aggregated statistics and patterns.
    """
    trades = db.query(TradeJournal).filter(TradeJournal.symbol == symbol).all()
    
    if not trades:
        raise HTTPException(status_code=404, detail=f"No trades found for {symbol}")
    
    closed_trades = [t for t in trades if t.pnl is not None]
    wins = [t for t in closed_trades if t.pnl > 0]
    
    analysis = {
        "symbol": symbol,
        "total_trades": len(trades),
        "closed_trades": len(closed_trades),
        "open_trades": len(trades) - len(closed_trades),
        "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
        "avg_confidence": sum(t.confidence for t in trades) / len(trades),
        "avg_pnl": sum(t.pnl for t in closed_trades) / len(closed_trades) if closed_trades else 0,
        "total_pnl": sum(t.pnl for t in closed_trades if t.pnl),
        "best_model": max(set(t.model for t in trades), key=lambda m: sum(1 for t in trades if t.model == m)),
        "recent_trades": [
            {
                "id": t.id,
                "direction": t.direction,
                "entry": t.entry_price,
                "pnl": t.pnl,
                "confidence": t.confidence,
                "model": t.model,
                "timestamp": t.timestamp.isoformat()
            }
            for t in sorted(trades, key=lambda x: x.timestamp, reverse=True)[:5]
        ]
    }
    
    return analysis
