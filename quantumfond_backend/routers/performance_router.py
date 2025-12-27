"""
Performance Router - Enhanced Analytics Engine
Real-time metrics computation from TradeJournal
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import math
from .auth_router import verify_token
from db.connection import SessionLocal
from db.models.trade_journal import TradeJournal
import pandas as pd
import numpy as np

router = APIRouter(prefix="/performance", tags=["Performance Analytics"])

def safe_float(value, default=0.0):
    """Safely convert value to float with fallback"""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default

def compute_metrics(df: pd.DataFrame):
    """Compute comprehensive trading metrics from DataFrame"""
    if df.empty:
        return {
            "total_return": 0.0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "average_win": 0.0,
            "average_loss": 0.0
        }, []
    
    # Clean and prepare data
    df["return"] = df["pnl"].fillna(0).astype(float)
    
    # Basic metrics
    total_return = safe_float(df["return"].sum())
    total_trades = len(df)
    winning_trades = int((df["return"] > 0).sum())
    losing_trades = int((df["return"] < 0).sum())
    winrate = safe_float(winning_trades / total_trades if total_trades > 0 else 0)
    
    # Win/Loss averages
    avg_win = safe_float(df.loc[df["return"] > 0, "return"].mean() if winning_trades > 0 else 0)
    avg_loss = safe_float(abs(df.loc[df["return"] < 0, "return"].mean()) if losing_trades > 0 else 0)
    profit_factor = safe_float(avg_win / avg_loss if avg_loss > 0 else 0)
    
    # Risk-adjusted metrics
    returns_std = df["return"].std()
    sharpe = safe_float(df["return"].mean() / (returns_std + 1e-9))
    
    downside_returns = df.loc[df["return"] < 0, "return"]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-9
    sortino = safe_float(df["return"].mean() / (downside_std + 1e-9))
    
    # Equity curve and drawdown
    df["equity"] = df["return"].cumsum()
    peak = df["equity"].cummax()
    drawdown = df["equity"] - peak
    max_dd = safe_float(drawdown.min())
    
    metrics = {
        "total_return": round(total_return, 2),
        "winrate": round(winrate, 3),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_dd, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "average_win": round(avg_win, 2),
        "average_loss": round(avg_loss, 2)
    }
    
    # Equity curve data points
    curve = df[["timestamp", "equity", "return"]].copy()
    curve["timestamp"] = curve["timestamp"].astype(str)
    curve_data = curve.to_dict(orient="records")
    
    return metrics, curve_data

@router.get("/metrics")
def get_performance_metrics():
    """Get real-time computed performance metrics from TradeJournal"""
    db = SessionLocal()
    try:
        rows = db.query(TradeJournal).filter(TradeJournal.pnl.isnot(None)).all()
        
        if not rows:
            return {"metrics": {}, "curve": []}
        
        # Convert to DataFrame
        data = [{
            "timestamp": r.timestamp,
            "pnl": r.pnl,
            "symbol": r.symbol,
            "direction": r.direction
        } for r in rows]
        
        df = pd.DataFrame(data)
        df = df.sort_values("timestamp")
        
        metrics, curve = compute_metrics(df)
        
        return {
            "metrics": metrics,
            "curve": curve
        }
    finally:
        db.close()

@router.get("/summary")
def get_performance_summary():
    """Get overall performance summary (legacy endpoint)"""
    db = SessionLocal()
    try:
        rows = db.query(TradeJournal).filter(TradeJournal.pnl.isnot(None)).all()
        
        if not rows:
            return {
                "total_return": {"daily": 0, "weekly": 0, "monthly": 0, "ytd": 0},
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "average_win": 0,
                "average_loss": 0
            }
        
        data = [{"timestamp": r.timestamp, "pnl": r.pnl} for r in rows]
        df = pd.DataFrame(data).sort_values("timestamp")
        
        metrics, _ = compute_metrics(df)
        
        return {
            "total_return": {
                "daily": safe_float(metrics["total_return"] / 30 if len(df) > 30 else metrics["total_return"]),
                "weekly": safe_float(metrics["total_return"] / 4 if len(df) > 7 else metrics["total_return"]),
                "monthly": safe_float(metrics["total_return"]),
                "ytd": safe_float(metrics["total_return"])
            },
            "sharpe_ratio": metrics["sharpe"],
            "sortino_ratio": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["winrate"],
            "profit_factor": metrics["profit_factor"],
            "total_trades": metrics["total_trades"],
            "winning_trades": metrics["winning_trades"],
            "losing_trades": metrics["losing_trades"],
            "average_win": metrics["average_win"],
            "average_loss": -metrics["average_loss"]
        }
    finally:
        db.close()
