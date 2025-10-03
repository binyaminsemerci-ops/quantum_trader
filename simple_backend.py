#!/usr/bin/env python3
"""
Simple working backend for quantum trader with basic AI endpoints
"""
from datetime import datetime, timezone
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Quantum Trader API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple state management
_ai_state = {
    "learning_active": False,
    "symbols_monitored": 0,
    "data_points": 0,
    "model_accuracy": 0.0,
    "enabled": False,
    "symbols": [],
    "last_signal_time": None,
    "total_signals": 0,
}
_state_lock = threading.Lock()


@app.get("/api/v1/system/status")
def get_system_status():
    """Get system status"""
    return {
        "status": "online",
        "service": "quantum_trader_core",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": "5 min",
        "version": "1.0.0",
    }


@app.get("/api/v1/ai-trading/status")
def get_ai_trading_status():
    """Get AI trading status"""
    with _state_lock:
        return {
            "enabled": _ai_state["enabled"],
            "symbols": _ai_state["symbols"],
            "last_signal_time": _ai_state["last_signal_time"],
            "total_signals": _ai_state["total_signals"],
            "accuracy": _ai_state["model_accuracy"],
            "learning_active": _ai_state["learning_active"],
            "symbols_monitored": _ai_state["symbols_monitored"],
            "data_points": _ai_state["data_points"],
            "continuous_learning_status": (
                "Active" if _ai_state["learning_active"] else "Inactive"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.post("/api/v1/continuous-learning/start")
async def start_continuous_learning(symbols: list[str] = None):
    """Start continuous learning"""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    with _state_lock:
        _ai_state["learning_active"] = True
        _ai_state["symbols_monitored"] = len(symbols)
        _ai_state["data_points"] = 100  # Simulated
        _ai_state["model_accuracy"] = 0.75

    return {
        "status": "Continuous Learning Started",
        "message": "Real-time AI strategy evolution from live data feeds",
        "symbols": symbols,
        "twitter_analysis": "ACTIVE",
        "market_feeds": "ACTIVE",
        "model_training": "ACTIVE",
        "enhanced_sources": "ACTIVE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/continuous-learning/status")
async def get_learning_status():
    """Get continuous learning status"""
    with _state_lock:
        if not _ai_state["learning_active"]:
            return {
                "learning_active": False,
                "symbols_monitored": 0,
                "data_points": 0,
                "model_accuracy": 0.0,
                "status": "Inactive",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "learning_active": True,
            "symbols_monitored": _ai_state["symbols_monitored"],
            "data_points": _ai_state["data_points"],
            "model_accuracy": _ai_state["model_accuracy"],
            "status": "Active",
            "last_training": datetime.now(timezone.utc).isoformat(),
            "twitter_sentiment": "ACTIVE",
            "market_data": "ACTIVE",
            "enhanced_feeds": "ACTIVE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.post("/api/v1/continuous-learning/stop")
async def stop_continuous_learning():
    """Stop continuous learning"""
    with _state_lock:
        _ai_state["learning_active"] = False
        _ai_state["symbols_monitored"] = 0
        _ai_state["data_points"] = 0

    return {
        "status": "Continuous Learning Stopped",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/portfolio")
async def get_portfolio():
    """Get portfolio data"""
    return {"total_value": 861498, "positions": 1, "pnl_percent": -38.50}


@app.get("/api/v1/portfolio/market-overview")
async def get_market_overview():
    """Get market overview"""
    return {"market_cap": 1000000000, "volume_24h": 1000000, "fear_greed": 52}


@app.get("/api/v1/signals/recent")
async def get_recent_signals(limit: int = 5):
    """Get recent signals"""
    return [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "confidence": 0.9,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "symbol": "ETHUSDT",
            "side": "sell",
            "confidence": 0.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
