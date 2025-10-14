from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime
import sys

# Add ai_engine to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(
    title="Quantum Trader API",
    description="AI-Powered Cryptocurrency Trading Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175", "http://localhost:5174", "http://localhost:5173", "http://127.0.0.1:5175", "http://127.0.0.1:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Quantum Trader API is running"}


@app.get("/api/ai/model/status")
async def get_model_status():
    """Get AI model status and metadata"""
    try:
        metadata_path = os.path.join(os.path.dirname(__file__), "..", "ai_engine", "models", "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return {
                "status": "Ready",
                "training_date": metadata.get("training_date"),
                "samples": metadata.get("samples"),
                "model_type": metadata.get("model_type"),
                "accuracy": metadata.get("accuracy", 0.85)
            }
        else:
            return {"status": "Not Trained", "training_date": None}

    except Exception as e:
        return {"status": "Error", "error": str(e)}


@app.get("/api/ai/signals/latest")
async def get_latest_signals():
    """Get latest AI trading signals"""
    try:
        # Mock signals for now - in production, these would come from your AI model
        signals = [
            {
                "id": "1",
                "symbol": "BTCUSDT",
                "type": "BUY",
                "confidence": 0.85,
                "price": 45200.50,
                "timestamp": datetime.now().isoformat(),
                "reason": "Strong upward momentum, positive sentiment"
            },
            {
                "id": "2",
                "symbol": "ETHUSDT",
                "type": "HOLD",
                "confidence": 0.65,
                "price": 2800.75,
                "timestamp": datetime.now().isoformat(),
                "reason": "Consolidation phase, wait for breakout"
            }
        ]

        return signals

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices/latest")
async def get_latest_price(symbol: str = "BTCUSDT"):
    """Get latest price for a symbol"""
    try:
        # Mock price data - in production, fetch from Binance API
        import random
        base_price = 45000 if symbol == "BTCUSDT" else 2800
        current_price = base_price + random.uniform(-1000, 1000)
        change = random.uniform(-200, 200)
        change_percent = (change / current_price) * 100

        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "change_percent": change_percent,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/retrain")
async def retrain_model():
    """Trigger AI model retraining"""
    try:
        import subprocess

        result = subprocess.run(
            ["cmd", "/c", "train_ai_model.bat"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )

        if result.returncode == 0:
            return {"status": "success", "message": "Model retraining started"}
        else:
            return {"status": "error", "message": result.stderr}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# COMMENTED OUT - These routers are not implemented yet
# When you create these router modules, uncomment the corresponding lines:
# 
# from routers import trades, stats, chart, settings, binance, signals, prices, candles
#
# app.include_router(trades.router, prefix="/trades")
# app.include_router(stats.router, prefix="/stats")
# app.include_router(chart.router, prefix="/chart")
# app.include_router(settings.router, prefix="/settings")
# app.include_router(binance.router, prefix="/binance")
# app.include_router(signals.router, prefix="/signals")
# app.include_router(prices.router, prefix="/prices")
# app.include_router(candles.router, prefix="/candles")
# app.include_router(trading_bot_router, prefix="/trading-bot", tags=["Trading Bot"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
