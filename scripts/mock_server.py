#!/usr/bin/env python3
"""
Mock FastAPI server for stress testing - simulerer backend API-er
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random
import time
import uvicorn

app = FastAPI(title="Mock Quantum Trader API")

@app.get("/api/stats/overview")
async def get_stats_overview():
    # Simuler litt delay for realisme
    time.sleep(random.uniform(0.01, 0.05))
    return {
        "total_trades": random.randint(100, 1000),
        "active_positions": random.randint(5, 15),
        "portfolio_value": round(random.uniform(10000, 50000), 2),
        "daily_pnl": round(random.uniform(-500, 500), 2),
        "win_rate": round(random.uniform(0.4, 0.8), 3)
    }

@app.get("/api/prices/latest")
async def get_latest_prices():
    time.sleep(random.uniform(0.01, 0.03))
    symbol = "BTCUSDT"  # Default
    return {
        "symbol": symbol,
        "price": round(random.uniform(40000, 70000), 2),
        "change_24h": round(random.uniform(-5, 5), 3),
        "volume": random.randint(1000000, 5000000),
        "timestamp": int(time.time())
    }

@app.get("/api/trade_logs")
async def get_trade_logs():
    time.sleep(random.uniform(0.02, 0.04))
    trades = []
    for i in range(random.randint(5, 15)):
        trades.append({
            "id": i + 1,
            "symbol": random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"]),
            "side": random.choice(["BUY", "SELL"]),
            "quantity": round(random.uniform(0.001, 1.0), 6),
            "price": round(random.uniform(1000, 70000), 2),
            "timestamp": int(time.time()) - random.randint(0, 86400)
        })
    return {"trades": trades, "total": len(trades)}

@app.get("/api/ai/signals/latest")
async def get_ai_signals():
    time.sleep(random.uniform(0.03, 0.06))
    signals = []
    for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]:
        signals.append({
            "symbol": symbol,
            "signal": random.choice(["BUY", "SELL", "HOLD"]),
            "confidence": round(random.uniform(0.5, 0.95), 3),
            "price_target": round(random.uniform(1000, 70000), 2),
            "timestamp": int(time.time())
        })
    return {"signals": signals[:random.randint(3, 10)]}

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": int(time.time())}

if __name__ == "__main__":
    print("[ROCKET] Starting Mock Quantum Trader API on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")