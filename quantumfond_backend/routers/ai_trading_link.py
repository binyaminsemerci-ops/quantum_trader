"""
AI-Trading Integration Router
Links AI predictions directly to trade execution
"""
from fastapi import APIRouter
from datetime import datetime
import httpx
import asyncio

try:
    from utils.realtime import push
except ImportError:
    async def push(channel, data):
        print(f"Mock push to {channel}: {data}")
        return True

router = APIRouter(prefix="/ai-trading", tags=["AI-Trading"])

@router.post("/trigger")
async def trigger_trade():
    """
    Trigger AI prediction → Trade execution pipeline
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get AI prediction
            base_url = "http://localhost:8026"  # QuantumFond backend
            p = await client.get(f"{base_url}/ai/predict")
            pred = p.json()
            
            # Mock price lookup (in production, fetch from exchange)
            price_map = {
                "BTCUSDT": 42000,
                "ETHUSDT": 2200,
                "SOLUSDT": 95,
                "XRPUSDT": 0.52
            }
            
            trade = {
                "symbol": pred["symbol"],
                "price": price_map.get(pred["symbol"], 10000),
                "direction": pred["signal"],
                "confidence": pred["confidence"],
                "model": pred["model"]
            }
            
            # Execute trade with dynamic TP/SL
            t = await client.post(f"{base_url}/trades/execute", json=trade)
            
            # Push signal to Redis for frontend
            await push("signals", pred)
            
            return {
                "status": "success",
                "prediction": pred,
                "execution": t.json(),
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
