from fastapi import APIRouter
from typing import List, Dict, Any
import numpy as np
import time
import random
import redis
import json
from datetime import datetime

router = APIRouter(prefix="/ai", tags=["AI Insights"])

# Redis connection for live predictions
try:
    redis_client = redis.Redis(host='quantum_redis', port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ AI Insights: Connected to Redis")
except Exception as e:
    redis_client = None
    print(f"⚠️ AI Insights: Redis not available - {e}")

def compute_drift_score(history: List[float]) -> float:
    """Return drift = variance / mean"""
    if not history or np.mean(history) == 0:
        return 0.0
    return float(np.var(history) / np.mean(history))

@router.get("/insights")
def get_ai_insights():
    # Simulate 10 latest accuracy samples
    accuracy_series = [round(random.uniform(0.65, 0.85), 3) for _ in range(10)]
    sharpe = round(random.uniform(0.9, 1.4), 2)
    latency = random.randint(140, 280)
    models = ["XGB", "LGBM", "N-HiTS", "TFT"]

    drift = compute_drift_score(accuracy_series)
    suggestion = "Retrain model" if drift > 0.25 else "Stable"

    return {
        "timestamp": time.time(),
        "models": models,
        "accuracy": accuracy_series[-1],
        "sharpe": sharpe,
        "drift_score": round(drift, 3),
        "latency": latency,
        "suggestion": suggestion
    }

@router.get("/predictions")
def get_ai_predictions():
    """
    Get latest AI predictions from Redis streams.
    Returns trade intents with confidence, entry/exit levels, and model decisions.
    """
    if not redis_client:
        return {
            "predictions": [],
            "count": 0,
            "error": "Redis not available"
        }
    
    try:
        # Fetch latest trade intents from Redis stream
        stream_data = redis_client.xrevrange(
            "quantum:stream:trade.intent",
            count=10
        )
        
        predictions = []
        for stream_id, fields in stream_data:
            # Parse payload JSON
            payload_str = fields.get('payload', '{}')
            try:
                payload = json.loads(payload_str)
                
                # Extract prediction details
                prediction = {
                    "id": stream_id,
                    "timestamp": payload.get("timestamp", fields.get("timestamp", "")),
                    "symbol": payload.get("symbol", ""),
                    "side": payload.get("side", ""),
                    "confidence": round(float(payload.get("confidence", 0)), 4),
                    "entry_price": round(float(payload.get("entry_price", 0)), 4),
                    "stop_loss": round(float(payload.get("stop_loss", 0)), 4),
                    "take_profit": round(float(payload.get("take_profit", 0)), 4),
                    "leverage": int(payload.get("leverage", 1)),
                    "model": payload.get("model", "unknown"),
                    "reason": payload.get("reason", ""),
                    "volatility": round(float(payload.get("volatility_factor", 0)), 2),
                    "regime": payload.get("regime", "unknown"),
                    "position_size_usd": float(payload.get("position_size_usd", 0))
                }
                predictions.append(prediction)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                continue
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "predictions": [],
            "count": 0,
            "error": str(e)
        }
