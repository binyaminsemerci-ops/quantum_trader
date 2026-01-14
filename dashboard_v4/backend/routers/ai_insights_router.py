from fastapi import APIRouter
from typing import List
import numpy as np
import time
import random

router = APIRouter(prefix="/ai", tags=["AI Insights"])

def compute_drift_score(history: List[float]) -> float:
    """Return drift = variance / mean"""
    if not history or np.mean(history) == 0:
        return 0.0
    return float(np.var(history) / np.mean(history))

@router.get("/insights")
async def get_ai_insights():
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
