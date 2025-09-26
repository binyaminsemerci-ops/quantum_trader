from fastapi import APIRouter
from typing import List, Dict
import datetime

router = APIRouter()


@router.get("/recent")
def recent_signals(limit: int = 20) -> List[Dict]:
    """Return a small list of mock signals for frontend development/testing.

    The shape is intentionally simple so the frontend can render it without
    needing external services.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    signals = []
    for i in range(limit):
        ts = (now - datetime.timedelta(seconds=(limit - i) * 15)).isoformat()
        signals.append({
            "id": f"sig-{i}",
            "symbol": "BTCUSDT",
            "score": round(0.5 + (i / max(1, limit)) * 0.5, 3),
            "timestamp": ts,
        })
    return signals
