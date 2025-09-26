from fastapi import APIRouter, Query
from typing import List, Dict, Literal
import datetime
import random

router = APIRouter()


def _iso_now_minus(seconds: int) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)).isoformat()


@router.get("/recent")
def recent_signals(
    limit: int = Query(20, ge=1, le=200),
    profile: Literal["left", "right", "mixed"] = Query("mixed"),
) -> List[Dict]:
    """Return a list of mock signals for frontend development/testing.

    Added fields: direction, confidence, details and different symbols. A
    `profile` query param allows producing left- or right-skewed score
    distributions for UI testing.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
    now = datetime.datetime.now(datetime.timezone.utc)
    signals: List[Dict] = []

    # seed random so results are stable-ish per process start
    rnd = random.Random(42)

    for i in range(limit):
        # spacing signals over the past (limit * 15) seconds
        seconds_ago = (limit - i) * 15
        ts = (now - datetime.timedelta(seconds=seconds_ago)).isoformat()
        symbol = symbols[i % len(symbols)]

        # score distribution depends on profile
        base = i / max(1, limit - 1)
        if profile == "left":
            score = round(max(0.0, 0.05 + (1 - base) * 0.95 * rnd.random()), 3)
        elif profile == "right":
            score = round(min(1.0, 0.05 + base * 0.95 * rnd.random()), 3)
        else:
            # mixed: small oscillation around 0.5
            score = round(0.3 + (rnd.random() * 0.4), 3)

        direction = "LONG" if score >= 0.5 else "SHORT"
        confidence = round(min(1.0, max(0.0, 0.2 + rnd.random() * 0.8)), 3)

        signals.append(
            {
                "id": f"sig-{i}",
                "symbol": symbol,
                "score": score,
                "direction": direction,
                "confidence": confidence,
                "timestamp": ts,
                "details": {
                    "source": "simulator",
                    "note": f"mock signal #{i} ({profile})",
                },
            }
        )

    return signals


from fastapi import Query


@router.get("/prices/recent")
def recent_prices(symbol: str = "BTCUSDT", limit: int = Query(50, ge=1, le=500)) -> List[Dict]:
    """Return a deterministic demo series of candles for the requested symbol.

    This endpoint is intentionally simple and deterministic so the frontend can
    display a demo price chart without external data.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    candles: List[Dict] = []
    base = 100.0 + (hash(symbol) % 50)
    for i in range(limit):
        t = (now - datetime.timedelta(minutes=(limit - i))).isoformat()
        # small deterministic walk using i
        open_p = base + (i * 0.1) + (0.5 * (i % 3))
        close_p = open_p + ((-1) ** i) * (0.5 * ((i % 5) / 5.0))
        high_p = max(open_p, close_p) + 0.4
        low_p = min(open_p, close_p) - 0.4
        volume = 10 + (i % 7)
        candles.append(
            {
                "time": t,
                "open": round(open_p, 3),
                "high": round(high_p, 3),
                "low": round(low_p, 3),
                "close": round(close_p, 3),
                "volume": volume,
            }
        )
    return candles
