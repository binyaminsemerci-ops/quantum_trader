from fastapi import APIRouter, Query
from typing import List, Dict, Literal, Annotated
import datetime
import random

router = APIRouter()


def _iso_now_minus(seconds: int) -> str:
    return (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)).isoformat()


@router.get("/recent")
def recent_signals(
    limit: Annotated[int, Query(20, ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query("mixed")] = "mixed",
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


    return signals
