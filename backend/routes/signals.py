from fastapi import APIRouter, Query
from typing import List, Dict, Literal, Annotated, Optional
import datetime
import random
from pydantic import BaseModel, Field

router = APIRouter()


class SignalDetails(BaseModel):
    source: str
    note: Optional[str]


class Signal(BaseModel):
    id: str
    timestamp: datetime.datetime
    symbol: str
    side: Literal["buy", "sell"]
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    details: SignalDetails


class PaginatedSignals(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[Signal]


def _generate_mock_signals(count: int, profile: Literal["left", "right", "mixed"]) -> List[Dict]:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
    now = datetime.datetime.now(datetime.timezone.utc)
    signals: List[Dict] = []
    rnd = random.Random(42)  # deterministic demo generator (no-sec)

    for i in range(count):
        seconds_ago = (count - i) * 15
        ts = (now - datetime.timedelta(seconds=seconds_ago))
        symbol = symbols[i % len(symbols)]

        base = i / max(1, count - 1)
        if profile == "left":
            score = round(max(0.0, 0.05 + (1 - base) * 0.95 * rnd.random()), 3)
        elif profile == "right":
            score = round(min(1.0, 0.05 + base * 0.95 * rnd.random()), 3)
        else:
            score = round(0.3 + (rnd.random() * 0.4), 3)

        side = "buy" if score >= 0.5 else "sell"
        confidence = round(min(1.0, max(0.0, 0.2 + rnd.random() * 0.8)), 3)

        signals.append(
            {
                "id": f"sig-{i}",
                "timestamp": ts,
                "symbol": symbol,
                "side": side,
                "score": score,
                "confidence": confidence,
                "details": {"source": "simulator", "note": f"mock signal #{i} ({profile})"},
            }
        )

    return signals


@router.get("/recent", response_model=List[Dict])
def recent_signals(
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query()] = "mixed",
) -> List[Dict]:
    """Legacy endpoint kept for tests and frontend stubs. Returns a list of
    mock signals as dicts (same shape as new typed endpoints).
    """
    return _generate_mock_signals(limit, profile)


@router.get("/", response_model=PaginatedSignals)
def list_signals(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query()] = "mixed",
    symbol: Optional[str] = None,
):
    """Return paginated mock signals. This is a deterministic generator for
    frontend/demo use and tests; in production this would query a database.
    """
    total_available = 500
    all_signals = _generate_mock_signals(total_available, profile)

    if symbol:
        all_signals = [s for s in all_signals if s["symbol"] == symbol]

    total = len(all_signals)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_signals[start:end]

    # Convert timestamps to datetime for Pydantic model parsing
    for it in page_items:
        if isinstance(it["timestamp"], datetime.datetime):
            continue
        it["timestamp"] = datetime.datetime.fromisoformat(it["timestamp"])

    # Convert dicts to Signal instances so the PaginatedSignals items list
    # has the expected type: List[Signal]. This satisfies mypy and ensures
    # response_model validation uses the Signal model.
    signal_items = [Signal(**it) for it in page_items]

    return PaginatedSignals(total=total, page=page, page_size=page_size, items=signal_items)
