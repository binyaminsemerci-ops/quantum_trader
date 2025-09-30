from fastapi import APIRouter, Query
from typing import List, Dict, Literal, Annotated, Any
import datetime

from backend.utils.market_data import fetch_recent_signals

router = APIRouter()


@router.get("/recent", response_model=List[Dict])
def recent_signals(
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query()] = "mixed",
    symbol: str = "BTCUSDT",
) -> List[Dict]:
    """Return recent signals.

    When live market data is enabled the signals are derived from price momentum.
    Otherwise deterministic demo signals are returned.
    """
    records = fetch_recent_signals(symbol=symbol, limit=limit, profile=profile)
    # FastAPI response_model expects serialisable timestamps
    result: List[Dict] = []
    for item in records:
        ts = item.get("timestamp")
        if isinstance(ts, datetime.datetime):
            serialised = item.copy()
            serialised["timestamp"] = ts.isoformat()
            result.append(serialised)
        else:
            result.append(item)
    return result


@router.get("/", response_model=Dict[str, Any])
def paginated(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query()] = "mixed",
    symbol: str = "BTCUSDT",
) -> Dict[str, Any]:
    items = fetch_recent_signals(symbol=symbol, limit=page_size, profile=profile)
    return {
        "total": len(items),
        "page": page,
        "page_size": page_size,
        "items": items,
    }
