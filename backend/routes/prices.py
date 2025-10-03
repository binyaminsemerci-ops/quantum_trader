from typing import Dict, List, Annotated
from fastapi import APIRouter, Query

from backend.utils.market_data import fetch_recent_candles

router = APIRouter()


@router.get("/recent")
def recent_prices(
    symbol: str = "BTCUSDT", limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> List[Dict]:
    """Return recent candles for the requested symbol.

    Uses ccxt when ENABLE_LIVE_MARKET_DATA=1 (and ccxt is installed), otherwise
    falls back to deterministic demo data so the frontend always has content.
    """
    candles = fetch_recent_candles(symbol=symbol, limit=limit)
    return candles
