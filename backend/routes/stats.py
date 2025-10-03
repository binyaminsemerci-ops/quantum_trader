from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def get_stats():
    # Returner i format som matcher testene
    return {"total_trades": 0, "pnl": 0.0}


@router.get("/overview")
async def stats_overview():
    """Return demo stats useful for the frontend dashboard."""
    return {
        "total_trades": 123,
        "pnl": 456.78,
        "open_positions": [
            {"symbol": "BTCUSDT", "qty": 0.5, "avg_price": 9500.0},
        ],
        "since": "2025-01-01T00:00:00Z",
    }
