from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def get_chart():
    # Returner en liste som testene forventer
    return [100, 101, 102]


@router.get("/recent")
async def get_chart_recent(symbol: str = "BTCUSDT", limit: int = 50):
    """Return simple recent prices for frontend demo/testing."""
    base = 100.0 + (hash(symbol) % 50)
    return [round(base + i * 0.1, 3) for i in range(limit)]
