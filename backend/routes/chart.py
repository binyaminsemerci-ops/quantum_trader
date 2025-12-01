from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def get_chart():
    # Return chart data in format expected by frontend
    import datetime
    base_equity = 125000.0
    points = []
    current_equity = base_equity
    
    for i in range(20):
        # Generate realistic equity curve with some volatility
        current_equity += (hash(f"equity_{i}") % 1000 - 500) / 10.0  # Random walk
        current_equity = max(current_equity, base_equity * 0.8)  # Don't drop below 80%
        
        timestamp = datetime.datetime.now() - datetime.timedelta(minutes=20-i)
        points.append({
            "timestamp": timestamp.isoformat(),
            "equity": round(current_equity, 2)
        })
    
    return points


@router.get("/recent")
async def get_chart_recent(symbol: str = "BTCUSDT", limit: int = 50):
    """Return simple recent prices for frontend demo/testing."""
    base = 100.0 + (hash(symbol) % 50)
    return [round(base + i * 0.1, 3) for i in range(limit)]
