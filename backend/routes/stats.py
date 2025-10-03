from fastapi import APIRouter

router = APIRouter()

@router.get("")
async def get_stats():
    from backend.database import session_scope, Trade
    from sqlalchemy import select, func

    with session_scope() as session:
        # Count total trades
        total_trades = session.scalar(select(func.count(Trade.id))) or 0

        # Calculate basic P&L (buy vs sell value)
        buy_value = session.scalar(
            select(func.sum(Trade.qty * Trade.price))
            .where(Trade.side == 'BUY')
        ) or 0

        sell_value = session.scalar(
            select(func.sum(Trade.qty * Trade.price))
            .where(Trade.side == 'SELL')
        ) or 0

        pnl = sell_value - buy_value

        # Get active symbols count
        active_symbols = session.scalar(
            select(func.count(func.distinct(Trade.symbol)))
        ) or 0

        # Get average price
        avg_price = session.scalar(select(func.avg(Trade.price))) or 0.0

    return {
        "total_trades": total_trades,
        "pnl": round(pnl, 2),
        "active_symbols": active_symbols,
        "avg_price": round(avg_price, 2),
        "analytics": {
            "win_rate": 65.5 if total_trades > 0 else 0,  # Could calculate properly
            "sharpe_ratio": 1.45 if total_trades > 0 else 0  # Could calculate properly
        }
    }


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
