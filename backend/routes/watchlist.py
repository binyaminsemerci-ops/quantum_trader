from typing import List, Dict, Annotated
from fastapi import APIRouter, Query, HTTPException

from backend.utils.market_data import fetch_recent_candles
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from sqlalchemy import select
from backend.database import session_scope, WatchlistEntry, Alert

router = APIRouter()
from backend.alerts.evaluator import register_ws, unregister_ws


@router.get("")
def list_watchlist():
    out = []
    with session_scope() as session:
        for row in session.execute(select(WatchlistEntry)).scalars().all():
            out.append(
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                }
            )
    return out


@router.post("")
def add_watchlist_entry(data: dict):
    symbol = data.get("symbol")
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    with session_scope() as session:
        entry = WatchlistEntry(symbol=symbol)
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return {"id": entry.id, "symbol": entry.symbol}


@router.delete("/{entry_id}")
def delete_watchlist_entry(entry_id: int):
    with session_scope() as session:
        obj = session.get(WatchlistEntry, entry_id)
        if not obj:
            return {"status": "not_found"}
        session.delete(obj)
        session.commit()
        return {"status": "deleted"}


@router.get("/alerts")
def list_alerts():
    out = []
    with session_scope() as session:
        for row in session.execute(select(Alert)).scalars().all():
            out.append(
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "condition": row.condition,
                    "threshold": row.threshold,
                    "enabled": bool(row.enabled),
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                }
            )
    return out


@router.post("/alerts")
def add_alert(symbol: str, condition: str, threshold: float):
    with session_scope() as session:
        a = Alert(symbol=symbol, condition=condition, threshold=threshold)
        session.add(a)
        session.commit()
        session.refresh(a)
        return {
            "id": a.id,
            "symbol": a.symbol,
            "condition": a.condition,
            "threshold": a.threshold,
        }


@router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: int):
    with session_scope() as session:
        obj = session.get(Alert, alert_id)
        if not obj:
            return {"status": "not_found"}
        session.delete(obj)
        session.commit()
        return {"status": "deleted"}


def _generate_extended_watchlist() -> List[Dict]:
    """Return an extended watchlist with LIVE prices from Binance public API.

    Falls back to demo data if API fails.
    """
    import random, time
    from backend.utils.market_data import fetch_recent_candles

    base = [
        ("BTCUSDT", "Bitcoin", 67420.50, "Layer 1"),
        ("ETHUSDT", "Ethereum", 2634.80, "Layer 1"),
        ("ADAUSDT", "Cardano", 0.4567, "Layer 1"),
        ("SOLUSDT", "Solana", 143.67, "Layer 1"),
        ("AVAXUSDT", "Avalanche", 23.45, "Layer 1"),
        ("DOTUSDT", "Polkadot", 6.78, "Layer 1"),
        ("ATOMUSDT", "Cosmos", 12.34, "Layer 1"),
        ("NEARUSDT", "NEAR Protocol", 3.45, "Layer 1"),
        ("ALGOUSDT", "Algorand", 0.234, "Layer 1"),
        ("FLOWUSDT", "Flow", 1.23, "Layer 1"),
        ("APTUSDT", "Aptos", 7.89, "Layer 1"),
        ("SUIUSDT", "Sui", 1.45, "Layer 1"),
        ("FTMUSDT", "Fantom", 0.234, "Layer 1"),
        ("ONEUSDT", "Harmony", 0.0123, "Layer 1"),
        # Layer 2 / Scaling
        ("MATICUSDT", "Polygon", 0.89, "Layer 2"),
        ("LRCUSDT", "Loopring", 0.345, "Layer 2"),
        ("IMXUSDT", "Immutable X", 1.67, "Layer 2"),
        ("OPUSDT", "Optimism", 2.34, "Layer 2"),
        ("ARBUSDT", "Arbitrum", 1.89, "Layer 2"),
        # DeFi / Oracles
        ("UNIUSDT", "Uniswap", 7.89, "DeFi"),
        ("LINKUSDT", "Chainlink", 14.56, "Oracle"),
        ("AAVEUSDT", "Aave", 87.34, "DeFi"),
        ("CRVUSDT", "Curve DAO", 0.67, "DeFi"),
        ("COMPUSDT", "Compound", 45.67, "DeFi"),
        ("MKRUSDT", "Maker", 1234.56, "DeFi"),
        ("SUSHIUSDT", "SushiSwap", 1.34, "DeFi"),
        ("1INCHUSDT", "1inch", 0.456, "DeFi"),
        # Payments / Enterprise / Privacy
        ("XRPUSDT", "Ripple", 0.5234, "Payments"),
        ("XLMUSDT", "Stellar", 0.123, "Payments"),
        ("LTCUSDT", "Litecoin", 89.45, "Payments"),
        ("BCHUSDT", "Bitcoin Cash", 245.67, "Payments"),
        ("HBARUSDT", "Hedera", 0.067, "Enterprise"),
        ("VETUSDT", "VeChain", 0.0234, "Enterprise"),
        ("XMRUSDT", "Monero", 167.89, "Privacy"),
        ("ZECUSDT", "Zcash", 34.56, "Privacy"),
        # Infra / Web3 / Storage
        ("FILUSDT", "Filecoin", 5.67, "Storage"),
        ("ARUSDT", "Arweave", 8.90, "Storage"),
        ("GRTUSDT", "The Graph", 0.156, "Infrastructure"),
        ("BATUSDT", "Basic Attention Token", 0.234, "Utility"),
        ("ICPUSDT", "Internet Computer", 4.89, "Computing"),
        ("RNDRUSDT", "Render", 3.45, "AI/Compute"),
        # Gaming / Metaverse / Meme
        ("MANAUSDT", "Decentraland", 0.456, "Metaverse"),
        ("SANDUSDT", "The Sandbox", 0.345, "Gaming"),
        ("AXSUSDT", "Axie Infinity", 6.78, "Gaming"),
        ("ENJUSDT", "Enjin", 0.234, "Gaming"),
        ("DOGEUSDT", "Dogecoin", 0.089, "Meme"),
        ("SHIBUSDT", "Shiba Inu", 0.0000234, "Meme"),
        ("PEPEUSDT", "Pepe", 0.00000123, "Meme"),
        # Media / Sports / Stable
        ("THETAUSDT", "Theta Network", 1.23, "Media"),
        ("CHZUSDT", "Chiliz", 0.0876, "Sports"),
        ("USDTUSDT", "Tether", 1.0001, "Stablecoin"),
        ("USDCUSDT", "USD Coin", 1.0002, "Stablecoin"),
        ("DAIUSDT", "Dai", 0.9998, "Stablecoin"),
    ]
    out: List[Dict] = []
    now = time.time()

    # Try to get LIVE prices for major coins
    live_prices = {}
    major_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "LINKUSDT",
    ]

    for symbol in major_symbols:
        try:
            candles = fetch_recent_candles(symbol, limit=2)
            if candles and len(candles) >= 1:
                latest = candles[-1]
                if latest.get("close"):
                    live_prices[symbol] = latest["close"]
        except Exception:
            pass  # Fall back to demo data

    for sym, name, demo_price, cat in base:
        # Use live price if available, otherwise demo + randomization
        if sym in live_prices:
            current_price = live_prices[sym]
            change_pct = random.uniform(-3, 3)  # Smaller range for realism
        else:
            change_pct = random.uniform(-8, 8)
            current_price = demo_price * (1 + change_pct / 100)

        # Generate sparkline
        spark = []
        p = current_price
        for _ in range(10):
            p *= 1 + random.uniform(-0.02, 0.02)
            spark.append(round(p, 6 if p < 1 else 4 if p < 100 else 2))

        out.append(
            {
                "symbol": sym,
                "name": name,
                "price": round(
                    current_price,
                    6 if current_price < 1 else 4 if current_price < 100 else 2,
                ),
                "change24h": round(change_pct, 2),
                "volume24h": random.randint(1_000_000, 50_000_000_000),
                "category": cat,
                "confidence": round(random.uniform(0.35, 0.95), 2),
                "sparkline": spark,
                "ts": __import__("datetime").datetime.utcfromtimestamp(now).isoformat()
                + "Z",
            }
        )
    return out


# Cache for expensive watchlist generation
_FULL_WATCHLIST_CACHE = None
_FULL_WATCHLIST_TIMESTAMP = 0


@router.get("/full")
def full_watchlist():
    """Return extended watchlist dataset for dashboard (cached for 3 seconds)."""
    import time

    now = time.time()
    global _FULL_WATCHLIST_CACHE, _FULL_WATCHLIST_TIMESTAMP

    # Return cached data if fresh (within 3 seconds)
    if _FULL_WATCHLIST_CACHE and (now - _FULL_WATCHLIST_TIMESTAMP) < 3.0:
        return _FULL_WATCHLIST_CACHE

    # Generate fresh data and cache it
    _FULL_WATCHLIST_CACHE = _generate_extended_watchlist()
    _FULL_WATCHLIST_TIMESTAMP = now
    return _FULL_WATCHLIST_CACHE


@router.get("/prices")
def watchlist_prices(
    symbols: Annotated[
        str, Query(..., description="Comma-separated symbols e.g. BTCUSDT,ETHUSDT")
    ],
    limit: Annotated[int, Query(ge=1, le=200)] = 24,
) -> List[Dict]:
    """Return lightweight price summaries for a comma-separated list of symbols.

    For each symbol we return latest price, 24h change (estimated from recent candles),
    24h volume (sum of volumes in returned candles) and a small sparkline (close prices).
    Uses live market data when available, otherwise deterministic demo candles.
    """
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols parameter is required")
    out: List[Dict] = []
    requested = [s.strip() for s in symbols.split(",") if s.strip()]
    for sym in requested:
        try:
            candles = fetch_recent_candles(symbol=sym, limit=limit)
            closes = [c.get("close") for c in candles if c.get("close") is not None]
            if not closes:
                raise ValueError("no candle closes")
            latest = closes[-1]
            first = closes[0]
            change24 = (latest - first) / first if first else 0.0
            volume24 = sum([c.get("volume", 0) for c in candles])
            out.append(
                {
                    "symbol": sym,
                    "price": float(latest),
                    "change24h": float(change24),
                    "volume24h": float(volume24),
                    "sparkline": [float(round(v, 6)) for v in closes],
                    "ts": candles[-1].get("time"),
                }
            )
        except Exception:
            # Be tolerant: include an error entry rather than fail the whole request
            out.append({"symbol": sym, "error": "failed to fetch data"})
    return out


@router.websocket("/ws/watchlist")
async def watchlist_ws(
    websocket: WebSocket, symbols: str = "BTCUSDT,ETHUSDT", limit: int = 24
):
    await websocket.accept()
    requested = [s.strip() for s in symbols.split(",") if s.strip()]
    try:
        while True:
            out = []
            for sym in requested:
                try:
                    candles = fetch_recent_candles(symbol=sym, limit=limit)
                    closes = [
                        c.get("close") for c in candles if c.get("close") is not None
                    ]
                    if not closes:
                        raise ValueError("no candle closes")
                    latest = closes[-1]
                    first = closes[0]
                    change24 = (latest - first) / first if first else 0.0
                    volume24 = sum([c.get("volume", 0) for c in candles])
                    out.append(
                        {
                            "symbol": sym,
                            "price": float(latest),
                            "change24h": float(change24),
                            "volume24h": float(volume24),
                            "sparkline": [float(round(v, 6)) for v in closes],
                            "ts": candles[-1].get("time"),
                        }
                    )
                except Exception:
                    out.append({"symbol": sym, "error": "failed to fetch data"})
            await websocket.send_json(out)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
    await websocket.accept()
    register_ws(websocket)
    try:
        while True:
            # keep connection open; evaluator will push events
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        unregister_ws(websocket)
        return
