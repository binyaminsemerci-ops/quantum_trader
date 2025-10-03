"""
Simplified backend server for development
"""

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import sqlite3
from datetime import datetime, timezone
import os
import threading
import time as _time
from ai_auto_trading_service import AIAutoTradingService

# In-memory runtime state for auto-training + auto-trading (lightweight)
runtime_state = {
    "auto_training_enabled": False,
    "auto_training_interval_sec": 600,  # 10 minutes default
    "last_training_start": None,
    "last_training_result": None,
    "training_thread": None,
    "trading_enabled": False,
    "trading_thread": None,
    "trading_cycle_interval_sec": 300,  # 5 minutes default
    "ai_auto_trading_enabled": False,
    "ai_auto_trading_service": None,
}

# Initialize AI Auto Trading Service
try:
    ai_trading_service = AIAutoTradingService()
    runtime_state["ai_auto_trading_service"] = ai_trading_service
    logging.info("AI Auto Trading Service initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize AI Auto Trading Service: {e}")
    runtime_state["ai_auto_trading_service"] = None

_STATE_LOCK = threading.Lock()


def _safe_log(msg: str):
    try:
        logger.info(msg)
    except Exception:
        pass


def _run_periodic_training():
    """Background loop that retrains every N seconds when enabled."""
    while True:
        with _STATE_LOCK:
            enabled = runtime_state["auto_training_enabled"]
            interval = runtime_state["auto_training_interval_sec"]
        if not enabled:
            _safe_log("[AUTO-TRAIN] Disabled -> exiting training thread")
            break
        start_ts = datetime.now(timezone.utc).isoformat()
        _safe_log(f"[AUTO-TRAIN] Starting training cycle at {start_ts}")
        try:
            # Lazy import to avoid heavy cost if unused
            from ai_engine.train_and_save import train_and_save

            result = train_and_save(limit=1200, backtest=False, write_report=True)
            with _STATE_LOCK:
                runtime_state["last_training_start"] = start_ts
                runtime_state["last_training_result"] = {
                    "metrics": result.get("metrics"),
                    "num_samples": (
                        result.get("backtest", {}).get("trades")
                        if result.get("backtest")
                        else result.get("metrics")
                    ),
                    "saved": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            _safe_log("[AUTO-TRAIN] Training cycle completed")
        except Exception as e:  # noqa: BLE001
            _safe_log(f"[AUTO-TRAIN] Training error: {e}")
            with _STATE_LOCK:
                runtime_state["last_training_result"] = {
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        # Sleep until next cycle
        for _ in range(interval):
            with _STATE_LOCK:
                if not runtime_state["auto_training_enabled"]:
                    break
            _time.sleep(1)
        with _STATE_LOCK:
            if not runtime_state["auto_training_enabled"]:
                break


def _start_training_thread():
    with _STATE_LOCK:
        if (
            runtime_state["training_thread"]
            and runtime_state["training_thread"].is_alive()
        ):
            return
        runtime_state["auto_training_enabled"] = True
        t = threading.Thread(target=_run_periodic_training, daemon=True)
        runtime_state["training_thread"] = t
        t.start()
        _safe_log("[AUTO-TRAIN] Background training thread started")


def _stop_training_thread():
    with _STATE_LOCK:
        runtime_state["auto_training_enabled"] = False


def _run_trading_loop():
    """Placeholder loop for future real trading integration (every 5m)."""
    while True:
        with _STATE_LOCK:
            enabled = runtime_state["trading_enabled"]
            interval = runtime_state["trading_cycle_interval_sec"]
        if not enabled:
            _safe_log("[AUTO-TRADE] Disabled -> exiting trading thread")
            break
        loop_start = datetime.now(timezone.utc).isoformat()
        _safe_log(f"[AUTO-TRADE] Running trading cycle at {loop_start} (demo mode)")
        # TODO: integrate real engine (binance_trading.BinanceTradeEngine) when keys configured
        # Sleep for interval
        for _ in range(interval):
            with _STATE_LOCK:
                if not runtime_state["trading_enabled"]:
                    break
            _time.sleep(1)
        with _STATE_LOCK:
            if not runtime_state["trading_enabled"]:
                break


def _start_trading_thread():
    with _STATE_LOCK:
        if (
            runtime_state["trading_thread"]
            and runtime_state["trading_thread"].is_alive()
        ):
            return
        runtime_state["trading_enabled"] = True
        t = threading.Thread(target=_run_trading_loop, daemon=True)
        runtime_state["trading_thread"] = t
        t.start()
        _safe_log("[AUTO-TRADE] Background trading thread started (demo)")


def _stop_trading_thread():
    with _STATE_LOCK:
        runtime_state["trading_enabled"] = False


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_START_TIME = _time.time()
app = FastAPI(title="Quantum Trader API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "quantum_trader.db")


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize database with basic tables
def init_db():
    """Initialize database with basic tables"""
    conn = get_db_connection()

    # Create trades table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            pnl REAL DEFAULT 0
        )
    """
    )

    # Insert some sample data if empty
    count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if count == 0:
        sample_trades = [
            ("BTCUSDC", "BUY", 0.1, 65000, "2025-01-01 10:00:00", 1250),
            ("ETHUSDC", "SELL", 2.0, 3200, "2025-01-01 11:00:00", -150),
            ("ADAUSDC", "BUY", 1000, 0.45, "2025-01-01 12:00:00", 50),
        ]
        conn.executemany(
            "INSERT INTO trades (symbol, side, quantity, price, timestamp, pnl) VALUES (?, ?, ?, ?, ?, ?)",
            sample_trades,
        )

    conn.commit()
    conn.close()


# Routes


@app.get("/")
async def root():
    return {"message": "Quantum Trader API", "status": "running"}


@app.get("/api/v1/system/status")
@app.get("/system/status")  # legacy / unversioned
async def system_status():
    """Lightweight system & runtime status for dashboard widget."""
    now = _time.time()
    with _STATE_LOCK:
        training = {
            "enabled": runtime_state["auto_training_enabled"],
            "interval_sec": runtime_state["auto_training_interval_sec"],
            "last_start": runtime_state["last_training_start"],
            "last_result": runtime_state["last_training_result"],
        }
        trading = {
            "enabled": runtime_state["trading_enabled"],
            "interval_sec": runtime_state["trading_cycle_interval_sec"],
        }
    cache_age = None
    try:
        if market_data_cache.get("last_updated"):
            cache_age = now - market_data_cache["last_updated"]
    except Exception:  # noqa: BLE001
        cache_age = None
    # Load config (optional) to expose exchange capability flags without leaking secrets
    has_binance_keys = None
    binance_testnet = None
    real_trading_enabled = None
    try:
        from config.config import load_config  # type: ignore

        cfg = load_config()
        has_binance_keys = bool(cfg.binance_api_key and cfg.binance_api_secret)
        binance_testnet = bool(getattr(cfg, "binance_use_testnet", False))
        real_trading_enabled = bool(getattr(cfg, "enable_real_trading", False))
    except Exception:  # pragma: no cover
        pass
    return {
        "service": "quantum_trader_simple",
        "version": app.version,
        "uptime_seconds": round(now - _START_TIME, 2),
        "training": training,
        "trading": trading,
        "market_data_cache_age_sec": (
            round(cache_age, 2) if cache_age is not None else None
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": os.environ.get("ENV", "dev"),
        "git_sha": os.environ.get("GIT_SHA"),
        "has_binance_keys": has_binance_keys,
        "binance_testnet": binance_testnet,
        "real_trading_enabled": real_trading_enabled,
    }


@app.get("/api/v1/portfolio")
@app.get("/portfolio")  # Legacy endpoint
async def get_portfolio():
    """Get portfolio data"""
    conn = get_db_connection()

    # Calculate portfolio metrics from trades
    trades = conn.execute(
        """
        SELECT symbol, SUM(CASE WHEN side='BUY' THEN quantity ELSE -quantity END) as position,
               AVG(price) as avg_price, SUM(pnl) as total_pnl
        FROM trades 
        GROUP BY symbol
        HAVING position != 0
    """
    ).fetchall()

    # More realistic portfolio values
    total_pnl = sum(trade["total_pnl"] for trade in trades) if trades else 0
    portfolio_value = 125000.50  # More realistic portfolio size
    day_pnl = 2850.75  # Realistic daily P&L

    positions = []
    for trade in trades:
        # Use more realistic current prices
        if trade["symbol"] == "BTCUSDC":
            current_price = 63500.0  # Current BTC price
        elif trade["symbol"] == "ETHUSDC":
            current_price = 3150.0  # Current ETH price
        elif trade["symbol"] == "ADAUSDC":
            current_price = 0.475  # Current ADA price
        else:
            current_price = trade["avg_price"] * 1.02

        market_value = abs(trade["position"]) * current_price
        unrealized_pnl = (current_price - trade["avg_price"]) * trade["position"]

        positions.append(
            {
                "symbol": trade["symbol"],
                "quantity": round(trade["position"], 6),
                "avgPrice": round(trade["avg_price"], 2),
                "currentPrice": round(current_price, 2),
                "marketValue": round(market_value, 2),
                "unrealizedPnL": round(unrealized_pnl, 2),
                "realizedPnL": round(trade["total_pnl"], 2),
            }
        )

    # Add additional positions to make portfolio more realistic
    if len(positions) < 3:
        positions.extend(
            [
                {
                    "symbol": "BTCUSDC",
                    "quantity": 1.85,
                    "avgPrice": 61200.0,
                    "currentPrice": 63500.0,
                    "marketValue": 117475.0,
                    "unrealizedPnL": 4255.0,
                    "realizedPnL": 1250.0,
                },
                {
                    "symbol": "ETHUSDC",
                    "quantity": 12.5,
                    "avgPrice": 3080.0,
                    "currentPrice": 3150.0,
                    "marketValue": 39375.0,
                    "unrealizedPnL": 875.0,
                    "realizedPnL": -150.0,
                },
                {
                    "symbol": "SOLUSDC",
                    "quantity": 45.0,
                    "avgPrice": 195.0,
                    "currentPrice": 205.0,
                    "marketValue": 9225.0,
                    "unrealizedPnL": 450.0,
                    "realizedPnL": 75.0,
                },
            ]
        )

    total_market_value = sum(pos["marketValue"] for pos in positions)
    total_unrealized_pnl = sum(pos["unrealizedPnL"] for pos in positions)
    cash_balance = portfolio_value - total_market_value

    conn.close()

    return {
        "totalValue": round(portfolio_value, 2),
        "totalPnL": round(total_unrealized_pnl + total_pnl, 2),
        "dayPnL": round(day_pnl, 2),
        "positions": positions,
        "cash": round(cash_balance, 2),
        "marketValue": round(total_market_value, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/portfolio/pnl")
@app.get("/portfolio/pnl")  # Legacy endpoint
async def get_pnl():
    """Get P&L data"""
    conn = get_db_connection()

    # Get total P&L from trades and calculate realistic metrics
    result = conn.execute(
        "SELECT SUM(pnl) as total_pnl, COUNT(*) as trade_count FROM trades"
    ).fetchone()
    base_pnl = result["total_pnl"] or 0
    trade_count = result["trade_count"] or 1

    # More realistic P&L data
    total_pnl = 5850.25  # Total unrealized + realized P&L
    day_pnl = 2850.75  # Today's P&L
    week_pnl = 4200.50  # This week's P&L
    month_pnl = total_pnl  # This month's total

    # Calculate realistic trading stats
    win_trades = max(1, int(trade_count * 0.68))
    total_trades = max(trade_count, 25)
    win_rate = (win_trades / total_trades) * 100

    conn.close()

    return {
        "totalPnL": round(total_pnl, 2),
        "dayPnL": round(day_pnl, 2),
        "weekPnL": round(week_pnl, 2),
        "monthPnL": round(month_pnl, 2),
        "winRate": round(win_rate, 1),
        "profitFactor": 1.85,
        "sharpeRatio": 1.42,
        "maxDrawdown": -8.5,
        "totalTrades": total_trades,
        "winningTrades": win_trades,
        "avgWin": round(total_pnl / win_trades, 2),
        "avgLoss": round(-450.25, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Cache for market data to avoid rate limiting
market_data_cache = {
    "data": None,
    "last_updated": 0,
    "cache_duration": 60,  # Cache for 60 seconds
}


@app.get("/api/v1/portfolio/market-overview")
@app.get("/portfolio/market-overview")  # Legacy endpoint
async def get_market_overview():
    """Get market overview data with LIVE prices"""
    import aiohttp
    import time

    # Check cache first to avoid rate limiting
    current_time = time.time()
    if (
        market_data_cache["data"] is not None
        and current_time - market_data_cache["last_updated"]
        < market_data_cache["cache_duration"]
    ):
        logger.info("Returning cached market data")
        return market_data_cache["data"]

    try:
        # Fetch real-time prices from CoinGecko API (free, no API key needed)
        logger.info("Fetching fresh market data from CoinGecko")
        async with aiohttp.ClientSession() as session:
            # Get live prices for multiple coins
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": "bitcoin,ethereum,cardano,solana,polkadot",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    symbols = [
                        {
                            "symbol": "BTCUSDC",
                            "price": data.get("bitcoin", {}).get("usd", 65000),
                            "change24h": data.get("bitcoin", {}).get(
                                "usd_24h_change", 0
                            ),
                            "volume24h": data.get("bitcoin", {}).get("usd_24h_vol", 0),
                        },
                        {
                            "symbol": "ETHUSDC",
                            "price": data.get("ethereum", {}).get("usd", 3200),
                            "change24h": data.get("ethereum", {}).get(
                                "usd_24h_change", 0
                            ),
                            "volume24h": data.get("ethereum", {}).get("usd_24h_vol", 0),
                        },
                        {
                            "symbol": "ADAUSDC",
                            "price": data.get("cardano", {}).get("usd", 0.45),
                            "change24h": data.get("cardano", {}).get(
                                "usd_24h_change", 0
                            ),
                            "volume24h": data.get("cardano", {}).get("usd_24h_vol", 0),
                        },
                        {
                            "symbol": "SOLUSDC",
                            "price": data.get("solana", {}).get("usd", 155),
                            "change24h": data.get("solana", {}).get(
                                "usd_24h_change", 0
                            ),
                            "volume24h": data.get("solana", {}).get("usd_24h_vol", 0),
                        },
                        {
                            "symbol": "DOTUSDC",
                            "price": data.get("polkadot", {}).get("usd", 7.5),
                            "change24h": data.get("polkadot", {}).get(
                                "usd_24h_change", 0
                            ),
                            "volume24h": data.get("polkadot", {}).get("usd_24h_vol", 0),
                        },
                    ]
                else:
                    # Fallback to demo data if API fails
                    symbols = [
                        {
                            "symbol": "BTCUSDC",
                            "price": 65000,
                            "change24h": 0,
                            "volume24h": 0,
                            "note": "API_ERROR",
                        },
                    ]

    except Exception as e:
        logger.warning(f"Failed to fetch live prices: {e}")
        # Fallback to realistic simulated data (rate limit recovery)
        import random

        base_prices = {
            "bitcoin": 114400,
            "ethereum": 4145,
            "cardano": 0.80,
            "solana": 209,
            "polkadot": 3.91,
        }

        symbols = []
        for coin, base_price in base_prices.items():
            # Add small random variation (±0.5%) to simulate market movement
            price_variation = (random.random() - 0.5) * 0.01 * base_price  # ±0.5%
            current_price = base_price + price_variation
            change_24h = (random.random() - 0.5) * 4  # ±2% daily change

            symbol_mapping = {
                "bitcoin": "BTCUSDC",
                "ethereum": "ETHUSDC",
                "cardano": "ADAUSDC",
                "solana": "SOLUSDC",
                "polkadot": "DOTUSDC",
            }

            symbols.append(
                {
                    "symbol": symbol_mapping[coin],
                    "price": round(current_price, 2 if current_price > 1 else 6),
                    "change24h": round(change_24h, 2),
                    "volume24h": random.randint(
                        50000000, 200000000
                    ),  # Realistic volume
                    "note": "SIMULATED",
                }
            )

    # Calculate top gainers and losers
    top_gainers = [coin for coin in symbols if coin["change24h"] > 0]
    top_gainers.sort(key=lambda x: x["change24h"], reverse=True)
    top_gainers = top_gainers[:3]  # Top 3 gainers

    top_losers = [coin for coin in symbols if coin["change24h"] < 0]
    top_losers.sort(
        key=lambda x: x["change24h"]
    )  # Sort ascending (most negative first)
    top_losers = top_losers[:3]  # Top 3 losers

    # Calculate total market volume
    total_volume = sum(coin.get("volume24h", 0) for coin in symbols)

    result = {
        "marketCap": 2500000000000,  # Frontend expects "marketCap" not "totalMarketCap"
        "volume24h": total_volume,
        "dominance": {"btc": 52.3, "eth": 17.8},
        "fearGreedIndex": 75,
        "topGainers": [
            {
                "symbol": coin["symbol"].replace(
                    "USDC", ""
                ),  # Remove USDC suffix for display
                "change": coin["change24h"],
                "price": coin["price"],
            }
            for coin in top_gainers
        ],
        "topLosers": [
            {
                "symbol": coin["symbol"].replace(
                    "USDC", ""
                ),  # Remove USDC suffix for display
                "change": coin["change24h"],
                "price": coin["price"],
            }
            for coin in top_losers
        ],
        "symbols": symbols,  # Keep original symbols array for backward compatibility
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Cache the result
    market_data_cache["data"] = result
    market_data_cache["last_updated"] = current_time
    logger.info("Market data cached successfully")

    return result


@app.get("/api/v1/stats")
@app.get("/stats")  # Legacy endpoint
async def get_stats():
    """Get trading statistics"""
    conn = get_db_connection()

    # Calculate realistic trading statistics
    result = conn.execute(
        "SELECT COUNT(*) as count, SUM(quantity * price) as volume FROM trades"
    ).fetchone()
    trades_count = result["count"] or 0
    total_volume = result["volume"] or 0

    # Enhanced trading statistics
    total_trades = max(trades_count, 156)  # Realistic trade count
    total_volume = max(total_volume, 2450000.0)  # 2.45M USDC volume
    avg_trade_size = total_volume / total_trades

    conn.close()

    return {
        "totalTrades": total_trades,
        "totalVolume": round(total_volume, 2),
        "averageTradeSize": round(avg_trade_size, 2),
        "winRate": 68.5,
        "profitFactor": 1.85,
        "maxDrawdown": -8.2,
        "sharpeRatio": 1.42,
        "calmarRatio": 0.95,
        "avgHoldTime": "4.2h",
        "bestTrade": 1850.50,
        "worstTrade": -420.75,
        "consecutiveWins": 7,
        "consecutiveLosses": 3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/trading/status")
@app.get("/trading/status")  # Legacy endpoint
async def get_trading_status():
    """Get trading engine status"""
    return {
        "is_running": False,  # Mock - not running yet
        "ai_model_loaded": False,
        "trading_symbols_count": 5,
        "balances": {"USDC": 25000.50, "USDT": 8500.25, "BTC": 0.15, "ETH": 2.5},
        "recent_trades": [
            {
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.05,
                "price": 65200.0,
                "timestamp": "2025-01-01T15:30:00Z",
            }
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/trading/start")
@app.post("/trading/start")  # Legacy endpoint
async def start_trading(interval_minutes: int = 5):
    """Start AI trading engine"""
    # Enable background trading loop (demo)
    with _STATE_LOCK:
        runtime_state["trading_cycle_interval_sec"] = interval_minutes * 60
    _start_trading_thread()
    return {
        "message": "Trading engine started (demo mode - placeholder)",
        "interval_minutes": interval_minutes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "demo",
    }


@app.post("/api/v1/trading/stop")
@app.post("/trading/stop")  # Legacy endpoint
async def stop_trading():
    """Stop AI trading engine"""
    _stop_trading_thread()
    return {
        "message": "Trading engine stop requested",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/trading/run-cycle")
@app.post("/trading/run-cycle")  # Legacy endpoint
async def run_trading_cycle():
    """Run one trading cycle"""
    # Manual single cycle placeholder
    _safe_log("[AUTO-TRADE] Manual trading cycle trigger (demo)")
    return {
        "message": "Trading cycle completed (demo manual)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/training/run-once")
async def trigger_training(background_tasks: BackgroundTasks, limit: int = 1200):
    """Trigger one training run asynchronously."""

    def _run():  # executed in background
        _safe_log(f"[TRAIN-ONCE] Triggered model training limit={limit}")
        try:
            from ai_engine.train_and_save import train_and_save

            res = train_and_save(limit=limit, backtest=False, write_report=True)
            with _STATE_LOCK:
                runtime_state["last_training_start"] = datetime.now(
                    timezone.utc
                ).isoformat()
                runtime_state["last_training_result"] = res.get("metrics", {})
        except Exception as e:  # noqa: BLE001
            _safe_log(f"[TRAIN-ONCE] Error: {e}")
            with _STATE_LOCK:
                runtime_state["last_training_result"] = {"error": str(e)}

    background_tasks.add_task(_run)
    return {
        "message": "Training started",
        "limit": limit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/training/auto/enable")
async def enable_auto_training(interval_minutes: int = 10):
    """Enable continuous auto-training every interval_minutes."""
    with _STATE_LOCK:
        runtime_state["auto_training_interval_sec"] = max(60, interval_minutes * 60)
    _start_training_thread()
    return {
        "message": "Auto-training enabled",
        "interval_minutes": interval_minutes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/training/auto/disable")
async def disable_auto_training():
    """Disable continuous auto-training."""
    _stop_training_thread()
    return {
        "message": "Auto-training disabled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/training/status")
async def training_status():
    """Get current training background status and last metrics."""
    with _STATE_LOCK:
        status = {
            "auto_training_enabled": runtime_state["auto_training_enabled"],
            "interval_sec": runtime_state["auto_training_interval_sec"],
            "last_training_start": runtime_state["last_training_start"],
            "last_training_result": runtime_state["last_training_result"],
        }
    return status


@app.post("/api/v1/trading/auto/enable")
async def enable_auto_trading(interval_minutes: int = 5):
    """Enable continuous demo trading loop."""
    with _STATE_LOCK:
        runtime_state["trading_cycle_interval_sec"] = max(60, interval_minutes * 60)
    _start_trading_thread()
    return {
        "message": "Auto-trading enabled (demo)",
        "interval_minutes": interval_minutes,
    }


@app.post("/api/v1/trading/auto/disable")
async def disable_auto_trading():
    """Disable auto trading loop."""
    _stop_trading_thread()
    return {"message": "Auto-trading disabled"}


@app.get("/api/v1/trading/auto/status")
async def auto_trading_status():
    """Return current auto-trading runtime status."""
    with _STATE_LOCK:
        st = {
            "trading_enabled": runtime_state["trading_enabled"],
            "interval_sec": runtime_state["trading_cycle_interval_sec"],
        }
    return st


@app.post("/api/v1/trading/update-config")
@app.post("/trading/update-config")  # Legacy endpoint
async def update_config(config: dict):
    """Update trading configuration"""
    return {
        "message": "Configuration updated",
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# AI Auto Trading Endpoints


@app.get("/api/v1/ai-trading/status")
async def get_ai_trading_status():
    """Get AI auto trading status and performance metrics"""
    try:
        with _STATE_LOCK:
            ai_service = runtime_state.get("ai_auto_trading_service")
            enabled = runtime_state.get("ai_auto_trading_enabled", False)

        if not ai_service:
            return {
                "enabled": False,
                "error": "AI Auto Trading Service not available",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        status = ai_service.get_status()
        status.update(
            {"enabled": enabled, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        return status

    except Exception as e:
        logger.error(f"Error getting AI trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ai-trading/start")
async def start_ai_trading(symbols: list[str] = None):
    """Start AI auto trading for specified symbols"""
    try:
        with _STATE_LOCK:
            ai_service = runtime_state.get("ai_auto_trading_service")

        if not ai_service:
            raise HTTPException(
                status_code=503, detail="AI Auto Trading Service not available"
            )

        if symbols is None:
            symbols = ["BTCUSDC", "ETHUSDC"]  # Default symbols

        result = ai_service.start_trading(symbols)

        with _STATE_LOCK:
            runtime_state["ai_auto_trading_enabled"] = True

        return {
            "message": "AI auto trading started",
            "symbols": symbols,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error starting AI trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ai-trading/stop")
async def stop_ai_trading():
    """Stop AI auto trading"""
    try:
        with _STATE_LOCK:
            ai_service = runtime_state.get("ai_auto_trading_service")

        if not ai_service:
            raise HTTPException(
                status_code=503, detail="AI Auto Trading Service not available"
            )

        result = ai_service.stop_trading()

        with _STATE_LOCK:
            runtime_state["ai_auto_trading_enabled"] = False

        return {
            "message": "AI auto trading stopped",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error stopping AI trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ai-trading/signals")
async def get_ai_trading_signals(symbol: str = None, limit: int = 10):
    """Get recent AI trading signals"""
    try:
        with _STATE_LOCK:
            ai_service = runtime_state.get("ai_auto_trading_service")

        if not ai_service:
            raise HTTPException(
                status_code=503, detail="AI Auto Trading Service not available"
            )

        signals = ai_service.get_recent_signals(symbol=symbol, limit=limit)

        return {
            "signals": (
                signals
                if isinstance(signals[0] if signals else {}, dict)
                else [signal.to_dict() for signal in signals]
            ),
            "symbol": symbol,
            "count": len(signals),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting AI trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ai-trading/executions")
async def get_ai_trading_executions(symbol: str = None, limit: int = 10):
    """Get recent AI trade executions"""
    try:
        with _STATE_LOCK:
            ai_service = runtime_state.get("ai_auto_trading_service")

        if not ai_service:
            raise HTTPException(
                status_code=503, detail="AI Auto Trading Service not available"
            )

        executions = ai_service.get_recent_executions(symbol=symbol, limit=limit)

        return {
            "executions": (
                executions
                if isinstance(executions[0] if executions else {}, dict)
                else [execution.to_dict() for execution in executions]
            ),
            "symbol": symbol,
            "count": len(executions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting AI trading executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ai-trading/config")
async def update_ai_trading_config(config: dict):
    """Update AI trading configuration"""
    try:
        with _STATE_LOCK:
            ai_service = runtime_state.get("ai_auto_trading_service")

        if not ai_service:
            raise HTTPException(
                status_code=503, detail="AI Auto Trading Service not available"
            )

        # Update configuration
        if "position_size" in config:
            ai_service.config["position_size"] = float(config["position_size"])
        if "stop_loss_pct" in config:
            ai_service.config["stop_loss_pct"] = float(config["stop_loss_pct"])
        if "take_profit_pct" in config:
            ai_service.config["take_profit_pct"] = float(config["take_profit_pct"])
        if "min_confidence" in config:
            ai_service.config["min_confidence"] = float(config["min_confidence"])
        if "max_positions" in config:
            ai_service.config["max_positions"] = int(config["max_positions"])
        if "risk_limit" in config:
            ai_service.config["risk_limit"] = float(config["risk_limit"])

        return {
            "message": "AI trading configuration updated",
            "config": ai_service.config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error updating AI trading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stress/summary")
@app.get("/stress/summary")  # Legacy endpoint for frontend compatibility
async def get_stress_summary():
    """Get stress test summary"""
    return {
        "status": "completed",
        "source": "AI Trading",
        "started_at": "2025-01-01T10:00:00Z",
        "finished_at": "2025-01-01T10:30:00Z",
        "iterations": 150,
        "duration": {"min": 45.2, "max": 125.8, "avg": 78.5},
        "totals": {"runs": 150},
        "duration_series": [45, 52, 67, 78, 89, 95, 102, 125],
        "tasks": [
            {
                "name": "AI Prediction",
                "counts": {"ok": 142, "fail": 5, "skipped": 2, "error": 1},
                "pass_rate": 94.7,
                "trend": [90, 92, 94, 95, 94],
            },
            {
                "name": "Trade Execution",
                "counts": {"ok": 138, "fail": 8, "skipped": 3, "error": 1},
                "pass_rate": 92.0,
                "trend": [89, 91, 90, 92, 92],
            },
        ],
        "recent_runs": [
            {
                "iteration": 150,
                "summary": {"trades": 5},
                "total_duration": 78.2,
                "ts": "2025-01-01T10:30:00Z",
            }
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# WebSocket endpoints
import asyncio


@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(2)
            data = {
                "type": "update",
                "data": {
                    "portfolio": await get_portfolio(),
                    "pnl": await get_pnl(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.websocket("/ws/ai-trading")
async def ai_trading_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time AI trading updates"""
    await websocket.accept()
    try:
        last_signal_id = 0
        last_execution_id = 0

        while True:
            await asyncio.sleep(1)  # Check every second for updates

            with _STATE_LOCK:
                ai_service = runtime_state.get("ai_auto_trading_service")
                enabled = runtime_state.get("ai_auto_trading_enabled", False)

            if not ai_service:
                continue

            # Get new signals since last check
            recent_signals = ai_service.get_recent_signals(limit=5)
            new_signals = [s for s in recent_signals if s.get("id", 0) > last_signal_id]
            if new_signals:
                last_signal_id = max(s.get("id", 0) for s in new_signals)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "signals",
                            "data": (
                                new_signals
                                if isinstance(
                                    new_signals[0] if new_signals else {}, dict
                                )
                                else [signal.to_dict() for signal in new_signals]
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )

            # Get new executions since last check
            recent_executions = ai_service.get_recent_executions(limit=5)
            new_executions = [
                e for e in recent_executions if e.get("id", 0) > last_execution_id
            ]
            if new_executions:
                last_execution_id = max(e.get("id", 0) for e in new_executions)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "executions",
                            "data": (
                                new_executions
                                if isinstance(
                                    new_executions[0] if new_executions else {}, dict
                                )
                                else [
                                    execution.to_dict() for execution in new_executions
                                ]
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )

            # Send status update periodically (every 10 seconds)
            if int(_time.time()) % 10 == 0:
                status = ai_service.get_status()
                status.update({"enabled": enabled})
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "data": status,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info("AI Trading WebSocket client disconnected")
    except Exception as e:
        logger.error(f"AI Trading WebSocket error: {e}")
        await websocket.close()


@app.get("/api/v1/signals/recent")
@app.get("/signals/recent")  # Legacy endpoint
async def get_recent_signals(symbol: str = "BTCUSDC", limit: int = 20):
    """Get recent trading signals"""
    # Mock trading signals
    import random

    signals = []
    symbols_list = ["BTC", "ETH", "SOL", "ADA", "XRP"]
    actions = ["BUY", "SELL"]

    for i in range(limit):
        signals.append(
            {
                "symbol": random.choice(symbols_list),
                "action": random.choice(actions),
                "confidence": random.randint(40, 99),
                "timestamp": f"{random.randint(1, 60)}s",
                "price": random.uniform(0.1, 70000),
                "strength": random.choice(["STRONG", "MEDIUM", "WEAK"]),
            }
        )

    return {
        "signals": signals,
        "symbol": symbol,
        "count": len(signals),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/chart/data")
@app.get("/chart/data")  # Legacy endpoint
async def get_chart_data(
    symbol: str = "BTCUSDC", interval: str = "1h", limit: int = 100
):
    """Get chart/candlestick data"""
    # Mock candlestick data
    import random

    base_price = 63500.0 if symbol == "BTCUSDC" else 3150.0

    candles = []
    for i in range(limit):
        open_price = base_price + random.uniform(-500, 500)
        close_price = open_price + random.uniform(-100, 100)
        high_price = max(open_price, close_price) + random.uniform(0, 50)
        low_price = min(open_price, close_price) - random.uniform(0, 50)

        candles.append(
            {
                "timestamp": datetime.now(timezone.utc).timestamp() - (i * 3600),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": random.uniform(10, 1000),
            }
        )

    return {
        "symbol": symbol,
        "interval": interval,
        "data": candles[::-1],  # Reverse to get chronological order
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Initialize database immediately
init_db()
logger.info("Database initialized")


@app.get("/api/v1/risk/overview")
async def get_risk_overview():
    """Provide consolidated risk metrics for frontend dashboard.

    This is a lightweight synthetic implementation using current portfolio data
    and mock risk modeling – can later be replaced with real analytics.
    """
    try:
        portfolio = await get_portfolio()
        positions = portfolio.get("positions", [])
        total_value = portfolio.get("totalValue", 0) or 1
        # Build exposure per symbol
        exposures = []
        total_exposure_pct = 0.0
        for pos in positions:
            mv = pos.get("marketValue", 0)
            pct = (mv / total_value) * 100
            exposures.append(
                {
                    "symbol": pos.get("symbol"),
                    "marketValue": mv,
                    "exposurePct": round(pct, 2),
                    "unrealizedPnL": pos.get("unrealizedPnL", 0),
                }
            )
            total_exposure_pct += pct

        # Derive basic risk signals
        largest_position = max(exposures, key=lambda x: x["exposurePct"], default=None)
        concentration_risk = (
            largest_position["exposurePct"] if largest_position else 0
        ) > 40
        # Heuristic volatility estimate (placeholder)
        volatility = 0.32  # 32% annualized synthetic
        # Simple VaR approximation (95%) using normal assumption + placeholder
        var95 = -round(total_value * 0.012, 2)
        margin_used_pct = round(min(100.0, total_exposure_pct * 0.85), 2)
        leverage = round(1 + (margin_used_pct / 140), 2)
        drawdown_risk = (
            "LOW"
            if margin_used_pct < 40
            else ("MEDIUM" if margin_used_pct < 70 else "HIGH")
        )
        correlation_risk = "MEDIUM" if concentration_risk else "LOW"
        risk_score = min(
            100,
            int(
                margin_used_pct * 0.6
                + (volatility * 100) * 0.25
                + (10 if correlation_risk != "LOW" else 0)
            ),
        )

        # Limits (static placeholders – to be externalized later)
        limits = {
            "max_daily_loss": round(total_value * 0.035, 2),
            "max_drawdown_allowed": 0.20,
            "max_symbol_exposure_pct": 35,
            "max_portfolio_leverage": 3.0,
        }

        payload = {
            "riskScore": risk_score,
            "volatility": volatility,
            "var95": var95,
            "exposureTotalPct": round(total_exposure_pct, 2),
            "leverage": leverage,
            "marginUsed": margin_used_pct,
            "drawdownRisk": drawdown_risk,
            "correlationRisk": correlation_risk,
            "liquidationPrice": None,  # Only meaningful for leveraged futures positions
            "limits": limits,
            "exposures": exposures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return payload
    except Exception as e:  # noqa: BLE001
        logger.error(f"Risk overview generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute risk overview")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Quantum Trader API on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
