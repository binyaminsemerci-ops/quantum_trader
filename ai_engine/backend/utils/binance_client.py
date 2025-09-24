from typing import Any
import datetime
from backend.utils.trade_logger import log_trade
from config.config import load_config

# Delay import of python-binance so this helper can be imported in CI/test
# environments without the package installed. Instantiate the Client only if
# credentials are available.
try:
    from binance.client import Client  # type: ignore
except Exception:
    Client = None  # type: ignore


cfg = load_config()
API_KEY = cfg.binance_api_key
API_SECRET = cfg.binance_api_secret

client: Any = None
mock = True
if API_KEY and API_SECRET and Client is not None:
    try:
        client = Client(API_KEY, API_SECRET)  # type: ignore
        mock = False
    except Exception:
        client = None
        mock = True


def fetch_recent_trades(symbol: str = "BTCUSDT", limit: int = 5):
    if mock or client is None:
        # Return deterministic mock trades when no credentials are configured
        now = datetime.datetime.utcnow().isoformat()
        mock_trades = [{"symbol": symbol, "qty": "0.01", "price": "100.0", "isBuyerMaker": True} for _ in range(limit)]
        for t in mock_trades:
            log_trade(symbol, "buy" if t["isBuyerMaker"] else "sell", t["qty"], t["price"], 0, 0, now)
        return mock_trades

    trades = client.get_recent_trades(symbol=symbol, limit=limit)
    timestamp = datetime.datetime.utcnow().isoformat()
    for t in trades:
        log_trade(symbol, "buy" if t.get("isBuyerMaker") else "sell", t.get("qty"), t.get("price"), 0, 0, timestamp)
    return trades
