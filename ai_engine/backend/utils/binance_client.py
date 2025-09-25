from typing import Any
import datetime
from backend.utils.trade_logger import log_trade
from config.config import load_config
from backend.utils.exchanges import get_exchange_client


cfg = load_config()
# Let the exchange factory pick the configured default exchange
client = get_exchange_client(api_key=cfg.binance_api_key, api_secret=cfg.binance_api_secret)


def fetch_recent_trades(symbol: str = "BTCUSDC", limit: int = 5):
    # Adapter will return mock trades when no credentials are configured
    trades = client.fetch_recent_trades(symbol=symbol, limit=limit)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for t in trades:
        try:
            log_trade(symbol, "buy" if t.get("isBuyerMaker") else "sell", t.get("qty"), t.get("price"), 0, 0, timestamp)
        except Exception:
            # best-effort logging; don't fail data fetch
            pass
    return trades
