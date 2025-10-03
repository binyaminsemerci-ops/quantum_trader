import os
from binance.client import Client
from backend.utils.trade_logger import log_trade
import datetime

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)


def fetch_recent_trades(symbol="BTCUSDT", limit=5):
    trades = client.get_recent_trades(symbol=symbol, limit=limit)
    timestamp = datetime.datetime.utcnow().isoformat()
    for t in trades:
        log_trade(
            symbol,
            "buy" if t["isBuyerMaker"] else "sell",
            t["qty"],
            t["price"],
            0,
            0,
            timestamp,
        )
    return trades
