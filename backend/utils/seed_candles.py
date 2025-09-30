import datetime
from datetime import timezone
import random

from backend.database import session_scope, Candle


def seed_candles(symbol: str = "BTCUSDT", days: int = 30) -> None:
    now = datetime.datetime.now(timezone.utc)
    price = 20_000.0

    with session_scope() as session:
        session.query(Candle).filter(Candle.symbol == symbol).delete()
        for i in range(days):
            timestamp = now - datetime.timedelta(days=days - i)
            open_price = price
            high_price = open_price * (1 + random.uniform(0.01, 0.03))  # nosec B311
            low_price = open_price * (1 - random.uniform(0.01, 0.03))  # nosec B311
            close_price = random.uniform(low_price, high_price)  # nosec B311
            volume = random.uniform(100, 500)  # nosec B311

            session.add(
                Candle(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )
            )
            price = close_price
