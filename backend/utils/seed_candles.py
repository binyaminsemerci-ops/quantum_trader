import datetime
import random
from backend.database import get_db


def seed_candles(symbol="BTCUSDT", days=30):
    db = get_db()
    cursor = db.cursor()

    now = datetime.datetime.utcnow()
    price = 20000  # startpris

    for i in range(days):
        date = (now - datetime.timedelta(days=days - i)).strftime("%Y-%m-%d")
        open_price = price
        # The seed data here is non-cryptographic demo/test data. Bandit flags
        # use of the stdlib random module (B311). Silence that using nosec
        # as this code is intentionally non-crypto and used only for seeding.
        high_price = open_price * (1 + random.uniform(0.01, 0.03))  # nosec: B311
        low_price = open_price * (1 - random.uniform(0.01, 0.03))   # nosec: B311
        close_price = random.uniform(low_price, high_price)         # nosec: B311
        volume = random.uniform(100, 500)                           # nosec: B311

        cursor.execute(
            """
            INSERT INTO candles (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, date, open_price, high_price, low_price, close_price, volume),
        )

        price = close_price

    db.commit()
