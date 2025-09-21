import os
from typing import Any
from binance.client import Client  # type: ignore


class BinanceClient:
    def __init__(self):
        api_key = os.getenv(
            "rzfgdBpQjHc0SCt7MoXqKxIGzJvKpIyBOaW7FC0KPgRQAcGYijPVwxaUKYVJw76V",
            None,
        )
        api_secret = os.getenv(
            "8JzH7Dxitc6AiqjE15wlZvg713CYkhaYHB2RzrYZPcAQWNZYOt75RywHJ5QErefG",
            None,
        )

        if api_key and api_secret:
            self.client: Any = Client(api_key, api_secret)  # type: ignore[assignment]
            self.mock = False
        else:
            self.client = None
            self.mock = True

    def get_price(self, symbol: str) -> dict:
        if self.mock:
            return {"symbol": symbol, "price": 100.0}
        avg_price = self.client.get_avg_price(symbol=symbol)
        return {"symbol": symbol, "price": float(avg_price["price"]) }
