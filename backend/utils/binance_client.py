import os
from typing import Any

# Delay import of python-binance to runtime so this module can be imported in
# environments where the package is not installed. We'll try to import it when
# we actually need to instantiate the client.
BinanceAPIClient = None
try:
    # type: ignore
    from binance.client import Client as BinanceAPIClient  # type: ignore
except Exception:
    BinanceAPIClient = None


class BinanceClient:
    # Explicit instance attribute annotations to help static checkers
    client: Any
    mock: bool
    def __init__(self):
        # Use environment variables for credentials. Don't keep secrets in repo.
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        if api_key and api_secret and BinanceAPIClient is not None:
            try:
                self.client: Any = BinanceAPIClient(api_key, api_secret)  # type: ignore[assignment]
                self.mock = False
            except Exception:
                # if client cannot be instantiated, fall back to mock mode
                self.client = None
                self.mock = True
        else:
            self.client = None
            self.mock = True

    def get_price(self, symbol: str) -> dict:
        if self.mock:
            return {"symbol": symbol, "price": 100.0}
        avg_price = self.client.get_avg_price(symbol=symbol)
        return {"symbol": symbol, "price": float(avg_price["price"])}
