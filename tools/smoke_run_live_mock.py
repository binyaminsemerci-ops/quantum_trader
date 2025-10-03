"""Smoke-run trainer with a mocked ccxt.binance to exercise --use-live-data path.

This script injects a fake `ccxt` module into sys.modules that provides a
`binance` factory returning an object with `fetch_ohlcv` method. It then
imports and calls `ai_engine.train_and_save` with `use_live_data=True` so the
live-features code path is executed without reaching the network.
"""
from __future__ import annotations

import sys
import types
import math
import datetime


class FakeExchange:
    def __init__(self, *args, **kwargs):
        self._now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=200):
        # return (ts_ms, open, high, low, close, vol) tuples
        out = []
        base = 100_000.0 if symbol.upper().startswith("BTC") else 100.0
        for i in range(limit):
            ts = self._now - (limit - i) * 60 * 1000
            open_p = base + math.sin(i / 3.0) * 50.0
            close_p = open_p + ((-1) ** i) * 2.0
            high_p = max(open_p, close_p) + 1.0
            low_p = min(open_p, close_p) - 1.0
            vol = 10 + (i % 7)
            out.append((ts, open_p, high_p, low_p, close_p, vol))
        return out


def inject_fake_ccxt():
    fake = types.ModuleType("ccxt")

    def binance_factory(cfg=None):
        return FakeExchange()

    fake.binance = binance_factory
    sys.modules["ccxt"] = fake


def main():
    inject_fake_ccxt()
    # call trainer programmatically
    from ai_engine.train_and_save import train_and_save

    result = train_and_save(symbols=["BTCUSDT"], limit=120, use_live_data=True, backtest=False, write_report=False)
    print("Smoke run result:")
    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
