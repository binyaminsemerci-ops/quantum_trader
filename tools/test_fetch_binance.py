#!/usr/bin/env python3
"""Small helper to test live OHLCV fetching via backend.routes.external_data.binance_ohlcv

Run from repo root: python tools/test_fetch_binance.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# show where we're running
print("CWD:", os.getcwd())
print("Python:", sys.executable)
print("ENABLE_LIVE_MARKET_DATA:", os.environ.get("ENABLE_LIVE_MARKET_DATA"))

from backend.routes.external_data import binance_ohlcv


def main():
    symbol = os.environ.get("TEST_SYMBOL", "BTCUSDT")
    limit = int(os.environ.get("TEST_LIMIT", "5"))
    print(f"Fetching {limit} candles for {symbol} via binance_ohlcv()...\n")
    payload = asyncio.run(binance_ohlcv(symbol=symbol, limit=limit))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
