#!/usr/bin/env python3
"""Test the live feature builder using Binance live candles."""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, '.')

from ai_engine.data.live_features import fetch_features_for_sklearn


def main():
    symbol = os.environ.get('TEST_SYMBOL', 'BTCUSDT')
    print('Symbol:', symbol)
    X, y, names = fetch_features_for_sklearn(symbol, limit=200, lags=5, horizon=1)
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('feature names:', names)
    print('first row X:', X[0].tolist())
    print('first y:', float(y[0]))


if __name__ == '__main__':
    os.environ['ENABLE_LIVE_MARKET_DATA'] = os.environ.get('ENABLE_LIVE_MARKET_DATA', '1')
    main()
