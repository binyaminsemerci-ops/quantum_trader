#!/usr/bin/env python3
"""Test XGBoost model predictions"""
import sys
sys.path.insert(0, '/app')
import pickle
import ccxt
import pandas as pd
import numpy as np
from ai_engine.agents.xgb_agent import XGBAgent

# Test with live data
print("Testing AI model with live data...")

exchange = ccxt.binance()
symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]

agent = XGBAgent()

for symbol in symbols:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        prediction = agent.predict_for_symbol(df)
        
        print(f"\n{symbol}:")
        print(f"  Action: {prediction['action']}")
        print(f"  Score: {prediction['score']:.4f}")
        print(f"  Confidence: {prediction['confidence']:.4f}")
        print(f"  Model: {prediction['model']}")
        
    except Exception as e:
        print(f"\n{symbol}: ERROR - {e}")

print("\nDone!")
